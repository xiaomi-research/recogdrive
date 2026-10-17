import copy
import json
import random
import sys
import tempfile
import unittest
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from recogdrive.data.mixture import MixtureDataset
from recogdrive.data.navsim import NavsimSamples
from recogdrive.data.registry import TaskBatchSampler, collate
from recogdrive.distributed.checkpoint import DONE_FILE, ResumeCheckpointer, export_model_state
from recogdrive.eval.registry import snapshot
from recogdrive.train.args import TrainArgs
from recogdrive.train.trainer import Trainer, _seed_everything


class LocalContext:
    rank = 0
    world_size = 1
    is_main = True
    device = torch.device("cpu")

    def gloo_group(self):
        return dist.group.WORLD if dist.is_initialized() else None

    def barrier(self):
        if dist.is_initialized():
            dist.barrier()

    def all_reduce_mean(self, value):
        return value

    def all_reduce_sum(self, value):
        return value


class GlooContext:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

    def gloo_group(self):
        return dist.group.WORLD

    def barrier(self):
        dist.barrier(group=dist.group.WORLD)


def _two_rank_checkpoint_worker(rank, root, rendezvous):
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    ctx = GlooContext(rank, 2)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    checkpointer = ResumeCheckpointer(Path(root) / "success", ctx, async_save=True)
    assert checkpointer.process_group is not dist.group.WORLD
    checkpointer.save(1, model, optimizer, {"epoch": 1})
    gathered = [None, None]
    dist.all_gather_object(gathered, rank, group=dist.group.WORLD)
    assert gathered == [0, 1]
    checkpointer.wait()
    dist.barrier()
    if rank == 0:
        assert (Path(root) / "success" / "step_00000001" / DONE_FILE).is_file()

    interleaved = ResumeCheckpointer(Path(root) / "interleaved", ctx, async_save=True)
    pending = Future()
    if rank == 0:
        pending.set_result(None)
    with patch("torch.distributed.checkpoint.async_save", return_value=pending), \
            patch("recogdrive.distributed.checkpoint.training_state", return_value={}):
        interleaved.save(2, model, optimizer, {"epoch": 2})
        gathered = [None, None]
        dist.all_gather_object(gathered, rank, group=dist.group.WORLD)
        assert gathered == [0, 1]
        if rank == 1:
            pending.set_result(None)
        interleaved.wait()
    dist.barrier()
    if rank == 0:
        assert (Path(root) / "interleaved" / "step_00000002" / DONE_FILE).is_file()

    failed = ResumeCheckpointer(Path(root) / "failure", ctx, async_save=True)
    future = Future()
    if rank == 0:
        future.set_result(None)
    else:
        future.set_exception(RuntimeError("rank-local write failed"))
    with patch("torch.distributed.checkpoint.async_save", return_value=future), \
            patch("recogdrive.distributed.checkpoint.training_state", return_value={}):
        failed.save(3, model, optimizer, {"epoch": 3})
        gathered = [None, None]
        dist.all_gather_object(gathered, rank, group=dist.group.WORLD)
        assert gathered == [0, 1]
        try:
            failed.wait()
        except RuntimeError:
            pass
        else:
            raise AssertionError("a rank-local async failure was not reported")
    dist.barrier()
    assert not (Path(root) / "failure" / "step_00000003" / DONE_FILE).exists()
    dist.destroy_process_group()


class TinyModel(nn.Module):
    def __init__(self, random_loss=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.random_loss = random_loss

    def forward(self, features, targets, tokens):
        prediction = self.weight * features["x"]
        if self.random_loss:
            prediction = prediction + torch.rand_like(prediction) * 0.1
        return SimpleNamespace(loss=(prediction - targets["y"]).square().mean())


class _ValidationSource(Dataset):
    def __init__(self, task, rows, name):
        self.task, self.rows, self.name = task, rows, name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.task == "vqa":
            labels, token_losses = self.rows[index]
            features = {"task": self.task, "labels": torch.tensor(labels),
                        "token_loss": torch.tensor(token_losses, dtype=torch.float32)}
        else:
            features = {"task": self.task, "sample_loss": torch.tensor(self.rows[index], dtype=torch.float32)}
        return features, {}, f"{self.name}-{index}"


class ResumeCheckpointerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owns_process_group = not dist.is_initialized()
        cls.rendezvous = tempfile.TemporaryDirectory() if cls.owns_process_group else None
        if cls.owns_process_group:
            store = Path(cls.rendezvous.name) / "dist-init"
            dist.init_process_group("gloo", init_method=f"file://{store.as_posix()}", rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        if cls.owns_process_group:
            dist.destroy_process_group()
            cls.rendezvous.cleanup()

    def test_failed_async_save_does_not_publish_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpointer = ResumeCheckpointer(Path(directory), LocalContext())
            future = Future()
            future.set_exception(RuntimeError("write failed"))
            model = nn.Linear(1, 1)
            with patch("torch.distributed.checkpoint.async_save", return_value=future), \
                    patch("recogdrive.distributed.checkpoint.training_state", return_value={}):
                checkpointer.save(1, model, torch.optim.SGD(model.parameters(), lr=0.1), {})
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    checkpointer.wait()
            self.assertFalse((Path(directory) / "step_00000001" / DONE_FILE).exists())

    def test_empty_training_batches_cannot_export_an_untrained_success(self):
        instance = Trainer.__new__(Trainer)
        instance.cfg = OmegaConf.create({"dataloader": {"params": {"batch_size": 2, "num_workers": 0,
                                                                      "drop_last": True}}})
        instance.args = TrainArgs(skip_validation=True)
        instance.ctx = LocalContext()
        instance.agent = object()
        with patch("recogdrive.train.trainer.build_split", return_value=[({}, {}, "one")]):
            with self.assertRaisesRegex(ValueError, "training loader has no batches"):
                instance.build_data()

    def test_configured_seed_covers_python_numpy_and_torch(self):
        _seed_everything(71)
        first = (random.random(), np.random.random(), torch.rand(()).item())
        _seed_everything(71)
        second = (random.random(), np.random.random(), torch.rand(()).item())
        self.assertEqual(first, second)

    def test_ddp_task_specific_unused_parameters_fail_before_wrapping(self):
        agent_type = type("ReCogDriveAgent", (), {"__module__": "recogdrive.adapters.navsim.agent"})
        agent = agent_type()
        agent.grpo = False
        agent.train_backbone = True
        instance = Trainer.__new__(Trainer)
        instance.cfg = OmegaConf.create({"mixture_batch_by_task": True})
        instance.args = TrainArgs(strategy="ddp", find_unused_parameters=False)
        instance.agent = agent
        with self.assertRaisesRegex(ValueError, "find_unused_parameters=true"):
            instance.validate_train_data(SimpleNamespace(sample_tasks=["planning"]))

        agent.train_backbone = False
        agent.vlm_trainable = True  # built-in ReCogDrive LoRA VQA leaves the planner head unused
        with self.assertRaisesRegex(ValueError, "find_unused_parameters=true"):
            instance.validate_train_data(SimpleNamespace(sample_tasks=["vqa"]))

        plugin = SimpleNamespace(grpo=False)
        instance.agent = plugin
        instance.args = TrainArgs(strategy="ddp", find_unused_parameters=False)
        instance.validate_train_data(SimpleNamespace(task="vqa"))
        with self.assertRaisesRegex(ValueError, "find_unused_parameters=true"):
            instance.validate_train_data(SimpleNamespace(sample_tasks=["planning", "vqa"]))

    def test_builtin_pdm_grpo_accepts_only_navsim_sources(self):
        module_name = "recogdrive.adapters.navsim.reward"
        module = ModuleType(module_name)
        pdm_type = type("PDMReward", (), {"__module__": module_name})
        module.PDMReward = pdm_type
        agent = SimpleNamespace(grpo=True, action_head=SimpleNamespace(reward_fn=pdm_type()))
        instance = Trainer.__new__(Trainer)
        instance.cfg = OmegaConf.create({"mixture_batch_by_task": True})
        instance.args = TrainArgs(strategy="fsdp")
        instance.agent = agent

        class NavsimSource(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {}, {}

        with patch.dict(sys.modules, {module_name: module}):
            instance.validate_train_data(NavsimSamples(NavsimSource(), 1))
            navsim_mixture = MixtureDataset([NavsimSamples(NavsimSource(), 1),
                                             NavsimSamples(NavsimSource(), 1)], [1.0, 2.0], True)
            instance.validate_train_data(navsim_mixture)
            with self.assertRaisesRegex(ValueError, "NAVSIM-only"):
                instance.validate_train_data(SimpleNamespace(sample_tasks=["planning", "vqa"]))

        custom_reward_agent = SimpleNamespace(grpo=True, action_head=SimpleNamespace(reward_fn=object()))
        instance.agent = custom_reward_agent
        instance.validate_train_data(SimpleNamespace(sample_tasks=["planning", "vqa"]))

    def test_validation_uses_source_weights_sample_counts_and_vqa_tokens(self):
        dataset = MixtureDataset(
            [_ValidationSource("planning", [1.0, 3.0, 5.0], "planning"),
             _ValidationSource("vqa", [([1], [2.0]), ([1, 1, 1], [4.0, 6.0, 8.0])], "vqa")],
            batch_by_task=True,
            source_weights=[1.0, 3.0],
        )

        class EvalModel:
            def eval(self):
                return self

            def __call__(self, features, targets, tokens):
                return features

        class EvalAgent:
            @staticmethod
            def compute_loss(features, targets, predictions):
                if features["task"][0] == "vqa":
                    count = (features["labels"] != -100).sum()
                    return features["token_loss"].sum() / count
                return features["sample_loss"].mean()

        logs = []
        instance = Trainer.__new__(Trainer)
        instance.model = EvalModel()
        instance.agent = EvalAgent()
        instance.ctx = LocalContext()
        instance.val_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=TaskBatchSampler(dataset.sample_tasks, batch_size=2, rank=0, world=1),
            collate_fn=collate,
        )
        instance.ema = None
        instance.monitor = SimpleNamespace(log=lambda metrics, step, epoch: logs.append(metrics))
        instance.global_step = instance.epoch = 0
        value = instance.validate(use_grpo=False)
        self.assertAlmostEqual(logs[0]["val/task/planning/loss"], 3.0)
        self.assertAlmostEqual(logs[0]["val/task/vqa/loss"], 5.0)
        self.assertAlmostEqual(value, 4.5)

    def test_non_best_epoch_is_snapshotted_before_topk_prunes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "epoch_0000_val_0.100000.ckpt"
            previous.write_bytes(b"previous best")
            (root / "best_checkpoints.json").write_text(
                '[{"path":"epoch_0000_val_0.100000.ckpt","val_loss":0.1}]'
            )
            instance = Trainer.__new__(Trainer)
            instance.output_dir = root
            instance.args = TrainArgs(top_k=1, max_epochs=2, val_every_n_epochs=1)
            instance.ctx = LocalContext()
            instance.ema = None
            instance.model = object()
            instance.global_step = 10
            instance.best_val = 0.1
            instance.val_loader = object()
            instance.validate = lambda use_grpo: 0.9
            instance.monitor = SimpleNamespace(log=lambda *args: None, plot=lambda: None)
            instance.write_ckpt = lambda state, path: path.write_bytes(b"new checkpoint")
            snapshots = []

            def evaluate(epoch, checkpoint):
                saved = snapshot(checkpoint, root / "eval" / f"epoch_{epoch + 1:04d}")
                snapshots.append(saved)

            instance.evaluate = evaluate
            instance.save_resume = lambda epoch, batch: None

            with patch("recogdrive.train.trainer.export_model_state", return_value={"weight": torch.ones(1)}):
                instance.end_epoch(0, 1.0, use_grpo=False)

            current = root / "epoch_0001_val_0.900000.ckpt"
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].read_bytes(), b"new checkpoint")
            self.assertFalse(current.exists())
            self.assertTrue(previous.is_file())
            self.assertEqual(json.loads((root / "best_checkpoints.json").read_text()),
                             [{"path": previous.name, "val_loss": 0.1}])

    def test_perf_log_counts_unique_input_batches_separately_from_optimizer_steps(self):
        metrics = []
        instance = Trainer.__new__(Trainer)
        instance.args = TrainArgs(log_every=1)
        instance.ctx = LocalContext()
        instance.optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}])
        instance.global_batch = 999
        instance.global_step = 3
        instance.monitor = SimpleNamespace(log=lambda values, step, epoch: metrics.append(values))
        window = {"time": 10.0, "wait": 0.0, "step": 1, "input_samples": 6}
        with patch("recogdrive.train.trainer.time.perf_counter", return_value=12.0), \
                patch("torch.cuda.is_available", return_value=False):
            instance.log_train(0, SimpleNamespace(), torch.tensor(1.0), None, window,
                               SimpleNamespace(wait_s=0.0))
        self.assertEqual(metrics[0]["perf/samples_per_s"], 3.0)
        self.assertEqual(metrics[0]["perf/optimizer_steps_s"], 1.0)
        self.assertEqual(window["input_samples"], 0)

    def test_full_vlm_export_saves_weights_tokenizer_and_processor(self):
        calls = []

        class FakeModel:
            def save_pretrained(self, directory, **kwargs):
                calls.append((Path(directory), kwargs))
                (Path(directory) / "model.safetensors").write_bytes(b"hf weights")

        class FakeConfig:
            def __init__(self, filename, payload):
                self.filename, self.payload = filename, payload

            def save_pretrained(self, directory):
                path = Path(directory) / self.filename
                path.write_text(json.dumps(self.payload))
                calls.append(path)

        backbone = SimpleNamespace(
            model=FakeModel(),
            tokenizer=FakeConfig("tokenizer_config.json", {"tokenizer": "saved"}),
            processor=FakeConfig("preprocessor_config.json", {"processor": "saved"}),
        )
        instance = Trainer.__new__(Trainer)
        instance.agent = SimpleNamespace(backbone=backbone)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.ckpt"
            state = {"layer.weight": torch.tensor([1.0, 2.0])}
            instance.save_vlm(state, checkpoint)

            out_dir, kwargs = calls[0]
            self.assertEqual(out_dir, checkpoint.with_name("last_vlm"))
            self.assertEqual(kwargs, {"state_dict": state, "safe_serialization": True, "max_shard_size": "10GB"})
            self.assertEqual(json.loads((out_dir / "preprocessor_config.json").read_text()), {"processor": "saved"})
            self.assertEqual(json.loads((out_dir / "tokenizer_config.json").read_text()), {"tokenizer": "saved"})
            self.assertEqual((out_dir / "model.safetensors").read_bytes(), b"hf weights")

    def test_custom_cnn_backbone_roundtrips_entire_checkpoint_state(self):
        instance = Trainer.__new__(Trainer)
        instance.agent = SimpleNamespace(backbone=SimpleNamespace(model=nn.Sequential(nn.Conv2d(3, 4, 1))))
        state = {
            "backbone.model.0.weight": torch.randn(4, 3, 1, 1),
            "backbone.model.0.bias": torch.randn(4),
            "backbone.projection.weight": torch.randn(2, 4),
            "action_head.weight": torch.randn(2, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.ckpt"
            instance.write_ckpt(state, checkpoint)
            restored = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            self.assertEqual(set(restored), set(state))
            for key in state:
                torch.testing.assert_close(restored[key], state[key])
            cnn_state = {key.removeprefix("backbone.model."): value for key, value in restored.items()
                         if key.startswith("backbone.model.")}
            instance.agent.backbone.model.load_state_dict(cnn_state)
            torch.testing.assert_close(instance.agent.backbone.model[0].weight, state["backbone.model.0.weight"])
            self.assertFalse(checkpoint.with_name("policy_vlm").exists())
            self.assertFalse(torch.load(checkpoint, map_location="cpu", weights_only=False)["vlm_export"])

    def test_hf_checkpoint_splits_only_model_weights_and_keeps_adapters(self):
        exported = []

        class FakeHFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(1, 1)

            def save_pretrained(self, *args, **kwargs):
                pass

        class FakePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.model = FakeHFModel()

        instance = Trainer.__new__(Trainer)
        instance.agent = FakePolicy()
        instance.save_vlm = lambda weights, checkpoint: exported.append(weights)
        state = {
            "action_head.weight": torch.tensor([1.0]),
            "backbone.projection.weight": torch.tensor([2.0]),
            "backbone.model.embed_tokens.weight": torch.tensor([3.0]),
            "backbone.model.layers.0.q_proj.lora_A.default.weight": torch.tensor([4.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.ckpt"
            instance.write_ckpt(state, checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            self.assertEqual(set(saved), {
                "action_head.weight", "backbone.projection.weight",
                "backbone.model.layers.0.q_proj.lora_A.default.weight",
            })
            self.assertEqual(set(exported[0]), {"embed_tokens.weight"})
            torch.testing.assert_close(exported[0]["embed_tokens.weight"], state["backbone.model.embed_tokens.weight"])
            self.assertTrue(torch.load(checkpoint, map_location="cpu", weights_only=False)["vlm_export"])

    def test_frozen_hf_lora_keeps_adapter_without_creating_partial_export(self):
        class FakeHFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Parameter(torch.ones(1), requires_grad=False)
                self.lora_A = nn.Parameter(torch.ones(1), requires_grad=True)

            def save_pretrained(self, *args, **kwargs):
                pass

        class FakePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.model = FakeHFModel()

        instance = Trainer.__new__(Trainer)
        instance.agent = FakePolicy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.ckpt"
            stale_export = checkpoint.with_name("policy_vlm")
            stale_export.mkdir()
            sentinel = stale_export / "previous-run.txt"
            sentinel.write_text("preserve existing output")
            state = {
                "backbone.model.base": torch.tensor([2.0]),
                "backbone.model.lora_A": torch.tensor([3.0]),
                "action_head.weight": torch.tensor([4.0]),
            }
            instance.write_ckpt(state, checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(set(saved["state_dict"]), {"backbone.model.lora_A", "action_head.weight"})
            self.assertFalse(saved["vlm_export"])
            self.assertEqual(sentinel.read_text(), "preserve existing output")

    def test_dcp_rl_export_drops_teacher_aliases_for_non_grpo_inference(self):
        class Planner(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = nn.Linear(2, 2)
                self.old_policy = copy.deepcopy(self.actor)
                for parameter in self.old_policy.parameters():
                    parameter.requires_grad_(False)
                self.old_policy.register_buffer("teacher_cache", torch.ones(1))
                self.ref_policy = self.old_policy

        class ReCogDriveAgent(nn.Module):
            __module__ = "recogdrive.adapters.navsim.agent"

            def __init__(self):
                super().__init__()
                self.action_head = Planner()

        class InferenceAgent(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_head = nn.Module()
                self.action_head.actor = nn.Linear(2, 2)

        training_agent = ReCogDriveAgent()
        state = export_model_state(training_agent)
        teacher_keys = {key for key in state if key.startswith(
            ("action_head.old_policy.", "action_head.ref_policy.")
        )}
        self.assertTrue(teacher_keys, "the DCP state fixture must contain frozen teacher aliases or buffers")

        instance = Trainer.__new__(Trainer)
        instance.agent = training_agent
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "rl.ckpt"
            instance.write_ckpt(state, checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertFalse(payload["vlm_export"])
            self.assertFalse(teacher_keys & set(payload["state_dict"]))
            inference_agent = InferenceAgent()
            from recogdrive.infer import load_weights

            load_weights(inference_agent, checkpoint)

        plugin = SimpleNamespace(backbone=None)
        instance.agent = plugin
        with tempfile.TemporaryDirectory() as directory:
            plugin_checkpoint = Path(directory) / "plugin.ckpt"
            instance.write_ckpt({"action_head.old_policy.weight": torch.ones(1),
                                 "action_head.ref_policy.weight": torch.ones(1)}, plugin_checkpoint)
            plugin_state = torch.load(plugin_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            self.assertEqual(set(plugin_state), {"action_head.old_policy.weight", "action_head.ref_policy.weight"})

    def test_two_rank_async_save_and_foreground_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(_two_rank_checkpoint_worker, args=(str(root), str(root / "rendezvous")), nprocs=2, join=True)

    def test_resume_matches_uninterrupted_training_and_flushes_tail(self):
        def batches():
            return [({"x": torch.ones(1)}, {"y": torch.zeros(1)}, None) for _ in range(3)]

        def trainer(root, random_loss, max_steps=0):
            instance = Trainer.__new__(Trainer)
            instance.args = TrainArgs(grad_accum=2, max_epochs=1, max_steps=max_steps, save_every_n_steps=0,
                                      log_every=1000, grad_clip=0.0)
            instance.ctx = LocalContext()
            instance.model = TinyModel(random_loss)
            instance.agent = instance.model
            instance.agent.grpo = False
            instance.optimizer = torch.optim.SGD(instance.model.parameters(), lr=0.1)
            instance.lr_scheduler = None
            instance.lr_interval = "epoch"
            instance.ema = None
            instance.clip_params = [instance.model.weight]
            instance.train_loader = batches()
            instance.checkpointer = ResumeCheckpointer(Path(root), instance.ctx, async_save=False)
            instance.start_epoch = instance.start_batch = instance.global_step = instance.micro = 0
            instance.best_val = float("inf")
            instance.resume_rng = instance.resume_epoch_rng = None
            instance.end_epoch = lambda *args: None
            return instance

        with tempfile.TemporaryDirectory() as directory:
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            baseline = trainer(Path(directory) / "baseline", random_loss=True)
            baseline.train_loop()

            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            interrupted = trainer(Path(directory) / "resume", random_loss=True, max_steps=1)
            interrupted.train_loop()
            self.assertEqual(interrupted.global_step, 1)

            resumed = trainer(Path(directory) / "resume", random_loss=True)
            resumed.resume()
            self.assertEqual(resumed.global_step, 1)
            self.assertEqual(resumed.start_batch, 2)
            resumed.train_loop()
            self.assertEqual(resumed.global_step, baseline.global_step)
            torch.testing.assert_close(resumed.model.weight, baseline.model.weight, rtol=0, atol=0)

            tail = trainer(Path(directory) / "tail", random_loss=False)
            tail.train_loop()
            self.assertEqual(tail.global_step, 2)
            torch.testing.assert_close(tail.model.weight, torch.tensor(0.64), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
