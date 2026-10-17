import gzip
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from recogdrive.adapters.waymoe2e.reward import WaymoADEReward, WaymoRFSReward
from recogdrive.data.waymoe2e import WaymoE2ECacheOnlyDataset, waymoe2e_loader
from recogdrive.eval.openloop import pad_raters, rater_feedback_score, to_4hz
from recogdrive.models.recogdrive.diffusion_planner import ReCogDriveDiffusionPlanner


def _save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as stream:
        pickle.dump(value, stream)


def _labels(lateral_offset=0.0):
    t = np.arange(1, 21, dtype=np.float64) * 0.25
    return {
        "rater_trajectories": [np.column_stack([t, np.full(20, lateral_offset)])],
        "rater_scores": np.array([9.0]),
        "init_speed": 5.0,
    }


def _candidates(lateral_offset=0.0):
    x = torch.arange(1, 11, dtype=torch.float32) * 0.5
    y = torch.full_like(x, lateral_offset)
    yaw = torch.zeros_like(x)
    return torch.stack([x, y, yaw], dim=-1).unsqueeze(0)


def _waypoints(lateral_offset=0.0, poses=10):
    x = np.arange(1, poses + 1, dtype=np.float32) * 0.5
    y = np.full(poses, lateral_offset, dtype=np.float32)
    yaw = np.zeros(poses, dtype=np.float32)
    return {"trajectory": np.column_stack([x, y, yaw])}


def _grpo_rollout_grad(reward, candidates):
    reward.prefetch(["frame_a"], per_scene=2)
    observed_tokens = []

    def record_reward(sampled, tokens):
        observed_tokens.append(list(tokens))
        return reward(sampled, tokens)

    record_reward.prefetch = reward.prefetch
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    nn.Module.__init__(planner)
    planner.config = SimpleNamespace(sampling_method="ddim")
    planner.grpo_objective = "main"
    planner.clip_eps = 0.2
    planner.bc_coeff = 0.0
    planner.kl_coef = 0.0
    planner.sample_time = 2
    planner.gamma_denoising = 0.6
    planner.clip_advantage_lower_quantile = 0.0
    planner.clip_advantage_upper_quantile = 1.0
    planner.test_policy_scale = nn.Parameter(torch.tensor(0.25))

    chains = torch.zeros(2, 3, 10, 3)
    chains[0].fill_(1.0)
    chains[1].fill_(2.0)

    def sample_chain(self, vl, his, status, **kwargs):
        return chains, candidates

    def get_logprobs(self, vl, his, status, candidate_chains, deterministic=False):
        coefficient = candidate_chains[:, 0, 0, 0]
        per_candidate_step = self.test_policy_scale * coefficient[:, None].expand(-1, 2)
        return per_candidate_step.reshape(-1, 1, 1)

    planner.sample_chain = MethodType(sample_chain, planner)
    planner.get_logprobs = MethodType(get_logprobs, planner)
    planner.reward_fn = record_reward
    action_input = SimpleNamespace(his_traj=torch.zeros(1, 4, 3), status_feature=torch.zeros(1, 8))
    rollout = planner.collect_grpo_rollout(torch.zeros(1, 1, 1), action_input, ["frame_a"])
    result = planner.grpo_loss_from_rollout(rollout)
    result["loss"].backward()
    return observed_tokens, rollout, planner.test_policy_scale.grad


class WaymoRFSRewardTest(unittest.TestCase):
    def test_scores_each_candidate_with_the_existing_openloop_rfs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _save(root / "training" / "frame_a" / "eval.gz", _labels())
            reward = WaymoRFSReward(str(root), "training")
            candidates = torch.cat([_candidates(0.0), _candidates(50.0)])
            actual = reward(candidates, ["frame_a", "frame_a"])

            raters, scores = pad_raters(_labels()["rater_trajectories"], _labels()["rater_scores"], 20)
            expected = [
                rater_feedback_score(to_4hz(candidate.numpy(), 5.0), raters, scores, 5.0)
                for candidate in candidates
            ]
            torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
            self.assertEqual(actual.shape, (2,))
            self.assertGreater(float(actual[0]), float(actual[1]))

    def test_waymo_rfs_drives_grpo_rollout_and_policy_gradient(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _save(root / "training" / "frame_a" / "eval.gz", _labels())
            reward = WaymoRFSReward(str(root), "training")
            candidates = torch.cat([_candidates(0.0), _candidates(50.0)])
            observed_tokens, rollout, gradient = _grpo_rollout_grad(reward, candidates)

            self.assertEqual(observed_tokens, [["frame_a", "frame_a"]])
            self.assertAlmostEqual(float(rollout.reward_mean), 6.5, places=5)
            self.assertGreater(float(rollout.adv_weighted[0]), 0)
            self.assertLess(float(rollout.adv_weighted[-1]), 0)
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient))
            self.assertNotEqual(float(gradient), 0.0)

    def test_public_train_waymo_ade_reward_and_grpo_gradient(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frame_dir = root / "training" / "frame_a"
            (frame_dir / "features.gz").parent.mkdir(parents=True)
            (frame_dir / "features.gz").write_bytes(b"features")
            _save(frame_dir / "targets.gz", _waypoints())
            reward = WaymoADEReward(str(root), "training")

            candidates = torch.cat([_candidates(0.0), _candidates(3.0)])
            actual = reward(candidates, ["frame_a", "frame_a"])
            torch.testing.assert_close(actual, torch.tensor([0.0, -3.0]), rtol=0, atol=1e-6)

            observed_tokens, rollout, gradient = _grpo_rollout_grad(
                reward, torch.cat([_candidates(0.0), _candidates(50.0)])
            )
            self.assertEqual(observed_tokens, [["frame_a", "frame_a"]])
            self.assertAlmostEqual(float(rollout.reward_mean), -25.0, places=5)
            self.assertGreater(float(rollout.adv_weighted[0]), 0)
            self.assertLess(float(rollout.adv_weighted[-1]), 0)
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient))
            self.assertNotEqual(float(gradient), 0.0)

    def test_waymo_ade_rejects_validation_missing_and_nonfinite_targets(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frame_dir = root / "training" / "frame_a"
            frame_dir.mkdir(parents=True)
            (frame_dir / "features.gz").write_bytes(b"features")
            _save(frame_dir / "targets.gz", _waypoints())

            with self.assertRaisesRegex(ValueError, "cannot read validation/test"):
                WaymoADEReward(str(root), "val")
            reward = WaymoADEReward(str(root), "training")
            with self.assertRaisesRegex(FileNotFoundError, "missing logged waypoints"):
                reward(_candidates(), ["frame_missing"])
            with self.assertRaisesRegex(ValueError, "invalid WOD-E2E training frame token"):
                reward(_candidates(), ["../outside"])
            with self.assertRaisesRegex(ValueError, "finite floating-point"):
                reward(_candidates(float("nan")), ["frame_a"])

            short_dir = root / "training" / "frame_short"
            short_dir.mkdir()
            (short_dir / "features.gz").write_bytes(b"features")
            _save(short_dir / "targets.gz", _waypoints(poses=8))
            with self.assertRaisesRegex(ValueError, "expected at least \(10, 2\)"):
                reward(_candidates(), ["frame_short"])

            broken = _waypoints()
            broken["trajectory"][0, 0] = np.nan
            _save(frame_dir / "targets.gz", broken)
            with self.assertRaisesRegex(ValueError, "non-finite values"):
                reward(_candidates(), ["frame_a"])

    def test_rejects_validation_split_and_missing_frame_labels(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _save(root / "training" / "frame_a" / "eval.gz", _labels())
            with self.assertRaisesRegex(ValueError, "cannot read validation/test"):
                WaymoRFSReward(str(root), "val")
            reward = WaymoRFSReward(str(root), "training")
            with self.assertRaisesRegex(FileNotFoundError, "missing rater labels"):
                reward(_candidates(), ["frame_missing"])
            with self.assertRaisesRegex(ValueError, "invalid WOD-E2E training frame token"):
                reward(_candidates(), ["../outside"])

    def test_rejects_bad_label_values_and_candidate_horizon(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            broken = _labels()
            broken["rater_trajectories"] = [np.zeros((20, 3))]
            _save(root / "training" / "frame_a" / "eval.gz", broken)
            reward = WaymoRFSReward(str(root), "training")
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                reward(_candidates(), ["frame_a"])

            broken["rater_trajectories"] = [np.zeros((20, 2))]
            broken["rater_scores"] = np.array([5.0, 6.0])
            _save(root / "training" / "frame_a" / "eval.gz", broken)
            with self.assertRaisesRegex(ValueError, "count/score shape mismatch"):
                reward(_candidates(), ["frame_a"])

            broken = _labels()
            broken["rater_trajectories"] = [np.array([[np.nan, 0.0]])]
            _save(root / "training" / "frame_a" / "eval.gz", broken)
            with self.assertRaisesRegex(ValueError, "non-finite values"):
                reward(_candidates(), ["frame_a"])

            _save(root / "training" / "frame_a" / "eval.gz", _labels())
            short = torch.zeros((1, 8, 3))
            with self.assertRaisesRegex(ValueError, "requires a 5 s horizon"):
                reward(short, ["frame_a"])

    def test_loader_can_filter_only_labelled_training_frames(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("training", "val"):
                for token in ("labelled", "unlabelled"):
                    frame_dir = root / split / token
                    frame_dir.mkdir(parents=True)
                    (frame_dir / "features.gz").write_bytes(b"features")
                    (frame_dir / "targets.gz").write_bytes(b"targets")
                _save(root / split / "labelled" / "eval.gz", _labels())

            cfg = OmegaConf.create({
                "cache_path": str(root),
                "waymoe2e_train_split": "training",
                "waymoe2e_val_split": "val",
                "waymoe2e_labelled_only": True,
            })
            agent = SimpleNamespace(
                trajectory_sampling=SimpleNamespace(num_poses=10, interval_length=0.5),
                get_target_builders=lambda: [],
            )
            train = waymoe2e_loader(cfg, agent, "train")
            val = waymoe2e_loader(cfg, agent, "val")

            self.assertEqual([path.name for path in train.tokens], ["labelled"])
            self.assertTrue(train.labelled_only)
            self.assertEqual(train.cache_path, root.resolve())
            self.assertEqual(train.split, "training")
            self.assertEqual(len(val), 2)  # the train-only flag does not alter validation or ordinary IL data
            self.assertFalse(val.labelled_only)
            self.assertEqual(val.eval_indices(), [0])
            cfg.waymoe2e_train_split = "val"
            with self.assertRaisesRegex(ValueError, "cannot read validation/test cache split"):
                waymoe2e_loader(cfg, agent, "train")

    def test_labelled_only_training_fails_when_no_labels_exist(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frame_dir = root / "training" / "unlabelled"
            frame_dir.mkdir(parents=True)
            (frame_dir / "features.gz").write_bytes(b"features")
            (frame_dir / "targets.gz").write_bytes(b"targets")
            with self.assertRaisesRegex(ValueError, "no rater-labelled WOD-E2E frames"):
                WaymoE2ECacheOnlyDataset(str(root), "training", future=10, labelled_only=True)


if __name__ == "__main__":
    unittest.main()
