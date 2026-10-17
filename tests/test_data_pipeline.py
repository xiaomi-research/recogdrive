import unittest

import torch
from torch.utils.data import Dataset

from recogdrive.data.mixture import MixtureDataset
from recogdrive.data.navsim import NavsimSamples
from recogdrive.data.registry import WeightedDistributedSampler, collate, make_loader


class _NavsimPair(Dataset):
    tokens = ["scene-1"]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"history": torch.zeros(2)}, {"trajectory": torch.zeros(3, 3)}


class _NavsimCache(_NavsimPair):
    def __getitem__(self, index):
        features, targets = super().__getitem__(index)
        return features, targets, "cache-token"


class _Source(Dataset):
    def __init__(self, size, task, name=None):
        self.size, self.task, self.name = size, task, name or task

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"task": self.task}, {}, f"{self.name}-{index}"


class DataPipelineTest(unittest.TestCase):
    def test_navsim_pair_gets_token_and_short_horizon_fails(self):
        sample = NavsimSamples(_NavsimPair(), num_poses=3)[0]
        self.assertEqual(sample[2], "scene-1")
        self.assertEqual(len(sample), 3)
        self.assertEqual(NavsimSamples(_NavsimCache(), num_poses=3)[0][2], "cache-token")
        with self.assertRaisesRegex(ValueError, "model requires 4"):
            NavsimSamples(_NavsimPair(), num_poses=4)[0]
        class BadTrajectory(_NavsimPair):
            def __getitem__(self, index):
                return {}, {"trajectory": torch.zeros(3)}

        with self.assertRaisesRegex(TypeError, r"shape \(poses, 3\)"):
            NavsimSamples(BadTrajectory(), num_poses=3)[0]

    def test_collate_rejects_mixed_sample_arity(self):
        with self.assertRaisesRegex(ValueError, r"item counts \[2, 3\]"):
            collate([({}, {}), ({}, {}, "scene-1")])
        features, _, _ = collate([
            ({"image_grid_thw": torch.ones(1, 3)}, {}, "a"),
            ({"image_grid_thw": torch.ones(2, 3)}, {}, "b"),
        ])
        self.assertEqual(tuple(features["image_grid_thw"].shape), (3, 3))

    def test_vqa_sequences_are_left_padded(self):
        features, _, _ = collate([
            ({"input_ids": torch.tensor([11, 12]), "attention_mask": torch.tensor([1, 1]),
              "labels": torch.tensor([-100, 12])}, {}, "long"),
            ({"input_ids": torch.tensor([13]), "attention_mask": torch.tensor([1]),
              "labels": torch.tensor([13])}, {}, "short"),
        ])
        self.assertTrue(torch.equal(features["input_ids"], torch.tensor([[11, 12], [0, 13]])))
        self.assertTrue(torch.equal(features["attention_mask"], torch.tensor([[1, 1], [0, 1]])))
        self.assertTrue(torch.equal(features["labels"], torch.tensor([[-100, 12], [-100, 13]])))

    def test_task_batches_match_across_ranks_and_repeat_by_seed(self):
        dataset = MixtureDataset(
            [_Source(2, "planning"), _Source(2, "planning"), _Source(4, "vqa")],
            [1.0, 1.0, 1.0],
            batch_by_task=True,
        )
        samplers = [
            WeightedDistributedSampler(dataset.sample_weights, rank, 2, seed=19, batch_size=2,
                                        task_ids=dataset.sample_tasks)
            for rank in range(2)
        ]
        for sampler in samplers:
            sampler.set_epoch(3)
        self.assertEqual(samplers[0].task_weights.tolist(), [2.0, 1.0])
        self.assertEqual(samplers[0].task_groups["planning"].tolist(), [0, 1, 2, 3])
        first = [list(sampler) for sampler in samplers]
        repeated = []
        for rank in range(2):
            sampler = WeightedDistributedSampler(dataset.sample_weights, rank, 2, seed=19, batch_size=2,
                                                  task_ids=dataset.sample_tasks)
            sampler.set_epoch(3)
            repeated.append(list(sampler))
        self.assertEqual(first, repeated)
        self.assertEqual(len(first[0]), len(first[1]))
        for start in range(0, len(first[0]), 2):
            rank_tasks = [
                {dataset.sample_tasks[index] for index in first[rank][start:start + 2]}
                for rank in range(2)
            ]
            self.assertEqual(rank_tasks[0], rank_tasks[1])
            self.assertEqual(len(rank_tasks[0]), 1)

    def test_drop_last_defaults_to_main_tail_coverage_and_can_be_enabled(self):
        dataset = _Source(5, "planning")
        keep_tail = make_loader(dataset, {"batch_size": 2}, train=True, rank=0, world=1)
        drop_tail = make_loader(dataset, {"batch_size": 2, "drop_last": True}, train=True, rank=0, world=1)
        self.assertEqual([len(tokens) for _, _, tokens in keep_tail], [2, 2, 1])
        self.assertEqual([len(tokens) for _, _, tokens in drop_tail], [2, 2])

    def test_distributed_loader_uses_configured_seed_and_keeps_tail(self):
        dataset = _Source(31, "planning")

        def order(seed):
            loader = make_loader(dataset, {"batch_size": 4}, train=True, rank=0, world=2, seed=seed)
            return [token for _, _, tokens in loader for token in tokens]

        self.assertEqual(order(17), order(17))
        self.assertNotEqual(order(17), order(18))
        self.assertEqual(len(order(17)), 16)

    def test_weighted_sampler_uses_seed_and_ceil_distributed_length(self):
        weights = torch.ones(31, dtype=torch.double)
        first = WeightedDistributedSampler(weights, rank=0, world=4, seed=3)
        repeat = WeightedDistributedSampler(weights, rank=0, world=4, seed=3)
        changed = WeightedDistributedSampler(weights, rank=0, world=4, seed=4)
        self.assertEqual(len(first), 8)
        self.assertEqual(list(first), list(repeat))
        self.assertNotEqual(list(first), list(changed))

    def test_validation_batches_keep_tasks_aligned_and_cover_sources(self):
        dataset = MixtureDataset(
            [_Source(3, "planning", "planning-a"), _Source(4, "planning", "planning-b"),
             _Source(5, "vqa", "vqa")],
            batch_by_task=True,
        )
        self.assertEqual(dataset.task_loss_weights, {"planning": 2 / 3, "vqa": 1 / 3})
        loaders = [make_loader(dataset, {"batch_size": 2}, train=False, rank=rank, world=2) for rank in range(2)]
        batches = [list(loader) for loader in loaders]
        self.assertEqual(len(batches[0]), len(batches[1]))
        self.assertEqual([loader.batch_sampler.valid_batch_sizes for loader in loaders],
                         [[2, 2, 2, 1], [2, 1, 2, 0]])
        for left, right in zip(*batches):
            left_tasks, right_tasks = set(left[0]["task"]), set(right[0]["task"])
            self.assertEqual(left_tasks, right_tasks)
            self.assertEqual(len(left_tasks), 1)
        valid_tokens = []
        for rank, rank_batches in enumerate(batches):
            for (_, _, batch_tokens), valid_count in zip(rank_batches, loaders[rank].batch_sampler.valid_batch_sizes):
                valid_tokens.extend(batch_tokens[:valid_count])
        expected = {f"planning-a-{i}" for i in range(3)} | {f"planning-b-{i}" for i in range(4)} | \
                   {f"vqa-{i}" for i in range(5)}
        self.assertEqual(set(valid_tokens), expected)
        self.assertEqual(len(valid_tokens), len(dataset))


if __name__ == "__main__":
    unittest.main()
