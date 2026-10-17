import io
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from omegaconf import OmegaConf
from PIL import Image

from recogdrive.data.waymoe2e import WaymoE2ECacheOnlyDataset, convert, e2ed_frame_class, waymoe2e_loader


def _frame(name, timestamp, rated=False):
    cls = e2ed_frame_class()
    sample = cls()
    sample.frame.context.name = name
    sample.frame.timestamp_micros = timestamp
    image = Image.new("RGB", (32, 18), (12, 34, 56))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    sample.frame.images.add(name=1, image=encoded.getvalue())
    sample.past_states.pos_x.extend(float(i - 15) for i in range(16))
    sample.past_states.pos_y.extend(0.0 for _ in range(16))
    sample.future_states.pos_x.extend(float(i + 1) for i in range(20))
    sample.future_states.pos_y.extend(0.0 for _ in range(20))
    sample.intent = 1
    if rated:
        rater = sample.preference_trajectories.add()
        rater.pos_x.extend(float(i + 1) for i in range(20))
        rater.pos_y.extend(0.0 for _ in range(20))
        rater.preference_score = 5.0
    return sample.SerializeToString()


def _write_tfrecord(path: Path, records):
    with path.open("wb") as stream:
        for record in records:
            stream.write(struct.pack("<Q", len(record)))
            stream.write(b"\0" * 4)  # CRC fields are not read by this converter.
            stream.write(record)
            stream.write(b"\0" * 4)


class WaymoCacheTest(unittest.TestCase):
    def test_context_name_is_preserved_as_frame_token_through_cache_and_eval(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "frames.tfrecord"
            _write_tfrecord(record, [_frame("segment_01-frame_001", 1_000_000, rated=True),
                                     _frame("segment_01-frame_002", 1_250_000)])
            self.assertEqual(convert([record], root / "cache", "val"), 2)
            dataset = WaymoE2ECacheOnlyDataset(root / "cache", "val", future=8)
            items = [dataset[index] for index in range(len(dataset))]

            self.assertEqual({item[2] for item in items}, {"segment_01-frame_001", "segment_01-frame_002"})
            self.assertEqual({path.name for path in dataset.tokens}, {item[2] for item in items})
            labeled = [dataset.tokens[index].name for index in dataset.eval_indices()]
            self.assertEqual(labeled, ["segment_01-frame_001"])
            self.assertEqual(len(dataset.eval_info("segment_01-frame_001")["rater_trajectories"]), 1)

    def test_duplicate_official_frame_name_fails_before_overwriting(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "duplicate.tfrecord"
            _write_tfrecord(record, [_frame("same-frame", 1_000_000, rated=True),
                                     _frame("same-frame", 1_250_000)])
            with self.assertRaisesRegex(ValueError, "duplicate WOD-E2E frame context name"):
                convert([record], root / "cache", "training")
            dataset = WaymoE2ECacheOnlyDataset(root / "cache", "training")
            self.assertEqual(dataset.eval_indices(), [0])
            self.assertEqual(len(dataset.eval_info("same-frame")["rater_trajectories"]), 1)

    def test_test_split_is_rejected_instead_of_aliasing_validation(self):
        cfg = OmegaConf.create({"cache_path": "/unused", "waymoe2e_train_split": "training",
                                "waymoe2e_val_split": "val"})
        agent = SimpleNamespace(trajectory_sampling=SimpleNamespace(num_poses=8, interval_length=0.5),
                                get_target_builders=lambda: [])
        with self.assertRaisesRegex(ValueError, "supports train/val splits"):
            waymoe2e_loader(cfg, agent, "test")


if __name__ == "__main__":
    unittest.main()
