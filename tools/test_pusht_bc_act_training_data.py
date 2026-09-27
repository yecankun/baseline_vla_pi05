"""Synthetic CPU array tests; no LeRobot, AV1, source dataset, or training."""
from fractions import Fraction
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.pusht_bc_act_training_data import (
    FRAME_COUNT, GATE_PINS, NUMERIC_COLUMNS, PushTTrainingData, check_video_frame,
    chunk_indices, episode_end_indices, float32_image_hash, load_training_data,
    split_indices, validate_statistics, verify_gate, verify_image_probes,
    decode_image_cache,
)
from tools.prepare_pusht_bc_act_data import IMAGE_STATS


def fixture():
    n = 22
    images = np.empty((n, 96, 96, 3), dtype=np.uint8)
    for i in range(n):
        images[i].fill(i)
    return dict(images=images, states=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
                actions=np.arange(n * 2, dtype=np.float32).reshape(n, 2) + 100,
                episodes=np.array([0] * 18 + [1] * 4), frames=np.r_[np.arange(18), np.arange(4)],
                split={"train_episodes": [0], "val_episodes": [1], "train_frames": 18, "val_frames": 4},
                stats={"observation.image": IMAGE_STATS,
                       "observation.state": {"mean": [1.0, 2.0], "std": [2.0, 3.0]},
                       "action": {"mean": [5.0, 6.0], "std": [4.0, 5.0]}})


class FakeFrame:
    def __init__(self, pts, time_base, pixels):
        self.pts, self.time_base, self.pixels = pts, time_base, pixels

    def to_ndarray(self, format):
        if format != "rgb24":
            raise AssertionError("unexpected pixel conversion")
        return self.pixels


class FakeContainer:
    def __init__(self, images, fps=Fraction(10), declared_frames=0):
        stream = SimpleNamespace(codec_context=SimpleNamespace(name="av1", width=96, height=96),
                                 average_rate=fps, frames=declared_frames, time_base=Fraction(1, 10))
        self.streams = SimpleNamespace(video=[stream], audio=[])
        self.images = images

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def decode(self, stream):
        for i, pixels in enumerate(self.images):
            yield FakeFrame(i, Fraction(1, 10), pixels)


class TestPushTTrainingData(unittest.TestCase):
    def test_batch_exact_interface_shapes_and_raw_values(self):
        data = PushTTrainingData(**fixture())
        batch = data.batch([0, 19, 0], device="cpu")
        self.assertEqual(set(batch), {"observation", "action", "action_is_pad", "indices"})
        self.assertEqual(set(batch["observation"]), {"observation.image", "observation.state"})
        self.assertEqual(batch["indices"], [0, 19, 0])
        self.assertEqual(batch["observation"]["observation.image"].shape, (3, 3, 96, 96))
        self.assertEqual(batch["action"].shape, (3, 16, 2))
        self.assertEqual(batch["action_is_pad"].dtype, torch.bool)
        self.assertEqual(batch["action"].dtype, torch.float32)
        self.assertTrue(torch.equal(batch["observation"]["observation.state"][1], torch.tensor([38.0, 39.0])))
        self.assertTrue(torch.equal(batch["action"][0, 0], torch.tensor([100.0, 101.0])))
        self.assertAlmostEqual(float(batch["observation"]["observation.image"][1, 0, 0, 0]), 19 / 255)

    def test_future_chunk_never_crosses_episode(self):
        data = PushTTrainingData(**fixture())
        batch = data.batch([0, 2, 17, 18, 21], device="cpu")
        self.assertEqual(batch["action_is_pad"].sum(dim=1).tolist(), [0, 0, 15, 12, 15])
        self.assertFalse(bool(batch["action_is_pad"][:, 0].any()))
        self.assertTrue(torch.equal(batch["action"][2], torch.tensor([134.0, 135.0]).repeat(16, 1)))
        self.assertTrue(torch.equal(batch["action"][4], torch.tensor([142.0, 143.0]).repeat(16, 1)))
        self.assertNotEqual(batch["action"][2, 15].tolist(), batch["action"][3, 0].tolist())

    def test_split_indices_keep_complete_episodes(self):
        data = PushTTrainingData(**fixture())
        self.assertTrue(np.array_equal(data.train_indices, np.arange(18)))
        self.assertTrue(np.array_equal(data.val_indices, np.arange(18, 22)))
        self.assertFalse(data.train_indices.flags.writeable)
        self.assertFalse(data.frame_indices.flags.writeable)

    def test_reject_bad_or_missing_split_members(self):
        values = fixture()
        for split in ({"train_episodes": [0], "val_episodes": [0], "train_frames": 18, "val_frames": 4},
                      {"train_episodes": [0, 0], "val_episodes": [1], "train_frames": 18, "val_frames": 4},
                      {"train_episodes": [0], "val_episodes": [2], "train_frames": 18, "val_frames": 4},
                      {"train_episodes": [0], "val_episodes": [1], "train_frames": 17, "val_frames": 5}):
            with self.subTest(split=split), self.assertRaises(ValueError):
                split_indices(values["episodes"], split)

    def test_batch_index_contract(self):
        data = PushTTrainingData(**fixture())
        for indices in ([], [-1], [22], [0.0], [True], [[0]], np.array([2**63], dtype=np.uint64)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                data.batch(indices, "cpu")

    def test_episode_and_frame_boundaries_fail_closed(self):
        for episodes, frames in (([0, 1, 0], [0, 0, 1]), ([0, 0], [1, 2]), ([0, 0], [0, 2]), ([0, 1], [0])):
            with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                episode_end_indices(episodes, frames)
        with self.assertRaises(ValueError):
            chunk_indices([0], np.array([0, 2]))

    def test_stats_are_copied_and_never_refit_in_batch(self):
        data = PushTTrainingData(**fixture())
        before = data.stats
        changed = data.stats
        changed["action"]["mean"][0] = 900
        data.batch([18, 19], "cpu")
        self.assertEqual(data.stats, before)
        split = data.split
        split["train_episodes"].append(1)
        self.assertEqual(data.split["train_episodes"], [0])

    def test_binding_stable_and_separate_from_volatile_load_diagnostics(self):
        binding = {"source_sha256": "a" * 64, "image_cache": {"sha256": "b" * 64}}
        first = PushTTrainingData(**fixture(), binding=binding, load_diagnostics={"load_seconds": 2.0})
        second = PushTTrainingData(**fixture(), binding=binding, load_diagnostics={"load_seconds": 5.0})
        self.assertEqual(first.binding, second.binding)
        altered = first.binding
        altered["image_cache"]["sha256"] = "c" * 64
        self.assertEqual(first.binding, binding)
        self.assertNotEqual(first.load_diagnostics, second.load_diagnostics)

    def test_stats_allowlist_and_nonpositive_std(self):
        values = fixture()
        with self.assertRaises(ValueError):
            validate_statistics({**values["stats"], "next.reward": {}})
        for bad in (0.0, float("nan")):
            stats = fixture()["stats"]
            stats["action"]["std"][0] = bad
            with self.assertRaises(ValueError):
                validate_statistics(stats)

    def test_privileged_columns_not_read_or_routed(self):
        forbidden = {"next.reward", "next.done", "next.success", "coverage", "environment_state", "task_index"}
        self.assertFalse(forbidden.intersection(NUMERIC_COLUMNS))
        values = fixture()
        with self.assertRaises(TypeError):
            PushTTrainingData(**values, reward=np.ones(22))
        batch = PushTTrainingData(**values).batch([0], "cpu")
        self.assertFalse(forbidden.intersection(batch["observation"]))

    def test_returned_batch_cannot_modify_cached_arrays(self):
        data = PushTTrainingData(**fixture())
        batch = data.batch([1], "cpu")
        expected = batch["action"].clone()
        batch["action"].zero_()
        batch["observation"]["observation.state"].zero_()
        batch["observation"]["observation.image"].zero_()
        later = data.batch([1], "cpu")
        self.assertTrue(torch.equal(later["action"], expected))
        self.assertEqual(later["observation"]["observation.state"].tolist(), [[2.0, 3.0]])
        self.assertGreater(float(later["observation"]["observation.image"].max()), 0)

    def test_array_shape_dtype_and_native_target_bounds(self):
        for name, replacement in (("images", np.zeros((22, 3, 96, 96), dtype=np.uint8)),
                                  ("images", np.zeros((22, 96, 96, 3), dtype=np.float32)),
                                  ("states", np.zeros((22, 2), dtype=np.float64)),
                                  ("actions", np.full((22, 2), 513.0, dtype=np.float32)),
                                  ("actions", np.full((22, 2), float("nan"), dtype=np.float32))):
            values = fixture()
            values[name] = replacement
            with self.subTest(name=name), self.assertRaises(ValueError):
                PushTTrainingData(**values)

    def test_float32_probe_exactness_and_mismatch_rejection(self):
        images = fixture()["images"]
        probes = [{"index": 0, "image_sha256": float32_image_hash(images[0])},
                  {"index": 21, "image_sha256": float32_image_hash(images[21])}]
        self.assertTrue(all(item["exact_match"] for item in verify_image_probes(images, probes)))
        changed = images.copy()
        changed[21, 0, 0, 0] += 1
        with self.assertRaises(ValueError):
            verify_image_probes(changed, probes)
        with self.assertRaises(ValueError):
            verify_image_probes(images, [probes[0], probes[0]])

    def test_pts_fps_alignment_and_shape_fail_closed(self):
        pixels = np.zeros((96, 96, 3), dtype=np.uint8)
        frame = FakeFrame(1024, Fraction(1, 10240), pixels)
        self.assertIs(check_video_frame(frame, 1, FRAME_COUNT), pixels)
        for bad_frame, index in ((FakeFrame(None, Fraction(1, 10), pixels), 0),
                                 (FakeFrame(2, Fraction(1, 10), pixels), 1),
                                 (FakeFrame(0, Fraction(1, 10), pixels[:, :, :1]), 0),
                                 (FakeFrame(0, Fraction(1, 10), pixels), FRAME_COUNT)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                check_video_frame(bad_frame, index, FRAME_COUNT)

    def test_gate_mismatch_and_no_alternate_decode_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in GATE_PINS:
                (path / name).write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_gate(path)
        with self.assertRaises(ValueError):
            load_training_data(cache_images=False)

    def test_sequential_decoder_one_pass_complete_and_hash_bound(self):
        images = fixture()["images"][:2]
        probes = [{"index": i, "image_sha256": float32_image_hash(image)} for i, image in enumerate(images)]
        container = FakeContainer(images, declared_frames=2)
        fake_av = SimpleNamespace(open=lambda *args, **kwargs: container, __version__="synthetic-test")
        module = "tools.pusht_bc_act_training_data"
        with patch.dict("sys.modules", {"av": fake_av}), patch(f"{module}.FRAME_COUNT", 2), patch(f"{module}._memory_check", return_value={"status": "synthetic"}):
            cache, binding = decode_image_cache(Path("unopened-synthetic.mp4"), probes)
        self.assertTrue(np.array_equal(cache, images))
        self.assertTrue(binding["all_frames_decoded"])
        self.assertEqual(binding["frame_count"], 2)
        self.assertEqual(len(binding["sha256"]), 64)
        self.assertFalse(binding["written_to_disk"])

    def test_sequential_decoder_rejects_wrong_count_and_fps(self):
        images = fixture()["images"][:2]
        module = "tools.pusht_bc_act_training_data"
        for selected, fps, declared in ((images[:1], Fraction(10), 0), (images, Fraction(20), 2), (images, Fraction(10), 99)):
            container = FakeContainer(selected, fps=fps, declared_frames=declared)
            fake_av = SimpleNamespace(open=lambda *args, **kwargs: container, __version__="synthetic-test")
            with self.subTest(count=len(selected), fps=fps, declared=declared), patch.dict("sys.modules", {"av": fake_av}), patch(f"{module}.FRAME_COUNT", 2), patch(f"{module}._memory_check", return_value={}), self.assertRaises(ValueError):
                decode_image_cache(Path("unopened-synthetic.mp4"), [])


if __name__ == "__main__":
    unittest.main()
