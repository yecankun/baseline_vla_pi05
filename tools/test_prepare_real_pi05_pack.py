from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from prepare_real_pi05_pack import EPISODES, encode_real_state, prepare_pack


def write_fixture(root):
    for task in ("left", "right"):
        episode = root / task
        episode.mkdir()
        manifest = {"task": task, "zero_orientation_delta": True,
                    "camera_setup": {"top_duplicated_from_side": False}}
        (episode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows = []
        for step in range(3):
            image = np.full((36, 64, 3), 30 + step * 40, dtype=np.uint8)
            cv2.imwrite(str(episode / f"{step}.png"), image)
            rows.append({"task": task, "step": step, "timestamp": step * .23,
                         "timestamps": {"side_image": step * .23, "top_image": step * .23 + .01,
                                        "elite_pose": step * .23 + .03, "piper_action": step * .23 + .05},
                         "side_image": f"{step}.png", "top_image": f"{step}.png",
                         "state": {"elite_tcp_pose_6d": [step * 2, 0, 0, 3, 0, 0],
                                   "controller_state": {"piper_step_after_command": 1,
                                                        "piper_busy": step == 0,
                                                        "piper_intent": "feed" if step == 0 else "hold"}},
                         "reference_action": {"elite_tcp_delta_6d": [2 if step < 2 else 0, 0, 0, 0, 0, 0],
                                              "piper_step_command": int(step == 0), "piper_burst_count": 1}})
        (episode / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RealAdapterTest(unittest.TestCase):
    def test_allowlist_and_observable_history(self):
        self.assertEqual(len(EPISODES), 10)
        self.assertTrue(EPISODES[0].endswith("left_s_bend_replacement_tip_fixedudp_002"))
        pose = [1, 2, 3, 3.06, 0, 0]
        empty, _ = encode_real_state(pose, "left")
        self.assertEqual(empty[27], 0)
        np.testing.assert_array_equal(empty[16:], np.zeros(16))
        history = {"piper_busy": True, "piper_step_after_command": 2}
        state, _ = encode_real_state(pose, "right", history)
        np.testing.assert_array_equal(state[[6, 16, 27]], [2, 1, 1])
        # Even forbidden fields in the snapshot cannot enter the allowlisted encoder.
        noisy = dict(history, piper_intent="feed", elite_path_index=999,
                     elite_requested_tcp_pose_6d=[999] * 6, exact_contact=True, event_id=9)
        np.testing.assert_array_equal(encode_real_state(pose, "right", noisy)[0], state)
        np.testing.assert_array_equal(state[17:27], np.zeros(10))

    def test_pack_and_current_label_does_not_enter_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)
            report = prepare_pack(root, root / "pack", episodes=("left", "right"))
            self.assertEqual(report["training_transitions"], 4)
            self.assertEqual(report["piper_intent_id"], {"1": 2, "2": 2})
            self.assertEqual(report["images_decoded_and_resized"], 12)
            with np.load(root / "pack/openpi_arrays.npz") as arrays:
                states = arrays["state_32"].copy()
                np.testing.assert_array_equal(arrays["piper_intent_id"], [2, 1, 2, 1])
                np.testing.assert_array_equal(arrays["action_32"][:, 6:9].argmax(axis=1), [2, 1, 2, 1])
                np.testing.assert_array_equal(states[:, 16], [0, 1, 0, 1])
                np.testing.assert_array_equal(states[:, 27], [0, 1, 0, 1])
            records_path = root / "left/records.jsonl"
            rows = [json.loads(line) for line in records_path.read_text().splitlines()]
            rows[0]["reference_action"]["piper_step_command"] = 0
            rows[0]["state"]["controller_state"]["piper_intent"] = "hold"
            records_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            prepare_pack(root, root / "changed", episodes=("left", "right"))
            with np.load(root / "changed/openpi_arrays.npz") as arrays:
                np.testing.assert_array_equal(arrays["state_32"], states)

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "LeRobot is installed on project4090 only")
    def test_remote_lerobot_conversion_and_readback(self):
        from export_openpi_compat_to_lerobot import import_lerobot_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)
            prepare_pack(root, root / "pack", episodes=("left", "right"))
            subprocess.run([sys.executable, str(Path(__file__).with_name("export_openpi_compat_to_lerobot.py")),
                            str(root / "pack"), "--out", str(root / "lerobot"), "--repo-id", "project2026/real-adapter-check",
                            "--fps", "5", "--no-videos"], check=True)
            dataset = import_lerobot_dataset()(repo_id="project2026/real-adapter-check", root=root / "lerobot")
            self.assertEqual(len(dataset), 4)
            np.testing.assert_array_equal(dataset[1]["observation.state"].numpy()[[6, 16, 27]], [1, 1, 1])
            self.assertEqual(tuple(dataset[0]["observation.images.side"].shape), (3, 224, 224))
            self.assertEqual(int(dataset[0]["piper_intent_id"].item()), 2)
            meta = json.loads((root / "lerobot/project2026_lerobot_export_manifest.json").read_text())
            self.assertEqual(meta["val_episode_indices"], [])
            self.assertFalse(meta["timing"]["lerobot_fps_is_physical"])
            self.assertTrue((root / "lerobot/project2026_source_index.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
