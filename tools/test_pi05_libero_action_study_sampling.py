"""Synthetic metadata-only sampling tests; no features, model or optimizer."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import pi05_libero_action_study_sampling as sampling


def windows(spec=((1312, 1), (1419, 100), (1518, 200))):
    result = []
    for episode_position, (eid, count) in enumerate(spec):
        base = episode_position * 10000
        for i in range(count):
            result.append({"episode_index": eid, "history": list(range(base+i, base+i+4)),
                           "actions": list(range(base+i+3, base+i+6)),
                           "targets": list(range(base+i+4, base+i+7))})
    return result


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.windows = windows()
        self.groups = {"source_group_1312": [1312], "source_group_1419": [1419, 1518]}

    def build(self, **kwargs):
        return sampling.build_draw_manifest(self.windows, self.groups, seed=20260912, **kwargs)

    def test_default_budget_schema_counts_and_train_only(self):
        manifest = self.build()
        self.assertEqual(manifest["draw_count"], 3200)
        self.assertEqual(manifest["batches"], 200)
        self.assertEqual(manifest["batch_size"], 16)
        self.assertEqual(sum(manifest["counts"]["by_group"].values()), 3200)
        self.assertEqual(sum(manifest["counts"]["by_episode"].values()), 3200)
        self.assertEqual(sum(manifest["counts"]["by_window_index"]), 3200)
        self.assertEqual(len(manifest["counts"]["by_window_index"]), len(self.windows))
        self.assertEqual(set(d["episode_index"] for d in manifest["draws"]), {1312, 1419, 1518})
        self.assertEqual(manifest["optimizer_steps"], 0)
        self.assertFalse(manifest["training_started"])
        self.assertFalse(manifest["group_metadata_policy_input"])
        self.assertFalse(manifest["normalization_fitted"])
        self.assertEqual(json.loads(json.dumps(manifest, allow_nan=False)), manifest)

    def test_repeat_and_seed_specific_replay(self):
        by_seed = [sampling.build_draw_manifest(self.windows, self.groups, s, batches=8) for s in sampling.SEEDS]
        for manifest in by_seed:
            self.assertTrue(sampling.verify_draw_manifest(manifest, self.windows, self.groups))
            self.assertEqual(manifest, sampling.build_draw_manifest(self.windows, self.groups, manifest["seed"], batches=8))
        self.assertEqual(len({m["draws_sha256"] for m in by_seed}), 3)
        self.assertEqual(len({m["rng"]["initial_state_sha256"] for m in by_seed}), 3)

    def test_exact_three_level_cpu_rng_replay_includes_singleton_calls(self):
        manifest = self.build(batches=4)
        generator = torch.Generator(device="cpu").manual_seed(20260912)
        groups = {gid: sorted(ids) for gid, ids in sorted(self.groups.items())}
        keys = list(groups)
        for expected in manifest["draws"]:
            group = keys[int(torch.randint(len(keys), (1,), generator=generator, device="cpu", dtype=torch.int64))]
            episode = groups[group][int(torch.randint(len(groups[group]), (1,), generator=generator, device="cpu", dtype=torch.int64))]
            support = [i for i, w in enumerate(self.windows) if w["episode_index"] == episode]
            index = support[int(torch.randint(len(support), (1,), generator=generator, device="cpu", dtype=torch.int64))]
            self.assertEqual((group, episode, index), (expected["group_id"], expected["episode_index"], expected["window_index"]))
        self.assertEqual(manifest["rng"]["calls_per_draw"], 3)

    def test_shared_arms_identical_one_schedule_not_arm_conditioned(self):
        manifest = self.build(batches=2)
        self.assertEqual(manifest["shared_arms"], ["observed_action", "normalized_zero_action"])
        same_by_arm = {arm: manifest["draws_sha256"] for arm in manifest["shared_arms"]}
        self.assertEqual(len(set(same_by_arm.values())), 1)
        self.assertTrue(all(set(draw) == {"window_index", "episode_index", "group_id", "batch_index", "draw_in_batch"}
                            for draw in manifest["draws"]))
        with self.assertRaises(TypeError):
            sampling.build_draw_manifest(self.windows, self.groups, 20260912, arm="observed_action")

    def test_group_then_episode_balance_not_uniform_windows_or_episodes(self):
        manifest = self.build()
        counts = manifest["counts"]
        # One group contains 1 window; the other has 300. Window-uniform would
        # almost never pick1312. Episode-uniform would pick it only about1/3.
        self.assertTrue(1400 <= counts["by_episode"]["1312"] <= 1800)
        self.assertTrue(650 <= counts["by_episode"]["1419"] <= 950)
        self.assertTrue(650 <= counts["by_episode"]["1518"] <= 950)
        probabilities = {p["episode_index"]: p for p in manifest["sampling_probabilities"]}
        self.assertEqual(probabilities[1312]["episode_probability"], .5)
        self.assertEqual(probabilities[1419]["episode_probability"], .25)
        self.assertEqual(probabilities[1518]["episode_probability"], .25)
        self.assertEqual(probabilities[1312]["per_window_probability"], .5)
        self.assertEqual(probabilities[1419]["per_window_probability"], .0025)
        self.assertEqual(probabilities[1518]["per_window_probability"], .00125)

    def test_all_eight_singleton_groups_supported(self):
        inputs = windows([(eid, 7) for eid in sorted(sampling.TRAIN_EPISODE_IDS)])
        groups = {f"source_group_{eid}": [eid] for eid in sorted(sampling.TRAIN_EPISODE_IDS)}
        manifest = sampling.build_draw_manifest(inputs, groups, 20260913)
        self.assertEqual(len(manifest["groups"]), 8)
        self.assertEqual(manifest["train_episode_indices"], sorted(sampling.TRAIN_EPISODE_IDS))
        self.assertTrue(all(290 < count < 520 for count in manifest["counts"]["by_group"].values()))

    def test_no_global_python_numpy_torch_rng_or_mode_changes(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        grad_enabled = torch.is_grad_enabled()
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.build(batches=3)
        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        self.assertTrue(np.array_equal(current_numpy_state[1], numpy_state[1]))
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(torch.is_grad_enabled(), grad_enabled)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)

    def test_input_order_is_preserved_but_group_and_member_order_canonicalized(self):
        original_windows, original_groups = deepcopy(self.windows), deepcopy(self.groups)
        first = self.build(batches=3)
        reverse_groups = {gid: list(reversed(members)) for gid, members in reversed(list(self.groups.items()))}
        second = sampling.build_draw_manifest(self.windows, reverse_groups, 20260912, batches=3)
        self.assertEqual(first, second)
        self.assertEqual(self.windows, original_windows)
        self.assertEqual(self.groups, original_groups)
        reversed_windows = list(reversed(self.windows))
        manifest = sampling.build_draw_manifest(reversed_windows, self.groups, 20260912, batches=3)
        self.assertNotEqual(first["train_windows_sha256"], manifest["train_windows_sha256"])
        for position, draw in enumerate(manifest["draws"]):
            self.assertEqual(reversed_windows[draw["window_index"]]["episode_index"], draw["episode_index"])
            self.assertEqual((draw["batch_index"], draw["draw_in_batch"]), divmod(position, 16))

    def test_validation_old_or_unknown_ids_rejected_even_if_groups_agree(self):
        for eid in (1530, 1476, 1458, 1566, 1400, 1401, 1402, 9999):
            with self.subTest(eid=eid), self.assertRaisesRegex(ValueError, "not a frozen training episode"):
                sampling.build_draw_manifest(windows([(eid, 1)]), {"group": [eid]}, 20260912)

    def test_group_coverage_disjointness_and_shape_fail_closed(self):
        bad_groups = ({}, [], {"a": []}, {"a": [1312]}, {"a": [1312, 1419, 1518, 1633]},
                      {"a": [1312, 1419, 1518, 1530]}, {"a": [1312, 1312, 1419, 1518]},
                      {"a": [1312, 1419], "b": [1419, 1518]},
                      {"": [1312, 1419, 1518]}, {"bad group": [1312, 1419, 1518]},
                      {True: [1312, 1419, 1518]}, {"x" * 129: [1312, 1419, 1518]},
                      {"a": (1312, 1419, 1518)}, {"a": [True, 1419, 1518]},
                      {"a": [1312., 1419, 1518]}, {"a": {"episodes": [1312, 1419, 1518]}})
        for groups in bad_groups:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                sampling.build_draw_manifest(self.windows, groups, 20260912)

    def test_window_unknown_fields_empty_duplicate_and_harmful_types_rejected(self):
        for inputs in ([], {}, tuple(self.windows), [self.windows[0], self.windows[0]]):
            with self.subTest(inputs=type(inputs)), self.assertRaises(ValueError):
                sampling.build_draw_manifest(inputs, self.groups, 20260912)
        for key in ("partition", "group_id", "sample_role", "diagnostic_targets", "reward", "exact_tip"):
            inputs = deepcopy(self.windows)
            inputs[0][key] = "not a model input"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exactly"):
                sampling.build_draw_manifest(inputs, self.groups, 20260912)
        for key, value in (("episode_index", True), ("episode_index", 1312.),
                           ("history", [False, 1, 2, 3]), ("history", [0., 1, 2, 3]),
                           ("history", [-1, 0, 1, 2]), ("history", ["0", 1, 2, 3]),
                           ("history", [0, 1, 2]), ("history", (0, 1, 2, 3)),
                           ("actions", [4, 5, 6]), ("targets", [5, 6, 7]),
                           ("targets", [4, 6, 7])):
            inputs = deepcopy(self.windows)
            inputs[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                sampling.build_draw_manifest(inputs, self.groups, 20260912)

    def test_global_rows_cannot_be_shared_across_episode_ids(self):
        inputs = windows([(1312, 1), (1419, 1)])
        for key in ("history", "actions", "targets"):
            inputs[1][key] = inputs[0][key].copy()
        with self.assertRaisesRegex(ValueError, "shared across episode"):
            sampling.build_draw_manifest(inputs, {"a": [1312, 1419]}, 20260912)

    def test_seed_and_prospective_budget_are_bounded(self):
        for seed in (True, 20260912., "20260912", -1, 0, 20260915):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                sampling.build_draw_manifest(self.windows, self.groups, seed)
        for kwargs in ({"batches": 0}, {"batches": 201}, {"batches": True}, {"batches": 200.},
                       {"batch_size": 0}, {"batch_size": 17}, {"batch_size": True}, {"batch_size": "16"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_replay_rejects_draw_summary_runtime_and_bool_int_mutations(self):
        original = self.build(batches=2)
        changes = (lambda m: m["draws"][0].update(window_index=999),
                   lambda m: m["draws"][0].update(batch_index=False),
                   lambda m: m["draws"][0].update(group_id="wrong"),
                   lambda m: m["draws"].reverse(),
                   lambda m: m["counts"]["by_window_index"].__setitem__(0, 999),
                   lambda m: m["rng"].update(torch_version="different"),
                   lambda m: m.update(optimizer_steps=False),
                   lambda m: m.update(extra="unknown"),
                   lambda m: m.update(draws_sha256="a" * 64))
        for change in changes:
            manifest = deepcopy(original)
            change(manifest)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "replay differs"):
                sampling.verify_draw_manifest(manifest, self.windows, self.groups)

    def test_replay_rejects_changed_window_support(self):
        manifest = self.build(batches=2)
        changed = list(reversed(self.windows))
        with self.assertRaisesRegex(ValueError, "replay differs"):
            sampling.verify_draw_manifest(manifest, changed, self.groups)

    def test_three_explicit_cpu_randint_calls_per_draw(self):
        original = torch.randint
        calls = []
        def counted(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)
        with patch.object(sampling.torch, "randint", side_effect=counted):
            manifest = self.build(batches=2, batch_size=3)
        self.assertEqual(len(calls), manifest["draw_count"] * 3)
        self.assertTrue(all(k["device"] == "cpu" and k["dtype"] == torch.int64
                            and isinstance(k["generator"], torch.Generator) for _, k in calls))


if __name__ == "__main__":
    unittest.main()
