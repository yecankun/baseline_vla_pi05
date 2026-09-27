"""Synthetic-array tests only: no source Parquet, video, model or network."""
import copy
import hashlib
import unittest

import numpy as np

from libero_action_study_source_checks import VIEWS, audit_sequences


def fixture(partitions=("train", "validation"), count=9):
    episodes, sequences = [], {}
    for eid, partition in enumerate(partitions, 1):
        metadata = {"episode_index": eid, "length": count}
        for view in VIEWS:
            prefix = f"videos/{view}/"
            metadata.update({prefix + "chunk_index": 0, prefix + "file_index": eid,
                             prefix + "from_timestamp": 0.0, prefix + "to_timestamp": count/10})
        episodes.append({"episode_index": eid, "partition": partition,
                         "metadata": metadata, "source_trajectory_id": f"synthetic/episode{eid}"})
        rng = np.random.default_rng(eid)
        state = (np.round(rng.normal(size=(count, 8)), 2) + eid*10).astype(np.float32)
        action = np.round(rng.normal(size=(count, 7)), 2).astype(np.float32)
        hashes = [hashlib.sha256(f"{eid}/{i}/a".encode()).hexdigest() +
                  hashlib.sha256(f"{eid}/{i}/b".encode()).hexdigest() for i in range(count)]
        sequences[eid] = (state, action, hashes)
    return episodes, sequences


def copy_sequence(sequences, src, dst):
    sequences[dst] = (sequences[src][0].copy(), sequences[src][1].copy(), sequences[src][2].copy())


class SourceSequenceAuditTests(unittest.TestCase):
    def test_distinct_complete_fourteen_all_pairs_and_groups(self):
        episodes, sequences = fixture(("train",)*8 + ("validation",)*4 + ("old_development",)*2)
        report = audit_sequences(episodes, sequences)
        self.assertEqual(report["status"], "passed_declared_duplicate_checks_only")
        self.assertEqual(report["pair_count"], 91)
        self.assertEqual(report["blocking_pair_count"], 0)
        self.assertEqual(report["new_group_count"], 12)
        self.assertEqual(report["new_group_counts_by_partition"], {"train": 8, "validation": 4})
        self.assertEqual(sorted(e for g in report["same_partition_groups"] for e in g["episode_indices"]), list(range(1, 13)))
        self.assertEqual(report["family_provenance"], "unknown")
        self.assertFalse(report["family_independence_verified"])
        self.assertFalse(report["training_ready"])
        self.assertEqual(report["optimizer_steps"], 0)
        pair = report["pairs"][0]["sequence_duplicate_audit"]
        self.assertIn("state_runs", pair)
        self.assertIn("action_runs", pair)
        self.assertIn("two_view_pixel_runs", pair)
        self.assertIn("initial_xyz_distance", pair)
        self.assertIn("median_nearest_xyz_distance_first_to_second", pair)

    def test_same_partition_matches_union_without_rejecting(self):
        episodes, sequences = fixture(("train", "train", "train", "validation"))
        copy_sequence(sequences, 1, 2)
        copy_sequence(sequences, 2, 3)
        report = audit_sequences(episodes, sequences)
        self.assertEqual(report["blocking_pairs"], [])
        self.assertEqual(report["new_group_count"], 2)
        self.assertEqual(report["same_partition_groups"][0]["episode_indices"], [1, 2, 3])
        self.assertEqual(len(report["same_partition_grouping_pairs"]), 3)

    def test_cross_split_duplicate_rejected(self):
        episodes, sequences = fixture()
        copy_sequence(sequences, 1, 2)
        report = audit_sequences(episodes, sequences)
        self.assertEqual(report["status"], "rejected_source_isolation")
        self.assertEqual(report["blocking_pair_count"], 1)
        self.assertIn("exact_transition_windows", report["blocking_pairs"][0]["reasons"])
        self.assertEqual(report["new_group_count"], 2)

    def test_new_old_overlap_rejected_for_both_new_partitions(self):
        for partition in ("train", "validation"):
            with self.subTest(partition=partition):
                episodes, sequences = fixture((partition, "old_development"))
                copy_sequence(sequences, 1, 2)
                report = audit_sequences(episodes, sequences)
                self.assertEqual(report["blocking_pair_count"], 1)
                self.assertEqual(report["pairs"][0]["relationship"], "new_vs_old_development")

    def test_old_old_overlap_is_descriptive_only(self):
        episodes, sequences = fixture(("old_development", "old_development"))
        copy_sequence(sequences, 1, 2)
        report = audit_sequences(episodes, sequences)
        self.assertEqual(report["blocking_pair_count"], 0)
        self.assertTrue(report["pairs"][0]["match_reasons"])
        self.assertEqual(report["pairs"][0]["decision"], "old_development_descriptive_only")

    def test_terminal_action_does_not_hide_duplicate_window(self):
        episodes, sequences = fixture(count=7)
        state, action, hashes = sequences[1]
        action2 = action.copy()
        action2[-1] += 15
        sequences[2] = (state.copy(), action2, sequences[2][2])
        report = audit_sequences(episodes, sequences)
        matches = report["pairs"][0]["sequence_duplicate_audit"]["cross_split_matching_windows"]
        self.assertEqual(matches["exact_transition_windows"], 1)
        self.assertEqual(matches["quantized_transition_windows"], 1)
        self.assertEqual(matches["two_view_pixel_windows"], 0)
        self.assertEqual(report["blocking_pair_count"], 1)

    def test_quantized_only_duplicate_rejected(self):
        episodes, sequences = fixture(count=7)
        state = np.arange(56, dtype=np.float32).reshape(7, 8)*np.float32(.01)
        action = np.arange(49, dtype=np.float32).reshape(7, 7)*np.float32(.01)
        sequences[1] = (state, action, sequences[1][2])
        sequences[2] = (state + np.float32(1e-6), action + np.float32(1e-6), sequences[2][2])
        report = audit_sequences(episodes, sequences)
        matches = report["pairs"][0]["sequence_duplicate_audit"]["cross_split_matching_windows"]
        self.assertEqual(matches["exact_transition_windows"], 0)
        self.assertEqual(matches["quantized_transition_windows"], 1)
        self.assertEqual(report["blocking_pair_count"], 1)

    def test_shifted_seven_row_suffix_match_rejected(self):
        episodes, sequences = fixture(count=11)
        state, action, hashes = sequences[2]
        state[4:] = sequences[1][0][2:9]
        action[4:10] = sequences[1][1][2:8]
        report = audit_sequences(episodes, sequences)
        audit = report["pairs"][0]["sequence_duplicate_audit"]
        self.assertEqual(audit["cross_split_matching_windows"]["exact_transition_windows"], 1)
        self.assertEqual(audit["state_runs"]["longest_exact_contiguous_match"], 7)
        self.assertEqual(audit["action_runs"]["longest_exact_contiguous_match"], 6)
        self.assertEqual(report["blocking_pair_count"], 1)

    def test_pixel_only_duplicate_rejected(self):
        episodes, sequences = fixture(count=7)
        sequences[2] = (sequences[2][0], sequences[2][1], sequences[1][2].copy())
        report = audit_sequences(episodes, sequences)
        self.assertEqual(report["blocking_pairs"][0]["reasons"], ["two_view_pixel_windows"])

    def test_same_video_overlapping_interval_rejected(self):
        episodes, sequences = fixture()
        key = f"videos/{VIEWS[0]}/file_index"
        episodes[1]["metadata"][key] = episodes[0]["metadata"][key]
        report = audit_sequences(episodes, sequences)
        self.assertEqual(report["blocking_pairs"][0]["reasons"], ["overlapping_source_video_intervals"])
        self.assertEqual(len(report["pairs"][0]["overlapping_video_intervals"]), 1)

    def test_adjacent_same_video_intervals_are_allowed(self):
        episodes, sequences = fixture()
        prefix = f"videos/{VIEWS[0]}/"
        episodes[1]["metadata"].update({prefix + "file_index": 1, prefix + "from_timestamp": .9,
                                        prefix + "to_timestamp": 1.8})
        self.assertEqual(audit_sequences(episodes, sequences)["blocking_pair_count"], 0)

    def test_known_family_links_transitive_and_cross_split_rejected(self):
        episodes, sequences = fixture(("train", "train", "validation"))
        report = audit_sequences(episodes, sequences, [(1, 2), (2, 3)])
        self.assertEqual(report["blocking_pair_count"], 2)
        self.assertEqual(report["known_family_link_components"], [[1, 2, 3]])
        self.assertEqual(report["same_partition_groups"][0]["episode_indices"], [1, 2])
        self.assertEqual(report["family_provenance"], "externally_supplied_links_not_exhaustive")
        self.assertFalse(report["family_independence_verified"])

    def test_known_family_new_old_rejected(self):
        episodes, sequences = fixture(("train", "old_development"))
        report = audit_sequences(episodes, sequences, [(1, 2)])
        self.assertEqual(report["blocking_pairs"][0]["reasons"], ["known_family_or_derivative_link"])

    def test_same_source_identity_is_not_independence(self):
        episodes, sequences = fixture()
        episodes[1]["source_trajectory_id"] = episodes[0]["source_trajectory_id"]
        report = audit_sequences(episodes, sequences)
        self.assertEqual(report["blocking_pairs"][0]["reasons"], ["same_source_trajectory_id"])

    def test_array_dtype_shape_and_nonfinite_fail_closed(self):
        for field, mutation in ((0, lambda a: a.astype(np.float64)), (1, lambda a: a[:, :6]),
                                (0, lambda a: np.full_like(a, np.nan)),
                                (1, lambda a: np.full_like(a, np.inf))):
            with self.subTest(field=field, mutation=mutation):
                episodes, sequences = fixture()
                triple = list(sequences[1])
                triple[field] = mutation(triple[field])
                sequences[1] = tuple(triple)
                with self.assertRaises(ValueError):
                    audit_sequences(episodes, sequences)

    def test_bad_frame_hashes_fail_closed(self):
        for bad in ("a"*64, "g"*128, "A"*128, 123, None):
            with self.subTest(bad=bad):
                episodes, sequences = fixture()
                sequences[1][2][0] = bad
                with self.assertRaises(ValueError):
                    audit_sequences(episodes, sequences)

    def test_metadata_invalid_timestamps_fail_closed(self):
        for bad in (float("nan"), float("inf"), "0", True, -1, 2):
            with self.subTest(bad=bad):
                episodes, sequences = fixture()
                episodes[0]["metadata"][f"videos/{VIEWS[0]}/from_timestamp"] = bad
                with self.assertRaises(ValueError):
                    audit_sequences(episodes, sequences)

    def test_unknown_or_duplicate_episode_and_sequence_keys_fail_closed(self):
        for mutation in (lambda e, s: e.append(copy.deepcopy(e[0])),
                         lambda e, s: e[0].update(partition="test"),
                         lambda e, s: s.update({99: s[1]}),
                         lambda e, s: s.pop(1),
                         lambda e, s: s.update({"1": s.pop(1)}),
                         lambda e, s: e[0]["metadata"].update(episode_index=99),
                         lambda e, s: e[0]["metadata"].update(length=8)):
            episodes, sequences = fixture()
            mutation(episodes, sequences)
            with self.assertRaises(ValueError):
                audit_sequences(episodes, sequences)

    def test_invalid_family_links_fail_closed(self):
        for links in (None, "12", [(1, 9)], [(1, 1)], [(1, 2), (2, 1)], [(1,)], [(True, 2)]):
            with self.subTest(links=links):
                episodes, sequences = fixture()
                with self.assertRaises(ValueError):
                    audit_sequences(episodes, sequences, links)

    def test_inputs_unchanged(self):
        episodes, sequences = fixture()
        old_episodes, old_sequences = copy.deepcopy(episodes), copy.deepcopy(sequences)
        audit_sequences(episodes, sequences)
        self.assertEqual(episodes, old_episodes)
        for eid, (state, action, hashes) in sequences.items():
            np.testing.assert_array_equal(state, old_sequences[eid][0])
            np.testing.assert_array_equal(action, old_sequences[eid][1])
            self.assertEqual(hashes, old_sequences[eid][2])


if __name__ == "__main__":
    unittest.main()
