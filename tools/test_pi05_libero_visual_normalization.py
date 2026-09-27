"""Synthetic train-only visual normalization contracts; no public extraction/training."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import pi05_libero_visual_normalization as normalization
from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
from pi05_libero_world_model_adapter import ARRAY_NAMES, SPLIT_SCHEMA
from test_pi05_libero_action_study_step0 import REGISTRY, TinyDataset


class TrainOnlyArray:
    """Readable training rows with a sentinel that forbids validation access."""
    def __init__(self, value, allowed):
        self.value = value
        self.allowed = set(allowed)
        self.reads = []
        self.shape, self.dtype, self.ndim = value.shape, value.dtype, value.ndim

    def __len__(self):
        return len(self.value)

    def __array__(self, *args, **kwargs):
        raise AssertionError("bulk conversion would inspect held-out rows")

    def __getitem__(self, key):
        rows = np.arange(len(self.value))[key]
        flat = np.asarray(rows).reshape(-1)
        if not set(map(int, flat)) <= self.allowed:
            raise AssertionError("held-out visual values are inaccessible")
        self.reads.extend(map(int, flat))
        return self.value[key]


def synthetic_pack():
    """Complete unequal-length train episodes (2/3 rows) and one held-out pair."""
    episodes, indices = {}, {}
    offset = 0
    for episode, count in ((11, 2), (22, 3), (33, 2)):
        episodes[episode] = dict(episode_index=episode, task_id=9,
            source_trajectory_id=f"synthetic-source-{episode}", leakage_group_id=f"synthetic-group-{episode}",
            record_count=count, complete_episode=True)
        indices[episode] = np.arange(offset, offset + count, dtype=np.int64)
        offset += count
    visual = np.empty((7, 2, 2048), np.float32)
    coordinate = np.arange(2048, dtype=np.float32) / 2048
    for row in range(7):
        visual[row, 0] = 3 + coordinate + row
        visual[row, 1] = -7 + coordinate + row * 2
    visual[:5, :, 0] = 9.0  # A genuinely constant coordinate exercises the floor.
    arrays = dict(visual_latent=visual, visual_valid=np.ones((7, 2), bool),
        episode_index=np.asarray([11, 11, 22, 22, 22, 33, 33], np.int64),
        frame_index=np.asarray([0, 1, 0, 1, 2, 0, 1], np.int64), task_id=np.full(7, 9, np.int64),
        state=np.zeros((7, 8), np.float32), state_valid=np.ones((7, 8), bool), action=np.zeros((7, 7), np.float32))
    manifest = dict(episodes=list(episodes.values()), task_registry=deepcopy(REGISTRY),
        source={"metadata_sha256": "b" * 64},
        arrays={name: {"sha256": format(i + 1, "064x")} for i, name in enumerate(sorted(ARRAY_NAMES))})
    pack = SimpleNamespace(arrays=arrays, indices=indices, episodes=episodes, manifest=manifest,
        manifest_sha256="a" * 64, visual_dim=2048, tasks={9: REGISTRY[0]}, _validated_splits={})
    split = dict(schema=SPLIT_SCHEMA, feature_pack_manifest_sha256=pack.manifest_sha256,
        train_episode_indices=[22, 11], validation_episode_indices=[33], split_sha256="c" * 64)
    register_split(pack, split)
    return pack, split


def register_split(pack, split):
    pack._validated_splits[split["split_sha256"]] = json.dumps(split, sort_keys=True)


class VisualNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.pack, self.split = synthetic_pack()
        self.config = LiberoWorldModelConfig(hidden_dim=4)

    def statistics(self, *, identity=False):
        value = normalization.fit_visual_statistics(self.pack, self.split, scale_floor=.1)
        if identity:
            value["mean"] = np.zeros((2, 2048)).tolist()
            value["std"] = np.ones((2, 2048)).tolist()
            value["scale"] = np.ones((2, 2048)).tolist()
        return value

    def pair(self, statistics):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(72)
            native = LiberoWorldModel(self.config, REGISTRY).eval()
            torch.manual_seed(72)
            normalized = normalization.NormalizedLiberoWorldModel(self.config, REGISTRY, statistics).eval()
        return native, normalized

    def inputs(self):
        dataset = TinyDataset()
        rng = np.random.default_rng(29)
        for item in dataset.items[:2]:
            item["inputs"]["history_visual_latent"][:] = rng.normal(2, .5, (4, 2, 2048)).astype(np.float32)
            item["inputs"]["history_state"][:] = rng.normal(0, .2, (4, 8)).astype(np.float32)
        return collate_window_inputs([item["inputs"] for item in dataset.items[:2]])

    def assert_tensor_bytes_equal(self, left, right):
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(torch.equal(left.detach().contiguous().view(torch.uint8), right.detach().contiguous().view(torch.uint8)))

    def test_fit_never_accesses_validation_values_or_bulk_arrays(self):
        for name in ("visual_latent", "visual_valid"):
            self.pack.arrays[name] = TrainOnlyArray(self.pack.arrays[name], range(5))
        result = self.statistics()
        self.assertEqual(result["train_record_count"], 5)
        self.assertEqual(result["train_row_indices"], list(range(5)))
        for name in ("visual_latent", "visual_valid"):
            self.assertEqual(set(self.pack.arrays[name].reads), set(range(5)))

    def test_float64_population_statistics_include_both_final_observation_rows(self):
        raw = self.pack.arrays["visual_latent"][:5].astype(np.float64)
        result = self.statistics()
        np.testing.assert_array_equal(result["mean"], raw.mean(0))
        np.testing.assert_allclose(result["std"], raw.std(0, ddof=0), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result["scale"], np.maximum(raw.std(0), .1))
        self.assertEqual(np.asarray(result["std"])[:, 0].tolist(), [0., 0.])
        self.assertEqual(np.asarray(result["scale"])[:, 0].tolist(), [.1, .1])
        self.assertEqual(result["train_episode_indices"], [11, 22])
        self.assertEqual([row["record_count"] for row in result["train_episodes"]], [2, 3])
        self.assertEqual(result["variance"], "population_ddof0_float64")

    def test_changed_held_out_values_and_masks_cannot_change_statistics(self):
        first = self.statistics()
        self.pack.arrays["visual_latent"][5:] = np.nan
        self.pack.arrays["visual_valid"][5:] = False
        second = self.statistics()
        self.assertEqual(first, second)
        self.assertEqual(normalization.statistics_sha256(first), normalization.statistics_sha256(second))

    def test_fitting_does_not_mutate_any_source_array_or_split(self):
        arrays = {name: value.copy() for name, value in self.pack.arrays.items()}
        split = deepcopy(self.split)
        result = self.statistics()
        self.assertEqual(self.split, split)
        for name, value in arrays.items():
            np.testing.assert_array_equal(self.pack.arrays[name], value)
        result["mean"][0][0] = -99
        np.testing.assert_array_equal(self.pack.arrays["visual_latent"], arrays["visual_latent"])

    def test_statistics_provenance_hashes_exact_selected_raw_training_values(self):
        result = self.statistics()
        self.assertEqual(result["feature_pack_manifest_sha256"], self.pack.manifest_sha256)
        self.assertEqual(result["split_sha256"], self.split["split_sha256"])
        self.assertEqual(result["source_metadata_sha256"], self.pack.manifest["source"]["metadata_sha256"])
        self.assertEqual(result["source_array_sha256"],
                         {name: self.pack.manifest["arrays"][name]["sha256"] for name in ARRAY_NAMES})
        self.assertEqual(result["train_visual_sha256"], normalization._array_sha(self.pack.arrays["visual_latent"][:5]))
        self.assertEqual(result["train_visual_valid_sha256"], normalization._array_sha(self.pack.arrays["visual_valid"][:5]))
        # A changed training value affects the actual fitted values and their digest;
        # unchanged manifest digests here are synthetic fixtures, not file validation.
        self.pack.arrays["visual_latent"][4, 0, 1] += 2
        changed = self.statistics()
        self.assertNotEqual(result["mean"], changed["mean"])
        self.assertNotEqual(result["train_visual_sha256"], changed["train_visual_sha256"])
        self.assertNotEqual(normalization.statistics_sha256(result), normalization.statistics_sha256(changed))

    def test_duplicate_overlap_empty_unknown_and_unvalidated_split_are_rejected(self):
        for mode in ("duplicate", "overlap", "empty", "unknown", "unvalidated"):
            pack, split = synthetic_pack()
            if mode == "duplicate":
                split["train_episode_indices"] = [11, 11, 22]
            elif mode == "overlap":
                split["validation_episode_indices"] = [22, 33]
            elif mode == "empty":
                split["train_episode_indices"] = []
            elif mode == "unknown":
                split["train_episode_indices"] = [11, 44]
            else:
                pack._validated_splits.clear()
            if mode != "unvalidated":
                register_split(pack, split)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                normalization.fit_visual_statistics(pack, split)

    def test_incomplete_or_duplicate_training_rows_are_rejected(self):
        for mode in ("duplicate", "missing", "frame_gap", "incomplete"):
            pack, split = synthetic_pack()
            if mode == "duplicate":
                pack.indices[22] = np.asarray([2, 2, 4])
            elif mode == "missing":
                pack.indices[22] = np.asarray([2, 3])
            elif mode == "frame_gap":
                pack.arrays["frame_index"][3] = 9
            else:
                pack.episodes[22]["complete_episode"] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                normalization.fit_visual_statistics(pack, split)

    def test_invalid_training_features_masks_dtype_and_floor_are_rejected(self):
        for mode in ("nan", "mask", "dtype", "shape"):
            pack, split = synthetic_pack()
            if mode == "nan":
                pack.arrays["visual_latent"][0, 0, 1] = np.nan
            elif mode == "mask":
                pack.arrays["visual_valid"][0, 0] = False
            elif mode == "dtype":
                pack.arrays["visual_latent"] = pack.arrays["visual_latent"].astype(np.float64)
            else:
                pack.arrays["visual_latent"] = pack.arrays["visual_latent"][:, :1]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                normalization.fit_visual_statistics(pack, split)
        for floor in (0., -.1, float("nan"), float("inf"), True):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                normalization.fit_visual_statistics(self.pack, self.split, scale_floor=floor)

    def test_validate_returns_owned_payload_and_rejects_artifact_drift(self):
        original = self.statistics()
        validated = normalization.validate_statistics(original, 2, 2048)
        self.assertEqual(validated, original)
        validated["mean"][0][0] += 1
        self.assertNotEqual(validated["mean"], original["mean"])
        for mode in ("extra", "missing", "schema", "mean_shape", "nonfinite", "scale", "floor", "scope", "rows", "hash"):
            bad = deepcopy(original)
            if mode == "extra": bad["unapproved"] = 1
            elif mode == "missing": del bad["std"]
            elif mode == "schema": bad["schema"] = "wrong"
            elif mode == "mean_shape": bad["mean"][0].pop()
            elif mode == "nonfinite": bad["mean"][0][0] = float("inf")
            elif mode == "scale": bad["scale"][0][1] *= 2
            elif mode == "floor": bad["scale_floor"] = 0.
            elif mode == "scope": bad["fitting_scope"] = "all_rows"
            elif mode == "rows": bad["train_record_count"] -= 1
            else: bad["feature_pack_manifest_sha256"] = "not-a-hash"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                normalization.validate_statistics(bad, 2, 2048)

    def test_identity_normalization_is_bitexact_native_forward_and_parameter_initialization(self):
        native, normalized = self.pair(self.statistics(identity=True))
        native_params, normalized_params = dict(native.named_parameters()), dict(normalized.named_parameters())
        self.assertEqual(set(native_params), set(normalized_params))
        for name in native_params:
            self.assert_tensor_bytes_equal(native_params[name], normalized_params[name])
        inputs = self.inputs()
        with torch.no_grad():
            baseline, result = native(**inputs), normalized(**inputs)
        for name in baseline:
            self.assert_tensor_bytes_equal(baseline[name], result[name])

    def test_identity_normalization_preserves_input_and_parameter_gradients(self):
        native, normalized = self.pair(self.statistics(identity=True))
        first = self.inputs()
        second = deepcopy(first)
        for values in (first, second):
            for name in ("history_visual_latent", "history_state", "candidate_actions"):
                values[name].requires_grad_(True)
        outputs = native(**first), normalized(**second)
        for value in outputs:
            (value["pred_state_delta"].square().mean() + value["pred_future_visual_latent"].square().mean()).backward()
        for name in ("history_visual_latent", "history_state", "candidate_actions"):
            self.assert_tensor_bytes_equal(first[name].grad, second[name].grad)
        for (name, left), (other, right) in zip(native.named_parameters(), normalized.named_parameters()):
            self.assertEqual(name, other)
            self.assert_tensor_bytes_equal(left.grad, right.grad)

    def test_only_visual_projection_inputs_change_and_masked_values_are_zero(self):
        stats = self.statistics()
        _, model = self.pair(stats)
        inputs = self.inputs()
        inputs["history_visual_valid"][0, 1, 0] = False
        inputs["history_visual_latent"][0, 1, 0] = float("nan")
        seen, handles = {}, []
        def record(view):
            def hook(module, args):
                seen[view] = args[0].detach().clone()
                return None
            return hook
        for i, layer in enumerate(model.view_projections):
            handles.append(layer.register_forward_pre_hook(record(i)))
        handles.append(model.action_projection.register_forward_pre_hook(record("action")))
        handles.append(model.history_projection.register_forward_pre_hook(record("history")))
        try:
            with torch.no_grad(): model(**inputs)
        finally:
            for handle in handles: handle.remove()
        raw = inputs["history_visual_latent"]
        mask = inputs["history_visual_valid"]
        mean, scale = torch.tensor(stats["mean"], dtype=torch.float32), torch.tensor(stats["scale"], dtype=torch.float32)
        expected = torch.where(mask[..., None], (torch.where(mask[..., None], raw, 0.) - mean) / scale, 0.)
        for view in range(2): self.assert_tensor_bytes_equal(seen[view], expected[:, :, view])
        self.assertEqual(torch.count_nonzero(seen[0][0, 1]).item(), 0)
        self.assert_tensor_bytes_equal(seen["action"], inputs["candidate_actions"][:, 0])
        self.assert_tensor_bytes_equal(seen["history"][..., -16:-8], inputs["history_state"])
        self.assert_tensor_bytes_equal(seen["history"][..., -8:], inputs["history_state_valid"].float())

    def test_nonidentity_backward_is_finite_without_updating_parameters_or_buffers(self):
        _, model = self.pair(self.statistics())
        inputs = self.inputs()
        for name in ("history_visual_latent", "history_state", "candidate_actions"):
            inputs[name].requires_grad_(True)
        before = deepcopy(model.state_dict())
        outputs = model(**inputs)
        (outputs["pred_state_delta"].square().mean() + outputs["pred_future_visual_latent"].square().mean()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for name in ("history_visual_latent", "history_state", "candidate_actions"):
            self.assertTrue(bool(torch.isfinite(inputs[name].grad).all()), name)
        for name, value in model.state_dict().items():
            self.assert_tensor_bytes_equal(value, before[name])
        self.assertTrue(all(value.grad is None for value in model.buffers()))

    def test_invalid_observable_forward_values_are_rejected_without_imputation(self):
        _, model = self.pair(self.statistics())
        for mode in ("visual_nan", "action_nan", "state_nan", "mask_dtype", "shape", "extra_key"):
            inputs = self.inputs()
            if mode == "visual_nan": inputs["history_visual_latent"][0, 0, 0, 0] = float("nan")
            elif mode == "action_nan": inputs["candidate_actions"][0, 0, 0, 0] = float("nan")
            elif mode == "state_nan": inputs["history_state"][0, 0, 0] = float("nan")
            elif mode == "mask_dtype": inputs["history_visual_valid"] = inputs["history_visual_valid"].float()
            elif mode == "shape": inputs["history_visual_latent"] = inputs["history_visual_latent"][:, :-1]
            else: inputs["future_visual_latent"] = torch.zeros(2, 3, 2, 2048)
            with self.subTest(mode=mode), self.assertRaises((ValueError, TypeError)):
                model(**inputs)

    def test_raw_skip_and_caller_inputs_are_unchanged_by_nonidentity_transform(self):
        _, model = self.pair(self.statistics())
        inputs = self.inputs()
        inputs["history_visual_valid"][0, -1, 1] = False
        inputs["history_visual_latent"][0, -1, 1] = float("nan")
        before = deepcopy(inputs)
        with torch.no_grad():
            model.visual_residual_head.weight.zero_()
            model.visual_residual_head.bias.zero_()
            result = model(**inputs)
        skip = torch.where(inputs["history_visual_valid"][..., None], inputs["history_visual_latent"], 0.)[:, -1, None, None]
        self.assert_tensor_bytes_equal(result["pred_future_visual_latent"], skip.expand_as(result["pred_future_visual_latent"]))
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor): self.assert_tensor_bytes_equal(value, before[name])
            else: self.assertEqual(value, before[name])

    def test_normalization_adds_only_frozen_persistent_buffers_with_owned_statistics(self):
        stats = self.statistics()
        native, model = self.pair(stats)
        self.assertEqual(sum(p.numel() for p in native.parameters()), sum(p.numel() for p in model.parameters()))
        self.assertEqual(set(dict(model.named_buffers())), {"visual_mean", "visual_scale"})
        original_hash = model.normalization_sha256
        for name, buffer in model.named_buffers():
            self.assertEqual(buffer.shape, (2, 2048))
            self.assertEqual(buffer.dtype, torch.float32)
            self.assertFalse(buffer.requires_grad)
            self.assertIn(name, model.state_dict())
        stats["mean"][0][0] = -999.
        self.assertNotEqual(model.visual_mean[0, 0].item(), -999.)
        self.assertEqual(model.normalization_sha256, original_hash)

    def test_state_dict_roundtrip_preserves_bound_normalizer_and_predictions(self):
        stats = self.statistics()
        _, model = self.pair(stats)
        _, restored = self.pair(stats)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        inputs = self.inputs()
        with torch.no_grad(): first, second = model(**inputs), restored(**inputs)
        for name in first: self.assert_tensor_bytes_equal(first[name], second[name])
        self.assertEqual(model.normalization_sha256, restored.normalization_sha256)

    def test_missing_or_mismatched_normalizer_cannot_load_even_non_strict(self):
        stats = self.statistics()
        _, model = self.pair(stats)
        for mode in ("missing", "wrong_mean", "wrong_scale", "non_strict", "assign"):
            state = deepcopy(model.state_dict())
            state[next(iter(dict(model.named_parameters())))].add_(3)
            kwargs = {}
            if mode == "missing": del state["visual_mean"]
            elif mode == "wrong_mean": state["visual_mean"][0, 0] += 1
            elif mode == "wrong_scale": state["visual_scale"][0, 0] *= 2
            elif mode == "non_strict":
                del state["visual_mean"]
                kwargs["strict"] = False
            else: kwargs["assign"] = True
            before = deepcopy(model.state_dict())
            with self.subTest(mode=mode), self.assertRaises((ValueError, RuntimeError)):
                model.load_state_dict(state, **kwargs)
            for name, value in before.items(): self.assert_tensor_bytes_equal(model.state_dict()[name], value)

    def test_legacy_state_dict_and_invalid_constructor_statistics_are_rejected(self):
        stats = self.statistics()
        native, model = self.pair(stats)
        with self.assertRaises((ValueError, RuntimeError)): model.load_state_dict(native.state_dict())
        for name in ("mean", "scale"):
            bad = deepcopy(stats)
            bad[name][0][0] = float("nan")
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalization.NormalizedLiberoWorldModel(self.config, REGISTRY, bad)

    def test_projection_overflow_is_rejected_before_any_linear_call(self):
        _, model = self.pair(self.statistics())
        inputs = self.inputs()
        inputs["history_visual_latent"][0, 0, 0, 0] = torch.finfo(torch.float32).max
        before = deepcopy(inputs["history_visual_latent"])
        seen = []
        handle = model.view_projections[0].register_forward_pre_hook(lambda module, args: seen.append(True))
        try:
            with self.assertRaisesRegex(ValueError, "normalized projection input"):
                model(**inputs)
        finally:
            handle.remove()
        self.assertEqual(seen, [])
        self.assert_tensor_bytes_equal(inputs["history_visual_latent"], before)

    def test_constructor_statistics_not_mutated_current_buffer_own_load_binding(self):
        _, model = self.pair(self.statistics())
        pristine = deepcopy(model.state_dict())
        with torch.no_grad(): model.visual_mean.add_(1)
        corrupted = deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "constructor-bound statistics"):
            model.load_state_dict(corrupted)
        # A correct bound checkpoint restores buffers; a corrupt current buffer
        # cannot redefine which incoming normalization artifact is acceptable.
        model.load_state_dict(pristine)
        self.assert_tensor_bytes_equal(model.visual_mean, pristine["visual_mean"])


if __name__ == "__main__":
    unittest.main()
