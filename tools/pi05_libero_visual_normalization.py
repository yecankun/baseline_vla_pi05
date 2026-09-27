"""Train-only preprojection visual normalization for the native LIBERO model.

This is not a PI0.5 processor change: only the two downstream view Linear inputs
are normalized. The raw visual skip, prediction/target units, state, actions,
activations, modules and trainable parameter names remain unchanged. The two
persistent buffers require a NEW checkpoint schema binding this statistics
artifact and its provenance; do not label these as old native checkpoints.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

import numpy as np
import torch

if __package__ in {None, ""}:
    from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, _tensor
    from pi05_libero_world_model_adapter import ARRAY_NAMES, SPLIT_SCHEMA
else:
    from .pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, _tensor
    from .pi05_libero_world_model_adapter import ARRAY_NAMES, SPLIT_SCHEMA


STATISTICS_SCHEMA = "pi05_libero_train_visual_normalization_v1"
NORMALIZED_MODEL_SCHEMA = "pi05_libero_native_visual_normalized_world_model_v1"
SCALE_FLOOR = 0.1
STATISTICS_KEYS = frozenset({"schema", "views", "visual_dim", "scale_floor", "fitting_scope", "variance",
    "projection_compute_dtype", "raw_visual_skip_unchanged", "mean", "std", "scale", "train_episode_indices",
    "train_row_indices", "train_record_count", "train_episodes", "feature_pack_manifest_sha256", "split_sha256",
    "source_metadata_sha256", "source_array_sha256", "train_visual_sha256", "train_visual_valid_sha256"})
EPISODE_KEYS = frozenset({"episode_index", "task_id", "source_trajectory_id", "leakage_group_id",
                          "record_count", "complete_episode"})


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def statistics_sha256(payload):
    """Hash the entire JSON statistics object; there is no self-referential hash."""
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _array_sha(value):
    array = np.asarray(value)
    digest = hashlib.sha256(_canonical({"dtype": str(array.dtype), "shape": list(array.shape)}).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _hash(value, name):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{name} must be a lowercase SHA256")


def _ids(values, name, *, sorted_required=False):
    _require(type(values) is list and bool(values) and all(type(v) is int and v >= 0 for v in values),
             f"{name} must be a nonempty list of nonnegative integer IDs")
    _require(len(values) == len(set(values)), f"{name} contains duplicates")
    _require(not sorted_required or values == sorted(values), f"{name} must be sorted")
    return values


def _episode(row):
    _require(type(row) is dict and set(row) == EPISODE_KEYS, "complete native episode declaration required")
    _require(type(row["episode_index"]) is int and row["episode_index"] >= 0
             and type(row["task_id"]) is int and 0 <= row["task_id"] < 10
             and type(row["record_count"]) is int and row["record_count"] >= 2
             and row["complete_episode"] is True, "invalid/incomplete native episode")
    _require(all(type(row[k]) is str and row[k].strip() for k in ("source_trajectory_id", "leakage_group_id")),
             "explicit source trajectory and leakage group required")


def validate_statistics(payload, views, visual_dim):
    """Fail closed on the standalone schema and return an owned validated copy.

    This verifies internal consistency, not that arbitrary supplied hashes are
    authentic. The caller binds the returned artifact to the fitted source and
    new checkpoint. There is deliberately no identity/test bypass in the model.
    """
    _require(type(views) is int and views == 2 and type(visual_dim) is int and visual_dim > 0,
             "native statistics require two views and a positive visual dimension")
    _require(type(payload) is dict and set(payload) == STATISTICS_KEYS, "visual statistics keys differ")
    _require(payload["schema"] == STATISTICS_SCHEMA and type(payload["views"]) is int and payload["views"] == views
             and type(payload["visual_dim"]) is int and payload["visual_dim"] == visual_dim,
             "visual statistics schema/dimensions differ")
    _require(type(payload["scale_floor"]) is float and payload["scale_floor"] == SCALE_FLOOR,
             "the prospectively fixed raw-feature scale floor is 0.1")
    _require(payload["fitting_scope"] == "unique_complete_train_rows_only"
             and payload["variance"] == "population_ddof0_float64"
             and payload["projection_compute_dtype"] == "float32"
             and payload["raw_visual_skip_unchanged"] is True, "visual normalization semantics differ")
    values = {}
    for key in ("mean", "std", "scale"):
        value = payload[key]
        _require(type(value) is list and len(value) == views and all(type(row) is list and len(row) == visual_dim for row in value),
                 f"{key} requires explicit per-view/per-coordinate lists")
        _require(all(type(v) in (float, int) and math.isfinite(v) for row in value for v in row),
                 f"{key} requires finite JSON numbers, not coerced values")
        values[key] = np.asarray(value, dtype=np.float64)
    _require(bool((values["std"] >= 0).all()) and np.array_equal(values["scale"], np.maximum(values["std"], SCALE_FLOOR)),
             "scale must equal max(population std,0.1) exactly")
    with np.errstate(over="ignore"):
        representable = all(np.isfinite(v.astype(np.float32)).all() for v in values.values())
    _require(representable, "statistics must remain finite when used by float32 projections")
    train = _ids(payload["train_episode_indices"], "train_episode_indices", sorted_required=True)
    rows = _ids(payload["train_row_indices"], "train_row_indices")
    _require(type(payload["train_record_count"]) is int and payload["train_record_count"] == len(rows),
             "unique training row count differs")
    episodes = payload["train_episodes"]
    _require(type(episodes) is list and len(episodes) == len(train), "training episode provenance differs")
    for row in episodes:
        _episode(row)
    _require([row["episode_index"] for row in episodes] == train
             and sum(row["record_count"] for row in episodes) == len(rows), "incomplete/duplicate training provenance")
    for key in ("feature_pack_manifest_sha256", "split_sha256", "source_metadata_sha256",
                "train_visual_sha256", "train_visual_valid_sha256"):
        _hash(payload[key], key)
    arrays = payload["source_array_sha256"]
    _require(type(arrays) is dict and set(arrays) == ARRAY_NAMES, "source array hash allowlist differs")
    for key, value in arrays.items():
        _hash(value, key)
    return deepcopy(payload)


def fit_visual_statistics(pack, split, scale_floor=SCALE_FLOOR):
    """Fit float64 population statistics from each complete train row once.

    The pack/split must already have passed the native loader. Only train rows
    of visual_latent/visual_valid are indexed; validation feature values/masks
    are never indexed or rehashed here. Full-array hashes are the native
    manifest's already-verified provenance, not newly read validation values.
    ID/frame metadata may be inspected for completeness and split integrity.
    Caller-owned source/pack lifetime and before/after file pins remain intact.
    """
    _require(type(scale_floor) is float and scale_floor == SCALE_FLOOR, "scale floor is fixed at 0.1")
    _require(type(split) is dict and set(split) == {"schema", "feature_pack_manifest_sha256", "train_episode_indices",
                 "validation_episode_indices", "split_sha256"}, "validated native split required")
    _require(split["schema"] == SPLIT_SCHEMA and split["feature_pack_manifest_sha256"] == pack.manifest_sha256
             and pack._validated_splits.get(split["split_sha256"]) == json.dumps(split, sort_keys=True),
             "statistics require the native loader's validated hash-bound split")
    train = sorted(_ids(split["train_episode_indices"], "train split"))
    validation = _ids(split["validation_episode_indices"], "validation split")
    _require(not set(train) & set(validation) and set(train) | set(validation) == set(pack.episodes),
             "split must be disjoint and cover complete registered episodes")
    for eid, row in pack.episodes.items():
        _episode(row)
        _require(type(eid) is int and eid == row["episode_index"], "episode registry identity differs")
    for key in ("source_trajectory_id", "leakage_group_id"):
        _require(not {pack.episodes[e][key] for e in train} & {pack.episodes[e][key] for e in validation},
                 f"{key} leaks across split")
    arrays = pack.arrays
    n, views, dimension = arrays["visual_latent"].shape
    _require(views == 2 and dimension == pack.visual_dim and arrays["visual_latent"].dtype == np.float32
             and arrays["visual_valid"].shape == (n, views) and arrays["visual_valid"].dtype == np.bool_,
             "native visual array shape/dtype differs")
    selected = []
    for eid in train:
        rows = np.asarray(pack.indices[eid])
        declaration = pack.episodes[eid]
        _require(rows.ndim == 1 and rows.dtype.kind in "iu" and len(rows) == declaration["record_count"]
                 and bool((rows >= 0).all()) and bool((rows < n).all()) and len(np.unique(rows)) == len(rows),
                 "training episode row indices are duplicate/incomplete/out of range")
        _require(np.array_equal(arrays["frame_index"][rows], np.arange(len(rows)))
                 and bool((arrays["episode_index"][rows] == eid).all())
                 and bool((arrays["task_id"][rows] == declaration["task_id"]).all()),
                 "training rows must contain the complete native episode in frame order")
        _require(int(np.count_nonzero(arrays["episode_index"] == eid)) == len(rows),
                 "episode selection omits registered source rows")
        selected.extend(map(int, rows))
    _require(len(selected) == len(set(selected)), "training rows must be unique, not repeated windows")
    indices = np.asarray(selected, dtype=np.int64)
    # Advanced indexing copies ONLY declared train rows, including final rows.
    visual = np.asarray(arrays["visual_latent"][indices])
    valid = np.asarray(arrays["visual_valid"][indices])
    _require(visual.shape == (len(indices), views, dimension) and visual.dtype == np.float32
             and valid.shape == (len(indices), views) and valid.dtype == np.bool_
             and bool(valid.all()) and bool(np.isfinite(visual).all()),
             "all selected training view masks must be valid and all features finite")
    values = visual.astype(np.float64)
    mean = values.mean(axis=0)
    std = np.sqrt(np.square(values - mean).mean(axis=0))
    result = dict(schema=STATISTICS_SCHEMA, views=views, visual_dim=dimension, scale_floor=SCALE_FLOOR,
        fitting_scope="unique_complete_train_rows_only", variance="population_ddof0_float64",
        projection_compute_dtype="float32", raw_visual_skip_unchanged=True,
        mean=mean.tolist(), std=std.tolist(), scale=np.maximum(std, SCALE_FLOOR).tolist(),
        train_episode_indices=train, train_row_indices=selected, train_record_count=len(selected),
        train_episodes=[deepcopy(pack.episodes[eid]) for eid in train],
        feature_pack_manifest_sha256=pack.manifest_sha256, split_sha256=split["split_sha256"],
        source_metadata_sha256=pack.manifest["source"]["metadata_sha256"],
        source_array_sha256={key: pack.manifest["arrays"][key]["sha256"] for key in sorted(ARRAY_NAMES)},
        train_visual_sha256=_array_sha(visual), train_visual_valid_sha256=_array_sha(valid))
    return validate_statistics(result, views, dimension)


class NormalizedLiberoWorldModel(LiberoWorldModel):
    """Same native learned modules; only view projection inputs are normalized.

    visual_mean/visual_scale are persistent nontrainable buffers. Saving and
    loading requires a normalization-aware checkpoint/provenance envelope, not
    the old parameter-only optimizer checkpoint format.
    """

    def __init__(self, config: LiberoWorldModelConfig, task_registry: list[dict], statistics: dict):
        validated = validate_statistics(statistics, config.views, config.visual_dim)
        super().__init__(config, task_registry)
        self._visual_statistics = validated
        self.normalization_sha256 = statistics_sha256(validated)
        self.register_buffer("visual_mean", torch.tensor(validated["mean"], dtype=torch.float32), persistent=True)
        self.register_buffer("visual_scale", torch.tensor(validated["scale"], dtype=torch.float32), persistent=True)

    def metadata(self):
        result = super().metadata()
        result["schema"] = NORMALIZED_MODEL_SCHEMA
        result["second_normalization"] = "train-only raw visual latents before view Linear; state/action not renormalized"
        result["visual_normalization"] = dict(schema=STATISTICS_SCHEMA, statistics_sha256=self.normalization_sha256,
            scale_floor=SCALE_FLOOR, fitting_scope="unique_complete_train_rows_only", persistent_buffers=["visual_mean", "visual_scale"],
            raw_visual_skip_unchanged=True, target_units_unchanged=True, pi05_processor_changed=False,
            requires_new_checkpoint_schema=True, old_parameter_only_checkpoint_compatible=False)
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        _require(strict is True and assign is False, "normalized checkpoints require strict owned loading")
        for name in ("visual_mean", "visual_scale"):
            expected = torch.tensor(self._visual_statistics["mean" if name == "visual_mean" else "scale"], dtype=torch.float32)
            value = state_dict.get(name)
            _require(isinstance(value, torch.Tensor) and value.dtype == expected.dtype
                     and value.shape == expected.shape and not value.requires_grad
                     and torch.equal(value.detach().cpu().contiguous().view(torch.uint8),
                                     expected.detach().cpu().contiguous().view(torch.uint8)),
                     "checkpoint normalization buffers differ from constructor-bound statistics")
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def forward(self, *, history_visual_latent, history_visual_valid, history_state, history_state_valid,
                task_instruction, candidate_actions):
        # Faithful native forward below. The sole computational change is the
        # normalization/re-masking immediately before the two view projections.
        cfg = self.config
        if not isinstance(history_visual_latent, torch.Tensor) or history_visual_latent.ndim != 4:
            raise ValueError("history_visual_latent requires [B,T,2,D]")
        b = history_visual_latent.shape[0]
        if b <= 0:
            raise ValueError("empty batches are unsupported")
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError("native forward model requires float32 parameters")
        device = parameter.device
        _tensor(history_visual_latent, "history_visual_latent", (b, cfg.context_len, cfg.views, cfg.visual_dim), torch.float32, device)
        _tensor(history_visual_valid, "history_visual_valid", (b, cfg.context_len, cfg.views), torch.bool, device)
        _tensor(history_state, "history_state", (b, cfg.context_len, cfg.state_dim), torch.float32, device)
        _tensor(history_state_valid, "history_state_valid", (b, cfg.context_len, cfg.state_dim), torch.bool, device)
        if not isinstance(candidate_actions, torch.Tensor) or candidate_actions.ndim != 4 or candidate_actions.shape[1] <= 0:
            raise ValueError("candidate_actions requires nonempty [B,K,H,7]")
        k = candidate_actions.shape[1]
        _tensor(candidate_actions, "candidate_actions", (b, k, cfg.horizon, cfg.action_dim), torch.float32, device)
        if (type(task_instruction) is not list or len(task_instruction) != b
                or any(type(text) is not str or text not in self._task_encoding for text in task_instruction)):
            raise ValueError("task_instruction requires one exact known native instruction per batch row")
        if not bool(torch.isfinite(history_visual_latent[history_visual_valid]).all()):
            raise ValueError("valid visual latents must be finite")
        if not bool(torch.isfinite(history_state[history_state_valid]).all()):
            raise ValueError("valid state coordinates must be finite")
        if not bool(torch.isfinite(candidate_actions).all()):
            raise ValueError("all candidate actions must be finite; actions cannot be imputed")
        for name in ("visual_mean", "visual_scale"):
            _tensor(getattr(self, name), name, (cfg.views, cfg.visual_dim), torch.float32, device)
            _require(not getattr(self, name).requires_grad and bool(torch.isfinite(getattr(self, name)).all()),
                     "normalization buffers must remain finite and nontrainable")
        _require(bool((self.visual_scale >= SCALE_FLOOR).all()), "normalization scale fell below the frozen floor")

        visual = torch.where(history_visual_valid[..., None], history_visual_latent, 0.0)
        state = torch.where(history_state_valid, history_state, 0.0)
        projected_visual = torch.where(history_visual_valid[..., None],
                                       (visual - self.visual_mean) / self.visual_scale, 0.0)
        _require(bool(torch.isfinite(projected_visual).all()), "normalized projection input must remain finite float32")
        views = [torch.tanh(layer(projected_visual[:, :, index])) for index, layer in enumerate(self.view_projections)]
        history = torch.cat([*views, history_visual_valid.to(torch.float32), state, history_state_valid.to(torch.float32)], dim=-1)
        history = self.dropout(torch.tanh(self.history_projection(history)))
        _, context = self.context_gru(history)
        task_ids = torch.tensor([self._task_encoding[text] for text in task_instruction], dtype=torch.long, device=device)
        initial = torch.tanh(self.context_task_projection(torch.cat([context[-1], self.task_embedding(task_ids)], dim=-1)))
        initial = initial[:, None, :].expand(b, k, cfg.hidden_dim).reshape(1, b * k, cfg.hidden_dim).contiguous()
        actions = candidate_actions.reshape(b * k, cfg.horizon, cfg.action_dim)
        actions = self.dropout(torch.tanh(self.action_projection(actions)))
        future, _ = self.action_gru(actions, initial)
        future = self.dropout(future).reshape(b, k, cfg.horizon, cfg.hidden_dim)
        residual = self.visual_residual_head(future).reshape(b, k, cfg.horizon, cfg.views, cfg.visual_dim)
        outputs = {"pred_future_visual_latent": visual[:, -1, None, None] + residual,
                   "pred_state_delta": self.state_residual_head(future)}
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in outputs.values()):
            raise ValueError("native world-model output must remain finite float32")
        return outputs
