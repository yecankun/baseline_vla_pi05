"""Read-only input/normalization audit of the frozen DP scorer experiment.

Writes a separate diagnostic report and copies observed RGB to PNG. No forward,
optimizer, environment, input replacement, threshold fitting or test access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

if __package__:
    from . import train_diffusion_pusht_action_effect as training
else:
    import train_diffusion_pusht_action_effect as training

shared, pair = training.shared, training.pair
OUT = shared.ROOT / "simulation_output/diffusion_pusht_scorer_input_audit_v1"
FIT = shared.ROOT / "simulation_output/diffusion_pusht_scorer_fit_diagnostic_v1"
FOCUS = (520005, 520006)


def quantiles(x):
    return dict(zip(("min", "median", "p90", "p95", "max"),
                    map(float, np.quantile(x, [0, .5, .9, .95, 1]))))


def key_list(x):
    return [int(v) for v in x]


def distances(query, reference, *, grid=False):
    q, r = query.reshape(len(query), -1).astype(np.float64), reference.reshape(len(reference), -1).astype(np.float64)
    denominator = np.square(q).sum(1)[:, None] + np.square(r).sum(1)[None]
    squared = np.maximum(denominator - 2 * q @ r.T, 0)
    return squared / denominator if grid else np.sqrt(squared / q.shape[1])


def neighbor_report(train, validation, train_sources, train_keys, *, grid=False):
    loo = distances(train, train, grid=grid)
    loo[train_sources[:, None] == train_sources[None]] = np.inf
    baseline = loo.min(1)
    matrix = distances(validation, train, grid=grid)
    index = matrix.argmin(1)
    rows = [{"nearest_training_key": key_list(train_keys[k]), "distance": float(matrix[i, k]),
             "fraction_training_LOSO_at_or_below": float(np.mean(baseline <= matrix[i, k]))}
            for i, k in enumerate(index)]
    return {"training_leave_source_out_distance": quantiles(baseline), "validation_rows": rows}


def run(args):
    started = time.perf_counter()
    data = training.DPData(args.data_root)
    original = shared.read(args.training_root / "report.json")
    if original["status"] != "completed_fixed_DP_paired_training_and_validation" or original["contract"] != data.contract:
        raise ValueError("requires the unchanged completed experiment")
    with np.load(args.training_root / "data_snapshot.npz", allow_pickle=False) as saved:
        current = data.snapshot()
        if set(saved.files) != set(current) or any(not np.array_equal(saved[k], v) for k, v in current.items()):
            raise ValueError("current pack differs from frozen training snapshot")
    norm = data.contract["normalization"]
    expected = {**norm["input"], "target_mean": norm["target"]["mean"],
                "target_std": norm["target"]["std"], "goal": data.goal}
    seeds = data.contract["protocol"]["paired_training_seeds"]
    for seed in seeds:
        for arm in pair.ARMS:
            saved = torch.load(args.training_root / str(seed) / arm / "final.pt", map_location="cpu", weights_only=False)
            if saved["contract"] != data.contract or saved["step"] != 2000 or saved["arm"] != arm or saved["training_seed"] != seed:
                raise ValueError("frozen checkpoint identity changed")
            for name, value in expected.items():
                if not torch.equal(saved["model"][name], torch.as_tensor(value, dtype=torch.float32)):
                    raise ValueError(f"checkpoint normalization/goal mismatch: {name}")
    packs = {"train": data.train, "validation": data.validation}
    keys, raw_rgb, stats, normalized = {}, {}, {}, {}
    packing_cast = {field: {"source_dtypes": set(), "maximum_absolute_float32_rounding": 0.0}
                    for field in ("agent_xy", "actions")}
    for split, pack in packs.items():
        keys[split] = np.array([(r["source_seed"], r["anchor_step"]) for r in pack.contexts], dtype=np.int64)
        for i, row in enumerate(pack.contexts):
            path = args.data_root / row["source_attempt"] / f"context_{row['anchor_step']}.npz"
            with np.load(path, allow_pickle=False) as saved:
                observed = training.data_source.vision.observe_rgb(saved["pixels"])
                native = {field: saved[field].astype(np.float32) for field in packing_cast}
                for field, audit in packing_cast.items():
                    audit["source_dtypes"].add(str(saved[field].dtype))
                    delta = np.abs(saved[field].astype(np.float64) - native[field].astype(np.float64)).max()
                    audit["maximum_absolute_float32_rounding"] = max(audit["maximum_absolute_float32_rounding"], float(delta))
                if (not observed.valid or not np.array_equal(observed.features.values[0], pack.observations["current"][i])
                        or any(not np.array_equal(native[field], pack.observations[field][i]) for field in native)):
                    raise ValueError(f"RGB/native input differs after the original pack float32 cast: {row['source_seed']}/{row['anchor_step']}")
                raw_rgb[tuple(keys[split][i])] = saved["pixels"].copy()
        obs = pack.observations
        z = {}
        for field, prefix in (("agent_xy", "state"), ("actions", "action")):
            mean, std = (np.asarray(norm["input"][prefix + suffix], dtype=np.float32) for suffix in ("_mean", "_std"))
            z[field] = (obs[field] - mean) / std
        normalized[split] = z
        stats[split] = {
            "contexts": len(pack.contexts), "sources": len(np.unique(keys[split][:, 0])),
            "grid_range": [float(obs["current"].min()), float(obs["current"].max())],
            "grid_on_one_sixteenth_lattice": bool(np.equal(obs["current"] * 16, np.round(obs["current"] * 16)).all()),
            "visible_pixels": quantiles(obs["current"].sum((1, 2)) * 16),
            **{field: {"native_min_xy": obs[field].reshape(-1, 2).min(0).tolist(),
                       "native_max_xy": obs[field].reshape(-1, 2).max(0).tolist(),
                       "normalized_abs": quantiles(np.abs(z[field])),
                       "outside_native_0_512": int(((obs[field] < 0) | (obs[field] > 512)).sum())}
               for field in z},
        }
    tr, val = data.train.observations, data.validation.observations
    tk, vk = keys["train"], keys["validation"]
    grid = neighbor_report(tr["current"], val["current"], tk[:, 0], tk, grid=True)
    state = neighbor_report(normalized["train"]["agent_xy"], normalized["validation"]["agent_xy"], tk[:, 0], tk)
    action_keys = np.array([[int(s), int(a), k] for s, a in tk for k in range(5)])
    actions = neighbor_report(normalized["train"]["actions"].reshape(-1, 8, 2),
                              normalized["validation"]["actions"].reshape(-1, 8, 2),
                              action_keys[:, 0], action_keys)
    seen = (tr["current"] > 0).any(0) | (data.goal > 0)
    loo_unseen = []
    for i, (source, _) in enumerate(tk):
        support = (tr["current"][tk[:, 0] != source] > 0).any(0) | (data.goal > 0)
        loo_unseen.append(float(tr["current"][i][~support].sum() / tr["current"][i].sum()))
    mse = {arm: [] for arm in pair.ARMS}
    for seed in seeds:
        with np.load(args.fit_root / f"predictions_{seed}.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["validation_keys"], vk):
                raise ValueError("diagnostic prediction keys changed")
            for arm in pair.ARMS:
                mse[arm].append(np.square(saved["validation_" + arm + "_scores"] - data.validation.targets).mean(1))
    mse = {arm: np.mean(values, axis=0) for arm, values in mse.items()}
    rows = []
    for i, key in enumerate(vk):
        rows.append({
            "key": key_list(key), "focus": int(key[0]) in FOCUS,
            "native_agent_xy": val["agent_xy"][i].tolist(),
            "state_z": normalized["validation"]["agent_xy"][i].tolist(),
            "max_abs_action_z": float(np.abs(normalized["validation"]["actions"][i]).max()),
            "visible_pixels": int(val["current"][i].sum() * 16),
            "unseen_grid_mass_fraction": float(val["current"][i][~seen].sum() / val["current"][i].sum()),
            "grid_neighbor": grid["validation_rows"][i], "state_neighbor": state["validation_rows"][i],
            "candidate_neighbors": actions["validation_rows"][i * 5:(i + 1) * 5],
            "saved_prediction_mse_mean_training_seeds": {arm: float(values[i]) for arm, values in mse.items()},
        })
    args.out.mkdir(parents=True, exist_ok=False)
    images = []
    for row in rows:
        if row["focus"]:
            for key in (row["key"], row["grid_neighbor"]["nearest_training_key"]):
                name = f"observed_{key[0]}_{key[1]}.png"
                Image.fromarray(raw_rgb[tuple(key)]).save(args.out / name)
                if name not in images:
                    images.append(name)
    report = {
        "schema": "diffusion_pusht_scorer_input_audit_v1", "status": "completed_saved_input_audit",
        "data_root": str(args.data_root), "training_root": str(args.training_root), "fit_root": str(args.fit_root),
        "focus_sources": list(FOCUS), "normalization": norm,
        "frozen_snapshot_exact": True, "six_checkpoint_normalization_and_goal_exact": True,
        "source_RGB_grid_and_float32_packed_native_exact_contexts": sum(len(p.contexts) for p in packs.values()),
        "original_packing_float32_cast": {field: {**audit, "source_dtypes": sorted(audit["source_dtypes"])}
                                         for field, audit in packing_cast.items()},
        "input_ranges": stats,
        "nearest_training_reference": {
            "grid": grid["training_leave_source_out_distance"], "state": state["training_leave_source_out_distance"],
            "action_chunk": actions["training_leave_source_out_distance"],
            "unseen_grid_mass": quantiles(loo_unseen),
        },
        "distance_semantics": {"grid": "quadratic_soft_Dice_on_same_24x24_coordinates_no_alignment",
                               "state_action": "RMS_in_frozen_checkpoint_normalized_units",
                               "candidate_matching": "any_training_chunk_not_fixed_candidate_index",
                               "support": "positive_current_training_grid_or_fixed_goal_grid",
                               "training_reference": "nearest_context_or_chunk_excluding_entire_same_source",
                               "percentiles": "descriptive_context_or_chunk_ECDF_not_OOD_confidence_or_a_policy_gate"},
        "validation_contexts": rows, "observed_RGB_pngs": images,
        "visual_status": "not_viewed", "posthoc_diagnostic_only": True,
        "causal_attribution_established": False, "normalization_or_inputs_modified": False,
        "seconds": time.perf_counter() - started, "model_forward_calls": 0,
        "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0, "test_sources_executed": 0,
    }
    shared.write(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "input_ranges": stats, "focus": [
                          {k: r[k] for k in ("key", "max_abs_action_z", "visible_pixels",
                                            "unseen_grid_mass_fraction", "grid_neighbor", "state_neighbor")}
                          for r in rows if r["focus"]]}, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=training.OUT)
    parser.add_argument("--data-root", type=Path, default=training.data_source.OUT)
    parser.add_argument("--fit-root", type=Path, default=FIT)
    parser.add_argument("--out", type=Path, default=OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
