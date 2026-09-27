from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SCHEMA = "project_2026_openpi_compat_pack_v0"
TACTILE_NAMES = [
    "estimated_contact_flag",
    "contact_estimator_confidence",
    "estimated_image_distance_px",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_pack(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected schema: {manifest.get('schema')!r}")
    return manifest_path.parent, manifest


def finite_values(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def summary(values: np.ndarray) -> dict[str, Any]:
    finite = finite_values(values)
    if finite.size == 0:
        return {
            "count": int(np.asarray(values).size),
            "finite_count": 0,
            "unique_count": 0,
        }
    unique = np.unique(finite)
    return {
        "count": int(np.asarray(values).size),
        "finite_count": int(finite.size),
        "unique_count": int(unique.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
        "unique_values_preview": [float(x) for x in unique[:10]],
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def cohen_d(a: np.ndarray, b: np.ndarray) -> float | None:
    a = finite_values(a)
    b = finite_values(b)
    if a.size < 2 or b.size < 2:
        return None
    pooled = np.sqrt(((a.size - 1) * np.var(a) + (b.size - 1) * np.var(b)) / max(a.size + b.size - 2, 1))
    if float(pooled) < 1e-12:
        return None
    return float((np.mean(a) - np.mean(b)) / pooled)


def group_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in sorted(int(x) for x in np.unique(labels)):
        mask = labels == label
        out[str(label)] = summary(values[mask])
    if {1, 2}.issubset(set(int(x) for x in np.unique(labels))):
        out["feed_minus_hold_cohen_d"] = cohen_d(values[labels == 2], values[labels == 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tactile/contact signal value in an OpenPI-compatible pack.")
    parser.add_argument("pack", type=Path, help="OpenPI-compatible pack directory or manifest.")
    parser.add_argument("--out", type=Path, required=True, help="JSON audit report path.")
    args = parser.parse_args()

    pack_dir, manifest = resolve_pack(args.pack)
    arrays_path = pack_dir / str(manifest.get("output", {}).get("arrays_npz", "openpi_arrays.npz"))
    if not arrays_path.exists():
        raise FileNotFoundError(f"arrays not found: {arrays_path}")
    arrays = dict(np.load(arrays_path))

    tactile = arrays["tactile_context"].astype(np.float64)
    tactile_valid = arrays["tactile_context_valid"].astype(bool)
    piper = arrays["piper_intent_id"].astype(np.int64)
    feed_binary = (piper == 2).astype(np.float64)
    elite = arrays["elite_tcp_delta_6d"].astype(np.float64)
    elite_xyz = elite[:, :3]
    elite_l2 = np.linalg.norm(elite_xyz, axis=1)
    elite_linf = np.max(np.abs(elite_xyz), axis=1)
    piper_state = arrays["piper_state"].astype(np.float64)
    task_one_hot = arrays["task_one_hot"].astype(np.float64)

    tactile_report: dict[str, Any] = {}
    recommendations: list[str] = []
    for i, name in enumerate(TACTILE_NAMES):
        values = tactile[:, i]
        field = {
            "summary": summary(values),
            "valid_true_count": int(tactile_valid[:, i].sum()),
            "valid_false_count": int((~tactile_valid[:, i]).sum()),
            "by_piper_intent_id": group_summary(values, piper),
            "correlation": {
                "piper_feed_binary": pearson(values, feed_binary),
                "elite_xyz_l2": pearson(values, elite_l2),
                "elite_xyz_linf": pearson(values, elite_linf),
                "elite_delta_x": pearson(values, elite_xyz[:, 0]),
                "elite_delta_y": pearson(values, elite_xyz[:, 1]),
                "elite_delta_z": pearson(values, elite_xyz[:, 2]),
                "piper_step": pearson(values, piper_state[:, 0]),
                "piper_insertion_length": pearson(values, piper_state[:, 1]),
                "task_left_one_hot": pearson(values, task_one_hot[:, 0]),
            },
        }
        tactile_report[name] = field
        if field["summary"].get("unique_count") == 1:
            recommendations.append(f"{name} is constant and cannot help this supervised split.")

    contact_flag_unique = tactile_report["estimated_contact_flag"]["summary"].get("unique_count", 0)
    confidence_piper_corr = tactile_report["contact_estimator_confidence"]["correlation"]["piper_feed_binary"]
    distance_piper_corr = tactile_report["estimated_image_distance_px"]["correlation"]["piper_feed_binary"]
    distance_elite_corr = tactile_report["estimated_image_distance_px"]["correlation"]["elite_xyz_l2"]

    if contact_flag_unique == 1:
        recommendations.append(
            "The contact flag has no positive examples; tactile ablation is expected to look similar unless "
            "confidence/distance carry independent signal."
        )
    weak_threshold = 0.15
    corrs = [x for x in [confidence_piper_corr, distance_piper_corr, distance_elite_corr] if x is not None]
    if corrs and all(abs(x) < weak_threshold for x in corrs):
        recommendations.append(
            "The varying tactile fields have weak linear correlation with Piper intent and Elite movement scale; "
            "use them as interface placeholders, not as validated useful tactile conditioning."
        )
    recommendations.append(
        "Next useful check: create harder/contact-rich examples or a real-image distance estimator before expecting "
        "tactile conditioning to improve policy metrics."
    )

    report = {
        "pack": str(args.pack),
        "samples": int(tactile.shape[0]),
        "piper_intent_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(piper, return_counts=True))},
        "tactile_fields": tactile_report,
        "elite_delta_xyz_l2": summary(elite_l2),
        "elite_delta_xyz_linf": summary(elite_linf),
        "recommendations": recommendations,
    }
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
