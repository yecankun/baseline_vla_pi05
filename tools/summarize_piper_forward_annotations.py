from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_exclude(values: list[str]) -> set[tuple[str, int]]:
    excluded: set[tuple[str, int]] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected DATASET:EVENT_ID, got {value!r}")
        dataset, event_id = value.rsplit(":", 1)
        excluded.add((dataset, int(event_id)))
    return excluded


def finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def stat_block(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p25": None, "p75": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.mean(ordered)),
        "p25": percentile(ordered, 0.25),
        "p75": percentile(ordered, 0.75),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def round_floats(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, digits) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Piper forward-progress annotations.")
    parser.add_argument(
        "--annotations",
        default="docs/_real_alignment_20260707_piper_forward_annotations/annotations.jsonl",
    )
    parser.add_argument(
        "--out",
        default="docs/_real_alignment_20260707_piper_forward_annotations/summary_filtered.json",
    )
    parser.add_argument("--quality", action="append", default=["usable"], help="Quality labels to include.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude one event as DATASET:EVENT_ID. Repeat as needed.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.annotations))
    included_quality = set(args.quality)
    excluded = parse_exclude(args.exclude)
    excluded_rows: list[dict[str, Any]] = []
    included_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("dataset")), int(row.get("event_id", -1)))
        distance = finite_number(row.get("euclidean_px"))
        reason = None
        if key in excluded:
            reason = "explicit_exclude"
        elif row.get("quality") not in included_quality:
            reason = "quality_excluded"
        elif distance is None:
            reason = "missing_distance"
        if reason:
            item = dict(row)
            item["exclude_reason"] = reason
            excluded_rows.append(item)
            continue
        included_rows.append(row)

    distances = [float(row["euclidean_px"]) for row in included_rows]
    by_dataset: dict[str, list[float]] = defaultdict(list)
    by_burst: dict[str, list[float]] = defaultdict(list)
    by_dataset_burst: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in included_rows:
        dataset = str(row["dataset"])
        burst = int(row["burst_count"])
        distance = float(row["euclidean_px"])
        by_dataset[dataset].append(distance)
        by_burst[str(burst)].append(distance)
        by_dataset_burst[(dataset, burst)].append(distance)

    summary = {
        "annotations": str(Path(args.annotations)),
        "included_quality": sorted(included_quality),
        "explicit_excludes": sorted([f"{dataset}:{event_id}" for dataset, event_id in excluded]),
        "total_rows": len(rows),
        "included_rows": len(included_rows),
        "excluded_rows": len(excluded_rows),
        "quality_counts_all": dict(Counter(str(row.get("quality")) for row in rows)),
        "overall_euclidean_px": stat_block(distances),
        "by_dataset": {dataset: stat_block(values) for dataset, values in sorted(by_dataset.items())},
        "by_burst": {burst: stat_block(values) for burst, values in sorted(by_burst.items())},
        "by_dataset_burst": {
            f"{dataset}|burst{burst}": stat_block(values)
            for (dataset, burst), values in sorted(by_dataset_burst.items())
        },
        "excluded_events": [
            {
                "dataset": row.get("dataset"),
                "event_id": row.get("event_id"),
                "burst_count": row.get("burst_count"),
                "quality": row.get("quality"),
                "euclidean_px": row.get("euclidean_px"),
                "exclude_reason": row.get("exclude_reason"),
            }
            for row in excluded_rows
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(round_floats(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_path.with_suffix(".included.csv"), included_rows)
    write_csv(out_path.with_suffix(".excluded.csv"), excluded_rows)
    print(json.dumps(round_floats(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
