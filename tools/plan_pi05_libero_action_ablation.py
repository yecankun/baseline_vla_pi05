"""Offline metadata-only inventory and prospective split selection; never training.

Only hash-pinned cached Parquet ID/task/frame columns are projected. No state,
action, image, reward, normalization or model computation is performed. Missing
task coverage refuses partial split selection instead of biasing toward cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PLAN_SHA = "210ff8bbb75e3316620590cda9e5067f7447125acddb53ae7fe751bea01810ac"
META = "meta/episodes/chunk-000/file-000.parquet"
COLS = ["episode_index", "task_index", "frame_index", "index", "timestamp"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def integral(value):
    # Arrow metadata sometimes encodes integer coordinates as float64.
    if type(value) not in (int, float) or int(value) != value or value < 0:
        raise ValueError("nonnegative exact integer metadata required")
    return int(value)


def task_inventory(episodes, projected_rows, fps):
    metadata = {}
    for row in episodes:
        eid = integral(row["episode_index"])
        if eid in metadata:
            raise ValueError("duplicate episode metadata")
        count = integral(row["length"])
        begin, end = integral(row["dataset_from_index"]), integral(row["dataset_to_index"])
        if end - begin != count or count < 1:
            raise ValueError("episode length/index interval inconsistent")
        metadata[eid] = {"episode_index": eid, "record_count": count, "dataset_from_index": begin}
    by_episode = {}
    for row in projected_rows:
        if set(row) != set(COLS):
            raise ValueError("only metadata projection columns accepted, never action/state/reward")
        eid = integral(row["episode_index"])
        if eid not in metadata:
            raise ValueError("projected row from unknown episode")
        by_episode.setdefault(eid, []).append(row)
    entries = []
    for eid, rows in sorted(by_episode.items()):
        md = metadata[eid]
        rows = sorted(rows, key=lambda r: integral(r["frame_index"]))
        if len(rows) != md["record_count"]:
            raise ValueError("partial cached episode: stop, do not infer task from partial rows")
        tids = {integral(r["task_index"]) for r in rows}
        if len(tids) != 1 or not tids <= set(range(40)):
            raise ValueError("task identity changes inside episode or source task out of range")
        for i, row in enumerate(rows):
            if integral(row["frame_index"]) != i or integral(row["index"]) != md["dataset_from_index"] + i:
                raise ValueError("frame/global row gap or duplicate")
            if not abs(float(row["timestamp"]) - i / fps) <= 2e-5:
                raise ValueError("metadata timestamp off native fps")
        entries.append({"episode_index": eid, "source_task_index": tids.pop(), "record_count": len(rows)})
    return entries, sorted(set(metadata) - set(by_episode))


def select_split(entries, missing, plan):
    if missing:
        return {"status": "blocked_incomplete_task_metadata", "episodes": [], "split": None,
                "reason": "No selection from a cache-biased subset; complete task-identity inventory required"}
    spec = plan["source"]
    if len({r["episode_index"] for r in entries}) != len(entries):
        raise ValueError("duplicate task inventory episode")
    choices = []
    for row in entries:
        if (row["source_task_index"] == spec["source_task_index"]
                and row["episode_index"] not in spec["excluded_episode_indices"]
                and row["record_count"] >= spec["minimum_complete_rows"]):
            payload = f"{spec['revision']}/{spec['native_task_id']}/{row['episode_index']}/native-action-study-v1"
            choices.append({**row, "selection_hash": hashlib.sha256(payload.encode("ascii")).hexdigest()})
    choices.sort(key=lambda r: (r["selection_hash"], r["episode_index"]))
    train_n, val_n = spec["train_episode_count"], spec["validation_episode_count"]
    if len(choices) < train_n + val_n:
        return {"status": "blocked_insufficient_new_task_episodes", "episodes": [], "split": None,
                "eligible_episode_count": len(choices)}
    chosen = choices[:train_n + val_n]
    if sum(r["record_count"] for r in chosen) > spec["maximum_selected_complete_rows"]:
        raise ValueError("whole-episode row budget exceeded; no truncation or replacement")
    for i, row in enumerate(chosen):
        row["partition"] = "train" if i < train_n else "validation"
    return {"status": "metadata_split_selected_payload_audit_pending", "episodes": chosen,
            "split": {"train_episode_indices": [r["episode_index"] for r in chosen[:train_n]],
                      "validation_episode_indices": [r["episode_index"] for r in chosen[train_n:]]},
            "eligible_episode_count": len(choices), "training_ready": False,
            "family_independence_verified": False, "checkpoint_training_overlap_unknown": True}


def run(root, plan_path, out):
    if out.exists():
        raise FileExistsError("new metadata report directory required")
    if sha(plan_path) != PLAN_SHA:
        raise ValueError("prospective plan SHA mismatch")
    plan = read(plan_path)
    if sha(root / "report.json") != plan["source"]["cached_audit_report_sha256"]:
        raise ValueError("pinned cached source report SHA mismatch")
    prior = read(root / "report.json")
    if prior["source"]["revision"] != plan["source"]["revision"]:
        raise ValueError("source revision differs")
    filemap = {item["path"]: item for item in prior["files"]}
    verified = {"report.json": sha(root / "report.json")}
    def source_file(name):
        item = filemap[name]
        path = (root / item["local_path"]).resolve()
        if (not path.is_relative_to(root.resolve()) or path.stat().st_size != item["size"]
                or sha(path) != item["sha256"]):
            raise ValueError("cached metadata/projection source hash/size/path mismatch")
        verified[item["local_path"]] = item["sha256"]
        return path
    info = read(source_file("meta/info.json"))
    if info["fps"] != 10 or plan["source"]["metadata_columns_allowed"] != COLS:
        raise ValueError("fps/projection contract changed")
    # Only the remote environment needs pyarrow; pure selection tests are local.
    import pyarrow.parquet as pq
    meta_path = source_file(META)
    meta_names = pq.ParquetFile(meta_path).schema.names
    metadata_columns = ["episode_index", "length", "dataset_from_index", "dataset_to_index",
                        "data/chunk_index", "data/file_index"]
    episodes = pq.read_table(meta_path, columns=metadata_columns).to_pylist()
    if len(episodes) != info["total_episodes"]:
        raise ValueError("cached episode metadata is not the complete declared inventory")
    rows, scanned = [], []
    for name in sorted(filemap):
        if name.startswith("data/") and name.endswith(".parquet"):
            path = source_file(name)
            projected = pq.read_table(path, columns=COLS).to_pylist()
            rows.extend(projected)
            scanned.append({"source_path": name, "projected_columns": COLS, "projected_rows": len(projected)})
    entries, missing = task_inventory(episodes, rows, info["fps"])
    selection = select_split(entries, missing, plan)
    need = sorted({info["data_path"].format(chunk_index=integral(r["data/chunk_index"]),
                   file_index=integral(r["data/file_index"])) for r in episodes if integral(r["episode_index"]) in set(missing)})
    for name, value in verified.items():
        if sha(root / name) != value:
            raise ValueError("metadata changed during planning")
    result = {"schema": "libero_action_study_cached_metadata_report_v1", "status": selection["status"],
        "plan_sha256": PLAN_SHA, "source": prior["source"], "code_sha256": sha(__file__),
        "verified_cached_files": verified, "metadata_schema_columns": meta_names,
        "declared_episode_count": info["total_episodes"], "metadata_episode_count": len(episodes),
        "projected_cached_shards": scanned, "verified_task_inventory": entries,
        "missing_task_episode_count": len(missing), "missing_task_episode_indices": missing,
        "selected_task_known_new_eligible_count": sum(r["source_task_index"] == plan["source"]["source_task_index"]
             and r["episode_index"] not in plan["source"]["excluded_episode_indices"] for r in entries),
        "selection": selection, "missing_metadata_projection_shards": need,
        "next_acquisition_gate": "separate metadata-only task/frame projection acquisition; verify revision, file sizes/hashes and total budget first; no action/state/image/reward read for selection",
        "shard_sizes_not_audited": True, "network_downloads": 0, "images_decoded": 0,
        "actions_or_states_loaded": False, "features_extracted": 0, "optimizer_steps": 0,
        "training_ready": False, "family_independence_verified": False,
        "checkpoint_training_overlap_unknown": True}
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "metadata_episodes": len(episodes),
          "task_known": len(entries), "missing_task": len(missing), "missing_shards": len(need),
          "selected_episodes": len(selection["episodes"]), "report_sha256": sha(out / "report.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root, args.plan, args.out)
