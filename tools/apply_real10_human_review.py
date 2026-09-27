"""Append explicit human review decisions; never infer labels or train a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from review_real10_event_windows import ReviewStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Append human revisions after checking their source snapshot")
    args = parser.parse_args()
    batch = read_json(args.batch)
    cases = {c["prediction"]["ui_index"]: c for c in read_json(batch["cases"])["cases"]}
    store = ReviewStore(batch["pack"], ".")  # No image/record/network/hardware reads.
    before = read_jsonl(store.path)
    seen, pending, decisions = set(), [], []
    marker = f"[{batch['review_id']}; user_via_chat]"
    for decision in batch["decisions"]:
        index = decision["ui_index"]
        if index in seen:
            raise ValueError(f"duplicate UI index {index}")
        seen.add(index)
        old = cases[index]["annotation"]
        if old["window_id"] != decision["window_id"]:
            raise ValueError(f"UI/window mismatch at {index}")
        if store.annotations[old["window_id"]] != old:
            raise ValueError(f"annotation changed since review snapshot at UI {index}; do not overwrite or reapply")
        updates = decision.get("updates", {})
        if set(updates) - {"joint_motion_response", "motion_evidence"}:
            raise ValueError("this review only supports response/evidence changes; preserve visibility")
        note = f"{marker} 用户原文：{decision['user_text']}"
        if decision.get("interpretation"):
            note += f"；记录说明：{decision['interpretation']}"
        new = {**old, **updates, "reviewer": "user_via_chat",
               "notes": "\n".join(x for x in (old.get("notes", ""), note) if x)}
        pending.append(new)
        decisions.append({**decision, "previous_saved_at_utc": old["saved_at_utc"],
                          "before_response": old["joint_motion_response"],
                          "after_response": new["joint_motion_response"],
                          "before_evidence": old["motion_evidence"],
                          "after_evidence": new["motion_evidence"]})
    summary = {"review_id": batch["review_id"], "applied": False,
               "confirmed_windows": len(pending),
               "response_changes": sum(d["before_response"] != d["after_response"] for d in decisions),
               "evidence_changes": sum(d["before_evidence"] != d["after_evidence"] for d in decisions),
               "visibility_changes": 0, "training_run": False, "metrics_recomputed": False,
               "frozen_experiments_modified": False, "decisions": decisions}
    if args.apply:
        # One new receipt per batch. On a partial failure inspect the receipt and
        # source tail; do not blindly reapply or restore old labels over new work.
        args.out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(store.path, args.out / "annotations_before.jsonl")
        write_json(args.out / "user_feedback.json", batch)
        # Detect a concurrent annotation write before the first append.
        if read_jsonl(store.path) != before:
            raise ValueError("annotation file changed during preparation")
        revisions = [store.save(row) for row in pending]
        after = read_jsonl(store.path)
        if after != before + revisions:
            raise ValueError("unexpected concurrent write; preserve all revisions and inspect")
        latest = {r["window_id"]: r for r in after}
        write_jsonl(args.out / "appended_revisions.jsonl", revisions)
        write_jsonl(args.out / "annotation_snapshot.jsonl", list(latest.values()))
        summary.update(applied=True, prior_revisions=len(before), total_revisions=len(after),
                       latest_windows=len(latest),
                       latest_response_counts=dict(Counter(r["joint_motion_response"] for r in latest.values())),
                       previous_revisions_preserved=True)
        write_json(args.out / "report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
