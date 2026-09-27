"""Stdout-only completed normalization audit; no torch, forward, or writes.

Reuses independent score/drift auditors, never the production summarizer.
Learned predictions/activations are not regenerated; checkpoint bytes are hashed,
not deserialized. Stored activation coordinate moments cannot be re-merged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import statistics
import time
import numpy as np
import audit_pi05_libero_action_study_step0 as a
import audit_pi05_libero_visual_dependence as d

ROOT = Path(__file__).resolve().parents[1]
RESULT = "simulation_output/pi05_libero_visual_normalization_v1"
SMOKE = "simulation_output/pi05_libero_visual_normalization_smoke_v1"
PLAN = "docs/libero-visual-normalization-plan-v1.json"
PLAN_SHA = "18b1454341b16ac2b1fd7139216fca92ba0be5d8b21245f5fcee6433298ac9e5"
VARIANTS = ("baseline", "train_visual_normalized")


def audit(root, report_path, report_sha):
    started = time.monotonic(); root = Path(root).resolve(); out = root / RESULT
    path = Path(report_path); path = path if path.is_absolute() else root / path
    a.require(path.resolve() == (out / "report.json").resolve(), "only this fixed study report is supported")
    r = a.read(a.checked(root, RESULT + "/report.json", report_sha))
    plan = a.read(a.checked(root, PLAN, PLAN_SHA))
    a.require(r["plan_sha256"] == PLAN_SHA and r["status"] == "completed_fixed_budget_normalization_comparison", "wrong completed protocol")
    status = {"status": "completed", "report_sha256": report_sha, "optimizer_steps": 2400}
    a.require(a.read(out / "status.json") == status and not (out / "failure.json").exists(), "completion/failure differs")
    a.require(r["optimizer_steps"] == r["backward_calls"] == 2400 and r["evaluation_forwards"] == 1584
              and r["feature_encodings"] == r["rollout_steps"] == 0 and r["old_baseline_replayed_bit_exact"] is True, "scope/budget differs")
    def rehash():
        for name, digest in r["input_sha256"].items(): a.checked(root, name, digest)
        for name, digest in r["output_sha256"].items(): a.checked(root, RESULT + "/" + name, digest)
        a.checked(root, RESULT + "/report.json", report_sha)
    rehash()
    a.require(all(r["input_sha256"].get(k) == v for k, v in {**plan["input_sha256"], PLAN: PLAN_SHA}.items()), "current code/plan pins missing")
    smoke = a.read(a.checked(root, SMOKE + "/report.json", r["smoke_report_sha256"]))
    norm = a.read(a.checked(root, SMOKE + "/visual_normalization.json", r["normalization_sha256"]))
    a.require(smoke["normalization_sha256"] == r["normalization_sha256"] and smoke["plan_sha256"] == PLAN_SHA, "normalization smoke binding differs")
    old = a.read(a.checked(root, d.TRAIN + "/report.json", d.TRAIN_SHA))
    step0 = a.read(a.checked(root, a.RESULT + "/report.json", a.REPORT_SHA))
    prior = a.read(a.checked(root, d.RESULT + "/report.json", d.REPORT_SHA))
    old_train = {(q["seed"], q["arm"], q["step"]): q for q in a.lines(root / d.TRAIN / "training_trace.jsonl")}
    old_initial = {(q["seed"], tuple(q["window_indices"])): q for q in a.lines(root / a.RESULT / "evaluation_trace.jsonl") if q["partition"] == "validation"}
    old_final = {(q["seed"], q["arm"], q["condition"], tuple(q["window_indices"])): q for q in a.lines(root / d.RESULT / "evaluation_trace.jsonl") if q["seed"] is not None}
    training = a.lines(out / "training_trace.jsonl"); evaluation = a.lines(out / "evaluation_trace.jsonl")
    tg, eg = defaultdict(list), defaultdict(list)
    for q in training:
        key = (q["seed"], q["arm"], q["variant"]); tg[key].append(q)
        ref = old_train[(q["seed"], q["arm"], q["step"])]; details = q["details"]
        a.require(all(q[k] == ref[k] for k in ("window_indices", "input_sha256", "target_sha256")), "shared training draws/inputs/targets differ")
        a.require(details["step"] == q["step"] and details["parameters_with_grad"] == 23 and details["parameter_changed"] is True
                  and 0 < details["global_grad_norm_after_clip"] <= 1.000001, "invalid actual update evidence")
        for k in ("loss", "visual_loss", "state_loss", "global_grad_norm_before_clip"):
            a.require(math.isfinite(details[k]) and details[k] >= 0, "nonfinite training evidence")
        a.require(q["arm"] != a.ARMS[1] or details["action_projection_weight_exact_zero_grad"] is True, "zero-action gradient differs")
        if q["variant"] == "baseline": a.require(details == ref["details"], "fresh baseline update details differ from old run")
    expected = {(s, arm, v) for s in a.SEEDS for arm in a.ARMS for v in VARIANTS}
    a.require(len(training) == 2400 and set(tg) == expected and all([q["step"] for q in rows] == list(range(1, 201)) for rows in tg.values()), "exact twelve complete update groups required")
    with a.LiberoFeaturePack(root / a.PACK) as pack:
        split = pack.load_split(root / a.PACK / "split.json"); stats = a.read(root / a.PACK / "preparation/normalization.json")
        train_ids = sorted(split["train_episode_indices"]); rows = np.concatenate([pack.indices[e] for e in train_ids])
        a.require(len(train_ids) == 8 and len(rows) == len(np.unique(rows)) == 1134 and norm["train_episode_indices"] == train_ids
                  and norm["train_row_indices"] == rows.tolist() and norm["train_record_count"] == 1134, "unique train-only fitting rows differ")
        a.require(norm["train_episodes"] == [pack.episodes[e] for e in train_ids] and norm["split_sha256"] == split["split_sha256"]
                  and norm["feature_pack_manifest_sha256"] == pack.manifest_sha256 and norm["scale_floor"] == .1, "normalization provenance differs")
        a.require(norm["source_metadata_sha256"] == pack.manifest["source"]["metadata_sha256"]
                  and norm["source_array_sha256"] == {k: v["sha256"] for k, v in pack.manifest["arrays"].items()}, "source array provenance differs")
        x = np.asarray(pack.arrays["visual_latent"][rows]); valid = np.asarray(pack.arrays["visual_valid"][rows])
        for key, value in (("train_visual_sha256", x), ("train_visual_valid_sha256", valid)):
            digest = hashlib.sha256(a.canonical({"dtype": str(value.dtype), "shape": list(value.shape)}).encode() + value.tobytes()).hexdigest()
            a.require(norm[key] == digest, "train selected array SHA differs")
        values = x.astype(np.float64); mean = values.mean(0); std = np.sqrt(np.square(values - mean).mean(0))
        a.require(valid.all() and np.array_equal(mean, norm["mean"]) and np.array_equal(std, norm["std"])
                  and np.array_equal(np.maximum(std, .1), norm["scale"]), "train-only mean/std/scale do not replay")
        dataset = a.LiberoWindowDataset(pack, split, partition="validation", normalization=stats)
        grouped = defaultdict(list)
        for i, w in enumerate(dataset.windows): grouped[w.episode_index].append(i)
        a.require(len(dataset) == 500 and set(grouped) == {1458, 1476, 1530, 1566}, "fixed validation support differs")
        a.require(r["anchor"] == prior["anchor"], "saved anchor provenance differs from prior diagnostic")
        anchor = np.asarray(pack.arrays["visual_latent"][447]); batches = {}
        a.require(int(pack.arrays["episode_index"][447]) == 1312 and int(pack.arrays["frame_index"][447]) == 0 and 1312 == min(train_ids), "fixed anchor differs")
        for eid, ids in grouped.items():
            for start in range(0, len(ids), 16):
                selected = ids[start:start+16]; items = [dataset[i] for i in selected]
                hist = np.stack([v["inputs"]["history_visual_latent"] for v in items]); hists = {"clean": hist, "repeat_current": np.broadcast_to(hist[:, -1:], hist.shape), "fixed_train_anchor": np.broadcast_to(anchor, hist.shape)}
                zero = a.array_hash(np.zeros_like(np.stack([v["inputs"]["candidate_actions"] for v in items])))
                batches[tuple(selected)] = eid, a.batch_fingerprints(items, "inputs"), a.batch_fingerprints(items, "targets"), zero, hists
        for q in evaluation:
            key = (q["seed"], q["arm"], q["variant"], q["phase"], q["condition"]); eg[key].append(q)
            eid, inputs, targets, zero, hists = batches[tuple(q["window_indices"])]; condition = q["condition"]
            want = {**inputs, "history_visual_latent": a.array_hash(hists[condition])}
            if q["arm"] == a.ARMS[1]: want["candidate_actions"] = zero
            a.require(q["episode_index"] == eid and q["input_sha256"] == want and q["target_sha256"] == targets, "condition input/clean target differs")
            a.check_summary(q["score"], len(q["window_indices"]), 1, stats); d.drift_check(q["drift"], len(q["window_indices"]), 1)
            for name, arrays in (("history_input_delta", hists), ("anchor_skip_delta", {c: np.broadcast_to(h[:, -1, None, None], (len(h), 1, 3, 2, 2048)) for c, h in hists.items()})):
                d.raw_drift(q["drift"][name], arrays["clean"], arrays[condition])
            if condition == "clean": a.require(all(q["drift"][k]["exact_changed_count"] == 0 for k in d.WIDTHS), "clean drift is nonzero")
            if q["variant"] == "baseline":
                ref = old_initial[(q["seed"], tuple(q["window_indices"]))]["predictions_sha256"][q["arm"]] if q["phase"] == "step0" else old_final[(q["seed"], q["arm"], condition, tuple(q["window_indices"]))]["prediction_sha256"]
                a.require(q["prediction_sha256"] == ref, "old baseline prediction bytes differ")
        expected_eval = {(s, arm, v, phase, c) for s, arm, v in expected for phase in ("step0", "final200") for c in (("clean",) if phase == "step0" else d.CONDITIONS)}
        a.require(len(evaluation) == 1584 and set(eg) == expected_eval, "complete evaluation groups required")
        for (s, arm, v, phase, c), records in eg.items():
            a.require(len(records) == 33 and Counter(i for q in records for i in q["window_indices"]) == Counter(range(500)), "evaluation coverage duplicated/missing")
            run = r["runs"][str(s)][arm][v]["initial" if phase == "step0" else "final"]["conditions"][c]
            reference = None if v != "baseline" else (step0["runs"][str(s)]["validation"]["arms"][arm] if phase == "step0" else prior["runs"][str(s)][arm][c])
            d.check_scoring_group(run, records, grouped, stats, reference)
    for s, arm, v in expected:
        run = r["runs"][str(s)][arm][v]; ck = run["checkpoint"]; b = ck["binding"]
        a.checked(root, RESULT + "/" + ck["path"], ck["sha256"])
        a.require(b["seed"] == s and b["arm"] == arm and b["variant"] == v and b["step"] == 200
                  and b["plan_sha256"] == PLAN_SHA and b["normalization_sha256"] == r["normalization_sha256"]
                  and b["smoke_report_sha256"] == r["smoke_report_sha256"] and b["draw_sha256"] == step0["sampling"][str(s)]["sha256"]
                  and b["normalization_applied"] is (v != "baseline") and ck["optimizer_state_saved"] is False and ck["resumable"] is False, "checkpoint binding differs")
        a.require(b["initial_named_parameter_sha256"] == smoke["results"][str(s)][arm][v]["initial_named_parameter_sha256"]
                  == smoke["results"][str(s)][arm]["baseline"]["initial_named_parameter_sha256"], "paired/smoke initialization binding differs")
        for key in ("initial_named_parameter_sha256", "final_named_parameter_sha256", "final_state_sha256"): a.digest(b[key])
        if v == "baseline": a.require(b["final_state_sha256"] == b["final_named_parameter_sha256"] == prior["checkpoint_checks"][str(s)][arm]["parameter_sha256_before"], "old final baseline state hash differs")
        a.require(run["training"]["optimizer_steps"] == 200 and run["training"]["first_step"] == tg[(s, arm, v)][0]["details"] and run["training"]["last_step"] == tg[(s, arm, v)][-1]["details"], "saved update endpoints differ")
        for phase in ("initial", "final"):
            act = run[phase]["activation"]
            a.require(set(act["per_episode"]) == set(map(str, grouped)), "activation episode membership differs")
            for eid, per_view in act["per_episode"].items():
                for val in per_view.values():
                    t, p = val["tanh"], val["post"]; n = len(grouped[int(eid)]) * 4 * 128
                    a.require(t["count"] == p["count"] == n and t["windows"] == p["windows"] == len(grouped[int(eid)]), "view activation support differs")
                    a.close_number(t["mean_slope"], t["sum_slope"] / n, "tanh slope mean")
                    a.close_number(t["post_abs_exact_1_fraction"], t["post_abs_exact_1_count"] / n, "exact-one fraction")
                    for group in ("post_abs_ge", "local_slope_le"):
                        for val in t[group].values():
                            a.require(0 <= val["count"] <= n, "tanh threshold count differs")
                            a.close_number(val["fraction"], val["count"] / n, "threshold fraction")
                    a.close_number(p["rms"] ** 2, p["coordinate_mean_rms"] ** 2 + p["coordinate_std_rms"] ** 2, "activation coordinate energy identity")
            for view in ("0", "1"):
                values = list(act["per_episode"].values())
                fields = {"saturated_fraction": lambda x: x[view]["tanh"]["post_abs_ge"]["0.99"]["fraction"], "exact_one_fraction": lambda x: x[view]["tanh"]["post_abs_exact_1_fraction"], "mean_slope": lambda x: x[view]["tanh"]["mean_slope"], "coordinate_std_rms": lambda x: x[view]["post"]["coordinate_std_rms"]}
                for k, f in fields.items(): a.close_number(act["episode_macro"][view][k], statistics.mean(f(x) for x in values), "activation macro")
    summary = r["summary"]
    for arm in a.ARMS:
        means = summary["clean_seed_means"][arm]
        for v in VARIANTS:
            for m, value in means[v].items(): a.close_number(value, statistics.mean(r["runs"][str(s)][arm][v]["final"]["conditions"]["clean"]["episode_macro"][m] for s in a.SEEDS), "seed mean")
        a.check_changes(summary["clean_changes_of_seed_means"][arm], means[VARIANTS[1]], means[VARIANTS[0]])
        for s in a.SEEDS: a.check_changes(summary["paired_seed_changes"][arm][str(s)], r["runs"][str(s)][arm][VARIANTS[1]]["final"]["conditions"]["clean"]["episode_macro"], r["runs"][str(s)][arm][VARIANTS[0]]["final"]["conditions"]["clean"]["episode_macro"])
        for m, value in summary["paired_delta_population_std"][arm].items(): a.close_number(value, statistics.pstdev(summary["paired_seed_changes"][arm][str(s)][m]["absolute"] for s in a.SEEDS), "paired population std")
    arm = a.ARMS[0]
    gain = all(summary["paired_seed_changes"][arm][str(s)]["masked_objective"]["absolute"] < 0 for s in a.SEEDS) and all(summary["clean_changes_of_seed_means"][arm][m]["absolute"] <= 0 for m in ("normalized_state_mae", "visual_mae"))
    desat = all(r["runs"][str(s)][arm][VARIANTS[1]]["final"]["activation"]["episode_macro"][v]["saturated_fraction"] < r["runs"][str(s)][arm][VARIANTS[0]]["final"]["activation"]["episode_macro"][v]["saturated_fraction"] for s in a.SEEDS for v in ("0", "1"))
    a.require(summary["primary_fixed_validation_prediction_gain"] is gain and summary["primary_desaturation_all_seed_view"] is desat and summary["decision"] == ("descriptive_fixed_validation_prediction_gain" if gain else "mixed_or_no_stable_overall_prediction_gain"), "predeclared decision differs")
    rehash()
    return dict(status="passed_independent_normalization_audit", report_sha256=report_sha, audit_code_sha256=a.sha256_file(Path(__file__)), input_pins=len(r["input_sha256"]), output_pins=len(r["output_sha256"]), optimizer_trace_rows=2400, optimizer_groups=12, evaluation_rows=1584, scoring_groups=len(eg), baseline_update_details_replayed=1200, baseline_prediction_hashes_replayed=792, raw_train_rows=1134, checkpoint_files_hashed=12, checkpoint_deserialization=False, model_forwards=0, backward_calls=0, optimizer_updates=0, wall_seconds=time.monotonic()-started, limits="Saved learned predictions/drift/activation statistics checked, not regenerated; coordinate activation vectors absent. Checkpoint payloads not deserialized; bindings checked against reports and file hashes. No significance or policy benefit established.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=RESULT + "/report.json")
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    print(a.canonical(audit(ROOT, args.report, args.report_sha256)))
