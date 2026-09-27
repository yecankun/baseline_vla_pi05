# Real Collection Onsite Audit - 2026-07-21

## Scope

This audit classifies the July 21 S-bend linked captures. It does not promote
partial or failed runs to expert training data. The action interface remains:

```text
elite_tcp_delta_6d + piper_intent_id
```

For the replacement feeder, `piper_intent_id=feed` means that a forward command
was issued. It is not proof that the guidewire physically advanced.

## Episode Classification

| Dataset suffix | Route | Records | Classification | Permitted use |
| --- | --- | ---: | --- | --- |
| `192749_left_s_bend_rgbd` | left | 350 | Full physical success, but 148 stale Elite poses | Visual/physical reference; not clean action training |
| `193016_left_s_bend_rgbd` | left | 297 | Failed, shared Elite connection | Failure analysis only |
| `193637_left_s_bend_rgbd` | left | 239 | Failed, shared Elite connection | Failure analysis only |
| `195033_left_s_bend_rgbd` | left | 54 | Clean pose sync; feeder commands had no observed motion | Command/nonresponse diagnostic |
| `200628_left_s_bend_rgbd` | left | 310 | Full physical success with clean pose sync | Clean-success candidate pending visual/action audit |
| `200813_left_s_bend_rgbd` | left | 171 | Physical failure with clean pose sync | Failure analysis only |
| `201018_left_s_bend_rgbd` | left | 64 | Physical failure with clean pose sync | Failure analysis only |
| `202113_right_s_bend_rgbd` | right | 189 | Incomplete right attempt with clean pose sync | Failure/route-prefix analysis |
| `202357_right_s_bend_rgbd` | right | 153 | Incomplete right attempt with clean pose sync | Failure/route-prefix analysis |
| `203449_right_s_bend_rgbd` | right | 183 | Passed the bifurcation, then feeder motion stopped | Primary right-route partial reference |

## Clean Left Candidate

`real_pilot_20260721_200628_left_s_bend_rgbd` contains:

```text
records: 310
operator-confirmed feeder events: 8
side/top RGB: 310 + 310 at 1920x1080
aligned uint16 depth: 310 + 310 at 1920x1080
elite_pose_stale_count: 0
side/top timestamp P95: 74.03 ms
image/pose timestamp P95: 81.23 ms
image/action timestamp P95: 87.92 ms
validator errors: 0
```

It is the only current full physical success with the corrected separate Elite
command connection. It still needs representative-frame or video review before
being exported as clean expert data.

## Right Partial Reference

`real_pilot_20260721_203449_right_s_bend_rgbd` contains:

```text
records: 183
logged feeder intents: 11
final Elite path index: 16 / 19
side/top RGB: 183 + 183 at 1920x1080
aligned uint16 depth: 183 + 183 at 1920x1080
elite_pose_stale_count: 0
side/top timestamp P95: 76.73 ms
image/pose timestamp P95: 81.84 ms
image/action timestamp P95: 89.60 ms
validator errors / warnings: 0 / 3
```

The operator confirmed that the guidewire passed the right bifurcation before
physical feeder motion stopped. The last several `feed` commands were logged as
sent but produced no observed motion. Therefore:

- the episode is not a complete right expert demonstration;
- its pre-stall prefix is valuable real right-route evidence;
- its suffix is an actuator-nonresponse/failure segment;
- a manual or image-progress sidecar must identify the usable cutoff before any
  training export.

## Hardware Interpretation

The onsite session lasted about four hours. The feeder fixture used hot-melt
adhesive; after heating, the fixture loosened and could no longer clamp the
guidewire consistently. This is an operator-observed mechanical failure mode,
not a UDP protocol failure. A software `sent` result must remain command
provenance only.

## Decision And Next Work

- End dynamic collection for this onsite session.
- Do not collect more real expert episodes until the feeder clamp is repaired
  with a mechanically and thermally stable fixture.
- Preserve `200628` as the clean-left candidate.
- Preserve `203449` as the main right partial reference and annotate its
  physical-progress cutoff offline.
- Keep other partial/failed runs separate from expert training data.
- Use the successful/partial captures for simulation camera, route, feeder,
  and failure-mode alignment; do not claim real-policy validation.
