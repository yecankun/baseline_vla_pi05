# Real Pilot Failure Audit

This report summarizes local integrity and timing checks for real pilot datasets.
It is a review aid, not an automatic physical-cause detector.

## real_pilot_20260705_left_mode2_linked_elite_piper_001

- records: 111
- tasks: `{'left': 111}`
- robot command mode: `piper_control_enabled`
- piper final step: `18`
- piper commands: `{'0': 93, '1': 18}`
- piper executed: `{'feed_async_started_by_elite_path': 18, 'hold': 93}`
- elite path index max/last: `19` / `19`
- sync image-pose p95/max ms: `2090.6381607055664` / `5108.225107192993`
- sync image-action p95/max ms: `2094.989538192749` / `5112.502813339233`
- synchronization judgment: `not train/eval suitable`
- Elite pose step p95/max mm: `17.412643691677605` / `103.66034600011452`
- side image adjacent diff median/p95: `1.3904166666666666` / `12.364010416666666`
- top image adjacent diff median/p95: `0.8234375` / `7.759635416666667`
- contact sheet: `docs\_real_pilot_20260705_failure_audit\real_pilot_20260705_left_mode2_linked_elite_piper_001_contact_sheet.jpg`

## real_pilot_20260705_right_mode2_linked_elite_piper_fast_001

- records: 131
- tasks: `{'right': 131}`
- robot command mode: `piper_control_enabled`
- piper final step: `12`
- piper commands: `{'0': 119, '1': 12}`
- piper executed: `{'feed_async_started_by_elite_path': 12, 'hold': 119}`
- elite path index max/last: `19` / `19`
- sync image-pose p95/max ms: `1653.3613204956055` / `5454.792022705078`
- sync image-action p95/max ms: `1658.233880996704` / `5460.387468338013`
- synchronization judgment: `not train/eval suitable`
- Elite pose step p95/max mm: `14.363031315825488` / `113.74365701749853`
- side image adjacent diff median/p95: `1.0309375` / `12.391822916666667`
- top image adjacent diff median/p95: `0.83921875` / `7.306041666666666`
- contact sheet: `docs\_real_pilot_20260705_failure_audit\real_pilot_20260705_right_mode2_linked_elite_piper_fast_001_contact_sheet.jpg`

## Current Conclusion

The current onsite pilot runs should be treated as failed/stuck pilot data, not successful demonstrations.
They are still useful for validating the real collection schema, camera setup, command logs,
and for locating the physical guidewire blockage before the next lab visit. They should not be used
as successful policy-training or strict shadow-evaluation data because the old collection run shows
large image-pose/action timestamp deltas.

Next local use:

1. Review the generated contact sheets around feed events.
2. Compare the stuck frames with the physical vessel groove/entry geometry.
3. Keep these datasets out of successful policy-training manifests.
4. Re-run this audit immediately after the next onsite collection.
