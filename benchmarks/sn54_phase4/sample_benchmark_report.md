# SN54 Phase 4 Benchmark Report (Sample)

- Total runs: 10
- Success: 9
- Failures: 1

## Best Request Types

- `screen_replay` mean_final_score=0.812341 (n=5)
- `expression_edit` mean_final_score=0.768551 (n=2)
- `background_edit` mean_final_score=0.744123 (n=1)

## Worst Request Types

- `pose_edit` mean_final_score=0.662913 (n=2)
- `background_edit` mean_final_score=0.744123 (n=1)
- `expression_edit` mean_final_score=0.768551 (n=2)

## Failure Reasons

- `generation_empty`: 1

## Per-Variation Results

| fixture_id | variation_type | intensity | success | latency_ms | adaface | arcface | adherence | final_score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixture_01_pose_medium_screen_laptop_keystone_moire | pose_edit | medium | True | 1470.34 | 0.831412 | 0.817551 | 0.702811 | 0.783445 |
| fixture_01_pose_medium_screen_laptop_keystone_moire | screen_replay | standard | True | 932.15 | 0.781005 | 0.759220 | 0.824761 | 0.804941 |
| fixture_04_pose_far_screen_tablet_edge_gamma | pose_edit | far | False | 2191.88 |  |  |  |  |

> This file is a format example. Real reports are generated under
> `benchmarks/sn54_phase4/results/<timestamp>/report.md`.
