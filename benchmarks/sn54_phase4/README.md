# SN54 Phase 4 Local Benchmarks

Validator-style regression harness for local image-generation quality checks.

## What This Runs

- Loads fixture JSON files from `benchmarks/sn54_phase4/fixtures/`.
- Compiles requests using `MIID.miner.request_spec.compile_phase4_variation_requests`.
- Runs miner generation via `MIID.miner.image_generator.generate_variations`.
- Records latency, selected ArcFace/AdaFace scores, adherence score, final score, and selected candidate metadata.
- Writes machine-readable artifacts (`summary.json`, `summary.csv`) and a human-readable report (`report.md`).

## Exact Command (all fixtures)

```bash
python benchmarks/sn54_phase4/run_benchmarks.py \
  --seed-image "/absolute/path/to/seed_face.png"
```

Optional overrides:

```bash
python benchmarks/sn54_phase4/run_benchmarks.py \
  --seed-image "/absolute/path/to/seed_face.png" \
  --fixtures-dir "benchmarks/sn54_phase4/fixtures" \
  --results-root "benchmarks/sn54_phase4/results"
```

## Output Layout

Each run creates:

`benchmarks/sn54_phase4/results/<YYYYmmdd_HHMMSS>/`

- `summary.json`
- `summary.csv`
- `report.md`
- `<fixture_id>/variation_*.json`
- `<fixture_id>/variation_*.png`

## Fixture Format

Each fixture must include:

- `fixture_id`
- `description`
- `variation_requests` (validator wire format entries with `type`, `intensity`, `description`, `detail`)

See the five shipped fixtures under `benchmarks/sn54_phase4/fixtures/`.
