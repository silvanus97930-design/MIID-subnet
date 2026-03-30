#!/usr/bin/env python3
"""Local SN54 Phase 4 benchmark harness with validator-style fixtures.

Outputs:
- results/<timestamp>/summary.json
- results/<timestamp>/summary.csv
- results/<timestamp>/report.md
- results/<timestamp>/<fixture_id>/variation_*.json
- results/<timestamp>/<fixture_id>/variation_*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from MIID.miner.image_generator import generate_variations
from MIID.miner.request_spec import compile_phase4_variation_requests


@dataclass
class VariationRun:
    fixture_id: str
    variation_type: str
    intensity: str
    success: bool
    failure_reason: str
    latency_ms: float
    adaface_score: Optional[float]
    arcface_score: Optional[float]
    adherence_score: Optional[float]
    final_score: Optional[float]
    selected_candidate_index: Optional[int]
    candidate_count: Optional[int]
    image_hash: str
    output_image: str
    output_json: str


def _load_fixture(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "fixture_id" not in data or "variation_requests" not in data:
        raise ValueError(f"invalid fixture format: {path}")
    return data


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _selected_identity_scores(var: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    idx = int(var.get("selected_candidate_index", 0) or 0)
    identity = var.get("identity_scores") or []
    if not isinstance(identity, list) or not (0 <= idx < len(identity)):
        return None, None
    row = identity[idx] or {}
    ada = _safe_float(((row.get("adaface") or {}).get("similarity")))
    arc = _safe_float(((row.get("insightface_arcface") or {}).get("similarity")))
    return ada, arc


def _selected_adherence_and_final(var: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    idx = int(var.get("selected_candidate_index", 0) or 0)
    rerank = var.get("candidate_rerank_scores") or []
    if isinstance(rerank, list) and 0 <= idx < len(rerank):
        row = rerank[idx] or {}
        return _safe_float(row.get("adherence_score")), _safe_float(row.get("final_score"))
    finals = var.get("ensemble_final_scores") or []
    if isinstance(finals, list) and 0 <= idx < len(finals):
        return None, _safe_float(finals[idx])
    return None, None


def _iter_fixtures(fixtures_dir: Path) -> Sequence[Path]:
    return sorted(fixtures_dir.glob("*.json"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)


def _summarize(runs: List[VariationRun]) -> Dict[str, Any]:
    failures: Dict[str, int] = {}
    by_type: Dict[str, List[float]] = {}

    for r in runs:
        if not r.success:
            failures[r.failure_reason] = failures.get(r.failure_reason, 0) + 1
        if r.final_score is not None:
            by_type.setdefault(r.variation_type, []).append(r.final_score)

    ranked = [
        {
            "variation_type": k,
            "mean_final_score": round(statistics.mean(v), 6),
            "count": len(v),
        }
        for k, v in by_type.items()
    ]
    ranked.sort(key=lambda x: x["mean_final_score"], reverse=True)

    return {
        "total_runs": len(runs),
        "success_count": sum(1 for r in runs if r.success),
        "failure_count": sum(1 for r in runs if not r.success),
        "best_request_types": ranked[:3],
        "worst_request_types": list(reversed(ranked[-3:])),
        "failure_reasons": dict(sorted(failures.items())),
    }


def _write_report(path: Path, summary: Dict[str, Any], runs: List[VariationRun]) -> None:
    lines: List[str] = []
    lines.append("# SN54 Phase 4 Benchmark Report")
    lines.append("")
    lines.append(f"- Total runs: {summary['total_runs']}")
    lines.append(f"- Success: {summary['success_count']}")
    lines.append(f"- Failures: {summary['failure_count']}")
    lines.append("")
    lines.append("## Best Request Types")
    for row in summary["best_request_types"]:
        lines.append(
            f"- `{row['variation_type']}` mean_final_score={row['mean_final_score']} (n={row['count']})"
        )
    lines.append("")
    lines.append("## Worst Request Types")
    for row in summary["worst_request_types"]:
        lines.append(
            f"- `{row['variation_type']}` mean_final_score={row['mean_final_score']} (n={row['count']})"
        )
    lines.append("")
    lines.append("## Failure Reasons")
    if not summary["failure_reasons"]:
        lines.append("- none")
    else:
        for k, v in summary["failure_reasons"].items():
            lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Per-Variation Results")
    lines.append("")
    lines.append(
        "| fixture_id | variation_type | intensity | success | latency_ms | adaface | arcface | adherence | final_score |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        lines.append(
            f"| {r.fixture_id} | {r.variation_type} | {r.intensity} | {r.success} | {r.latency_ms:.2f} | "
            f"{'' if r.adaface_score is None else round(r.adaface_score, 6)} | "
            f"{'' if r.arcface_score is None else round(r.arcface_score, 6)} | "
            f"{'' if r.adherence_score is None else round(r.adherence_score, 6)} | "
            f"{'' if r.final_score is None else round(r.final_score, 6)} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(fixtures_dir: Path, seed_image_path: Path, results_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(seed_image_path) as im:
        base_image = im.convert("RGB")

    runs: List[VariationRun] = []
    fixture_paths = _iter_fixtures(fixtures_dir)
    if not fixture_paths:
        raise RuntimeError(f"no fixtures found in {fixtures_dir}")

    for fixture_path in fixture_paths:
        fixture = _load_fixture(fixture_path)
        fixture_id = str(fixture["fixture_id"])
        fixture_out = out_dir / fixture_id
        fixture_out.mkdir(parents=True, exist_ok=True)

        compiled_ir, errs = compile_phase4_variation_requests(fixture["variation_requests"])
        if compiled_ir is None:
            run = VariationRun(
                fixture_id=fixture_id,
                variation_type="compile",
                intensity="n/a",
                success=False,
                failure_reason="request_spec_invalid",
                latency_ms=0.0,
                adaface_score=None,
                arcface_score=None,
                adherence_score=None,
                final_score=None,
                selected_candidate_index=None,
                candidate_count=None,
                image_hash="",
                output_image="",
                output_json="",
            )
            runs.append(run)
            _write_json(fixture_out / "compile_errors.json", {"errors": errs})
            continue

        for i, compiled in enumerate(compiled_ir.variations, start=1):
            request = compiled.as_protocol_request()
            timings: Dict[str, float] = {}
            t0 = time.perf_counter()
            try:
                generated = generate_variations(
                    base_image,
                    [request],
                    pipeline_timings_out=timings,
                    obs_context={"benchmark_fixture": fixture_id, "benchmark_request_index": i},
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                runs.append(
                    VariationRun(
                        fixture_id=fixture_id,
                        variation_type=compiled.variation_type.value,
                        intensity=compiled.intensity.value,
                        success=False,
                        failure_reason=f"generate_exception:{type(e).__name__}",
                        latency_ms=latency_ms,
                        adaface_score=None,
                        arcface_score=None,
                        adherence_score=None,
                        final_score=None,
                        selected_candidate_index=None,
                        candidate_count=None,
                        image_hash="",
                        output_image="",
                        output_json="",
                    )
                )
                continue

            if not generated:
                runs.append(
                    VariationRun(
                        fixture_id=fixture_id,
                        variation_type=compiled.variation_type.value,
                        intensity=compiled.intensity.value,
                        success=False,
                        failure_reason="generation_empty",
                        latency_ms=latency_ms,
                        adaface_score=None,
                        arcface_score=None,
                        adherence_score=None,
                        final_score=None,
                        selected_candidate_index=None,
                        candidate_count=None,
                        image_hash="",
                        output_image="",
                        output_json="",
                    )
                )
                continue

            var = generated[0]
            image = var.get("image")
            img_path = fixture_out / f"variation_{i:02d}_{compiled.label().replace('(', '_').replace(')', '')}.png"
            if isinstance(image, Image.Image):
                image.save(img_path, format="PNG")

            adaface, arcface = _selected_identity_scores(var)
            adherence, final_score = _selected_adherence_and_final(var)
            selected_idx = int(var.get("selected_candidate_index", 0) or 0)
            count = int(var.get("candidate_count", 0) or 0)
            out_json = fixture_out / f"variation_{i:02d}_result.json"
            payload = {
                "fixture_id": fixture_id,
                "variation_index": i,
                "compiled_label": compiled.label(),
                "timings_ms": timings,
                "latency_ms": latency_ms,
                "selected_candidate_index": selected_idx,
                "candidate_count": count,
                "image_hash": var.get("image_hash", ""),
                "identity_scores_selected": var.get("identity_score_selected"),
                "candidate_rerank_scores": var.get("candidate_rerank_scores"),
                "ensemble_leaderboard": var.get("ensemble_leaderboard"),
                "screen_replay_device": var.get("screen_replay_device"),
                "visual_cue_keys": var.get("visual_cue_keys"),
            }
            _write_json(out_json, payload)

            runs.append(
                VariationRun(
                    fixture_id=fixture_id,
                    variation_type=compiled.variation_type.value,
                    intensity=compiled.intensity.value,
                    success=True,
                    failure_reason="",
                    latency_ms=latency_ms,
                    adaface_score=adaface,
                    arcface_score=arcface,
                    adherence_score=adherence,
                    final_score=final_score,
                    selected_candidate_index=selected_idx,
                    candidate_count=count,
                    image_hash=str(var.get("image_hash", "")),
                    output_image=str(img_path.relative_to(out_dir)) if img_path.exists() else "",
                    output_json=str(out_json.relative_to(out_dir)),
                )
            )

    summary = _summarize(runs)
    summary_path = out_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "generated_at": stamp,
            "seed_image": str(seed_image_path),
            "summary": summary,
            "runs": [r.__dict__ for r in runs],
        },
    )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(VariationRun.__annotations__.keys()))
        writer.writeheader()
        for r in runs:
            writer.writerow(r.__dict__)

    report_path = out_dir / "report.md"
    _write_report(report_path, summary, runs)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local SN54 Phase 4 benchmark fixtures.")
    parser.add_argument(
        "--seed-image",
        required=True,
        help="Path to local seed image (RGB face photo).",
    )
    parser.add_argument(
        "--fixtures-dir",
        default=str(Path(__file__).resolve().parent / "fixtures"),
        help="Directory containing fixture JSON files.",
    )
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Parent directory for timestamped results.",
    )
    args = parser.parse_args()

    out_dir = run_benchmark(
        fixtures_dir=Path(args.fixtures_dir),
        seed_image_path=Path(args.seed_image),
        results_root=Path(args.results_root),
    )
    print(f"SN54 benchmark completed: {out_dir}")
    print(f"- summary: {out_dir / 'summary.json'}")
    print(f"- csv: {out_dir / 'summary.csv'}")
    print(f"- report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
