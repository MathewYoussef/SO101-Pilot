#!/usr/bin/env python3
"""Estimate Transfer run cost/time from a pilot throughput measurement."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BudgetEstimate:
    """Transfer budget estimate summary."""

    episodes: int
    total_input_seconds: float
    variants_per_demo: int
    total_output_seconds: float
    pilot_output_seconds_per_gpu_second: float
    estimated_gpu_hours: float
    estimated_wall_hours: float
    estimated_energy_kwh: float
    max_variants_by_gpu_hours: int | None
    max_variants_by_wall_hours: int | None
    max_variants_by_energy_kwh: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=Path("data/manifests/episode_video_manifest.jsonl"),
        help="Episode manifest JSONL produced by scripts/00a_validate_transfer.py.",
    )
    parser.add_argument("--variants-per-demo", type=int, default=3, help="Planned number of synthetic variants per demo.")
    parser.add_argument(
        "--pilot-output-seconds",
        type=float,
        required=True,
        help="Total output video seconds generated during pilot run.",
    )
    parser.add_argument(
        "--pilot-wall-seconds",
        type=float,
        required=True,
        help="Wall-clock seconds taken by pilot run.",
    )
    parser.add_argument(
        "--pilot-num-gpus",
        type=int,
        default=1,
        help="Number of GPUs used during pilot run.",
    )
    parser.add_argument(
        "--target-num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for full run (for wall-time estimate).",
    )
    parser.add_argument(
        "--avg-gpu-power-watts",
        type=float,
        default=350.0,
        help="Average power draw per GPU in watts for energy estimate.",
    )
    parser.add_argument(
        "--gpu-hours-budget",
        type=float,
        default=None,
        help="Optional budget cap in aggregate GPU-hours.",
    )
    parser.add_argument(
        "--wall-hours-budget",
        type=float,
        default=None,
        help="Optional budget cap in wall-clock hours at --target-num-gpus.",
    )
    parser.add_argument(
        "--energy-kwh-budget",
        type=float,
        default=None,
        help="Optional budget cap in kWh.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/manifests/transfer_budget_estimate.json"),
        help="Output JSON summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variants_per_demo < 1:
        raise SystemExit("--variants-per-demo must be >= 1")
    if args.pilot_output_seconds <= 0 or args.pilot_wall_seconds <= 0:
        raise SystemExit("--pilot-output-seconds and --pilot-wall-seconds must be > 0")
    if args.pilot_num_gpus < 1 or args.target_num_gpus < 1:
        raise SystemExit("--pilot-num-gpus and --target-num-gpus must be >= 1")

    manifest_records = [json.loads(line) for line in args.manifest_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_input_seconds = 0.0
    for record in manifest_records:
        if "from_timestamp" in record and "to_timestamp" in record:
            total_input_seconds += float(record["to_timestamp"]) - float(record["from_timestamp"])
        elif "num_steps" in record and "fps" in record:
            total_input_seconds += float(record["num_steps"]) / float(record["fps"])
        else:
            raise SystemExit("Manifest rows must include either from/to timestamps or num_steps+fps.")

    throughput = args.pilot_output_seconds / (args.pilot_wall_seconds * args.pilot_num_gpus)
    total_output_seconds = total_input_seconds * args.variants_per_demo
    estimated_gpu_seconds = total_output_seconds / throughput
    estimated_gpu_hours = estimated_gpu_seconds / 3600.0
    estimated_wall_hours = estimated_gpu_hours / args.target_num_gpus
    estimated_energy_kwh = estimated_gpu_hours * (args.avg_gpu_power_watts / 1000.0)

    max_variants_by_gpu_hours = _max_variants(args.gpu_hours_budget, total_input_seconds, throughput)
    max_variants_by_wall_hours = _max_variants(
        args.wall_hours_budget * args.target_num_gpus if args.wall_hours_budget else None,
        total_input_seconds,
        throughput,
    )
    max_variants_by_energy_kwh = _max_variants(
        args.energy_kwh_budget / (args.avg_gpu_power_watts / 1000.0) if args.energy_kwh_budget else None,
        total_input_seconds,
        throughput,
    )

    estimate = BudgetEstimate(
        episodes=len(manifest_records),
        total_input_seconds=total_input_seconds,
        variants_per_demo=args.variants_per_demo,
        total_output_seconds=total_output_seconds,
        pilot_output_seconds_per_gpu_second=throughput,
        estimated_gpu_hours=estimated_gpu_hours,
        estimated_wall_hours=estimated_wall_hours,
        estimated_energy_kwh=estimated_energy_kwh,
        max_variants_by_gpu_hours=max_variants_by_gpu_hours,
        max_variants_by_wall_hours=max_variants_by_wall_hours,
        max_variants_by_energy_kwh=max_variants_by_energy_kwh,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(estimate), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote estimate: {args.output_json}")
    print(f"Episodes: {estimate.episodes}")
    print(f"Total input seconds: {estimate.total_input_seconds:.2f}")
    print(f"Planned variants/demo: {estimate.variants_per_demo}")
    print(f"Estimated GPU-hours: {estimate.estimated_gpu_hours:.2f}")
    print(f"Estimated wall-hours (@{args.target_num_gpus} GPU): {estimate.estimated_wall_hours:.2f}")
    print(f"Estimated energy (kWh): {estimate.estimated_energy_kwh:.2f}")


def _max_variants(gpu_hours_budget: float | None, total_input_seconds: float, throughput: float) -> int | None:
    """Return max integer variants per demo for a given GPU-hours budget."""
    if gpu_hours_budget is None:
        return None
    max_output_seconds = gpu_hours_budget * 3600.0 * throughput
    variants = math.floor(max_output_seconds / max(total_input_seconds, 1e-9))
    return max(0, int(variants))


if __name__ == "__main__":
    main()
