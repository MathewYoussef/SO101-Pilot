#!/usr/bin/env python3
"""Build deterministic Reason labeling requests (all original + subset augmented)."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-manifest-jsonl",
        type=Path,
        default=Path("data/manifests/episode_video_manifest.jsonl"),
        help="Original episode manifest from scripts/00a_validate_transfer.py.",
    )
    parser.add_argument(
        "--transfer-jobs-jsonl",
        type=Path,
        default=None,
        help="Transfer jobs manifest from scripts/03_run_transfer.py.",
    )
    parser.add_argument(
        "--no-include-original",
        action="store_true",
        help="If set, skip original episodes and sample only augmented entries.",
    )
    parser.add_argument(
        "--augmented-sample-ratio",
        type=float,
        default=0.25,
        help="Stratified sampling ratio for augmented entries.",
    )
    parser.add_argument(
        "--augmented-sample-max",
        type=int,
        default=None,
        help="Optional global cap on sampled augmented entries.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--keyframes",
        type=str,
        default="start,mid,end",
        help="Comma-separated keyframe anchors to request from Reason.",
    )
    parser.add_argument(
        "--output-requests-jsonl",
        type=Path,
        default=Path("data/manifests/reason_requests.jsonl"),
        help="Output request manifest for Reason machine.",
    )
    parser.add_argument(
        "--output-report-json",
        type=Path,
        default=Path("data/manifests/reason_sampling_report.json"),
        help="Output sampling report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    keyframes = [token.strip() for token in args.keyframes.split(",") if token.strip()]
    if not keyframes:
        raise SystemExit("--keyframes must contain at least one value.")
    if args.augmented_sample_ratio < 0:
        raise SystemExit("--augmented-sample-ratio must be >= 0.")

    original_records = _read_jsonl(args.original_manifest_jsonl)
    if not original_records:
        raise SystemExit(f"No rows in {args.original_manifest_jsonl}")

    # Build lookup for carrying over task text into augmented requests.
    task_by_demo_id: dict[int, str] = {}
    for row in original_records:
        demo_id = int(row["demo_id"])
        task_by_demo_id[demo_id] = str(row.get("task_text", ""))

    requests: list[dict[str, object]] = []
    original_count = 0
    include_original = not args.no_include_original
    if include_original:
        for row in original_records:
            demo_id = int(row["demo_id"])
            video_path = _resolve_original_video_path(row)
            requests.append(
                {
                    "request_id": f"orig_{demo_id:04d}",
                    "source_type": "original",
                    "demo_id": demo_id,
                    "variant_idx": None,
                    "task_text": str(row.get("task_text", "")),
                    "video_path": video_path,
                    "keyframes": keyframes,
                }
            )
        original_count = len(original_records)

    augmented_total = 0
    augmented_selected = 0
    if args.transfer_jobs_jsonl is not None and args.transfer_jobs_jsonl.exists():
        transfer_jobs = _read_jsonl(args.transfer_jobs_jsonl)
        augmented_total = len(transfer_jobs)
        sampled_jobs = _sample_augmented_jobs(
            jobs=transfer_jobs,
            task_by_demo_id=task_by_demo_id,
            ratio=args.augmented_sample_ratio,
            max_items=args.augmented_sample_max,
            rng=rng,
        )
        augmented_selected = len(sampled_jobs)

        for job in sampled_jobs:
            demo_id = int(job["demo_id"])
            variant_idx = int(job["variant_idx"])
            requests.append(
                {
                    "request_id": f"aug_{demo_id:04d}_v{variant_idx:02d}",
                    "source_type": "augmented",
                    "demo_id": demo_id,
                    "variant_idx": variant_idx,
                    "task_text": task_by_demo_id.get(demo_id, ""),
                    "video_path": str(job["expected_output_video"]),
                    "keyframes": keyframes,
                    "prompt_profile": job.get("prompt_profile", ""),
                    "seed": int(job.get("seed", -1)),
                }
            )

    args.output_requests_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_requests_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in requests) + "\n",
        encoding="utf-8",
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "include_original": include_original,
        "original_total": len(original_records),
        "original_selected": original_count,
        "augmented_total": augmented_total,
        "augmented_selected": augmented_selected,
        "augmented_sample_ratio": args.augmented_sample_ratio,
        "augmented_sample_max": args.augmented_sample_max,
        "requests_total": len(requests),
        "requests_path": str(args.output_requests_jsonl),
    }
    args.output_report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote requests: {args.output_requests_jsonl}")
    print(f"Wrote report:   {args.output_report_json}")
    print(f"Original selected: {original_count}")
    print(f"Augmented selected: {augmented_selected}")
    print(f"Total requests: {len(requests)}")


def _sample_augmented_jobs(
    jobs: list[dict[str, object]],
    task_by_demo_id: dict[int, str],
    ratio: float,
    max_items: int | None,
    rng: random.Random,
) -> list[dict[str, object]]:
    """Stratified sampling by task text; deterministic via seeded RNG."""
    if ratio <= 0 or not jobs:
        return []

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for job in jobs:
        demo_id = int(job["demo_id"])
        task_text = task_by_demo_id.get(demo_id, "<unknown>")
        grouped[task_text].append(job)

    sampled: list[dict[str, object]] = []
    for task_text in sorted(grouped.keys()):
        group = grouped[task_text]
        k = max(1, math.ceil(len(group) * ratio))
        k = min(k, len(group))
        sampled.extend(rng.sample(group, k))

    if max_items is not None and len(sampled) > max_items:
        sampled = rng.sample(sampled, max_items)

    sampled.sort(key=lambda row: (int(row["demo_id"]), int(row["variant_idx"])))
    return sampled


def _resolve_original_video_path(row: dict[str, object]) -> str:
    """Prefer exported per-demo path, fallback to resolved source path."""
    expected_export_name = row.get("expected_export_name")
    if expected_export_name:
        return str(Path("data/videos/original") / str(expected_export_name))
    return str(row.get("video_path_resolved", ""))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

if __name__ == "__main__":
    main()
