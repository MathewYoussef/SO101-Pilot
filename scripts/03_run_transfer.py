#!/usr/bin/env python3
"""Build a deterministic Transfer handoff bundle for remote execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


DEMO_NAME_PATTERN = re.compile(r"^demo_(\d+)_(.+)\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-videos-dir",
        type=Path,
        default=Path("data/videos/original"),
        help="Directory containing exported source videos (demo_<id>_*.mp4).",
    )
    parser.add_argument(
        "--video-glob",
        type=str,
        default="demo_*_*.mp4",
        help="Glob pattern for source videos within --input-videos-dir.",
    )
    parser.add_argument("--variants-per-demo", type=int, default=3, help="Number of synthetic variants per demo.")
    parser.add_argument("--base-seed", type=int, default=1000, help="Base seed for deterministic per-job seeds.")
    parser.add_argument(
        "--prompt-profiles",
        type=str,
        default="lighting,texture,sensor",
        help="Comma-separated profile tags; cycled across variants.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run ID. Defaults to UTC timestamp.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("data/manifests/transfer_runs"),
        help="Root directory to write run bundles.",
    )
    parser.add_argument(
        "--remote-input-root",
        type=str,
        default="/workspace/so101/transfer_input",
        help="Input root path expected on remote Transfer machine.",
    )
    parser.add_argument(
        "--remote-output-root",
        type=str,
        default="/workspace/so101/transfer_output",
        help="Output root path expected on remote Transfer machine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variants_per_demo < 1:
        raise SystemExit("--variants-per-demo must be >= 1")

    input_videos = sorted(args.input_videos_dir.glob(args.video_glob))
    if not input_videos:
        raise SystemExit(f"No videos found in {args.input_videos_dir} with glob {args.video_glob}")

    prompt_profiles = [value.strip() for value in args.prompt_profiles.split(",") if value.strip()]
    if not prompt_profiles:
        raise SystemExit("--prompt-profiles must contain at least one value")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("transfer_%Y%m%d_%H%M%S")
    run_dir = args.run_root / run_id
    params_dir = run_dir / "requests"
    run_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    jobs_path = run_dir / "transfer_jobs.jsonl"
    expected_path = run_dir / "transfer_expected_outputs.jsonl"
    checksums_path = run_dir / "transfer_input_checksums.sha256"
    metadata_path = run_dir / "run_metadata.json"

    job_lines: list[str] = []
    expected_lines: list[str] = []
    checksum_lines: list[str] = []
    unique_demo_ids: set[int] = set()

    job_id = 0
    for video_path in input_videos:
        parsed = parse_demo_filename(video_path.name)
        if parsed is None:
            continue
        demo_id, camera_key = parsed
        unique_demo_ids.add(demo_id)
        sha = sha256_file(video_path)
        checksum_lines.append(f"{sha}  {video_path.name}")

        for variant_idx in range(args.variants_per_demo):
            profile = prompt_profiles[variant_idx % len(prompt_profiles)]
            seed = args.base_seed + demo_id * 1000 + variant_idx
            output_name = f"demo_{demo_id:04d}_v{variant_idx:02d}_{camera_key}.mp4"

            request_payload = {
                "run_id": run_id,
                "job_id": f"job_{job_id:06d}",
                "demo_id": demo_id,
                "variant_idx": variant_idx,
                "prompt_profile": profile,
                "seed": seed,
                "local_input_video": str(video_path),
                "local_expected_output_video": str(Path("data/videos/augmented") / output_name),
                "remote_input_video": f"{args.remote_input_root}/{video_path.name}",
                "remote_expected_output_video": f"{args.remote_output_root}/{output_name}",
                "notes": [
                    "Map this request payload into the exact Cosmos Transfer2.5 params schema used on remote.",
                    "Keep demo_id and variant_idx unchanged for deterministic import mapping.",
                ],
            }
            request_path = params_dir / f"job_{job_id:06d}.json"
            request_path.write_text(json.dumps(request_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

            job_record = {
                "run_id": run_id,
                "job_id": f"job_{job_id:06d}",
                "demo_id": demo_id,
                "variant_idx": variant_idx,
                "camera_key": camera_key,
                "seed": seed,
                "prompt_profile": profile,
                "input_video": str(video_path),
                "expected_output_video": str(Path("data/videos/augmented") / output_name),
                "request_json": str(request_path),
            }
            expected_record = {
                "run_id": run_id,
                "job_id": f"job_{job_id:06d}",
                "demo_id": demo_id,
                "variant_idx": variant_idx,
                "expected_output_filename": output_name,
                "expected_output_video": str(Path("data/videos/augmented") / output_name),
            }
            job_lines.append(json.dumps(job_record, ensure_ascii=True))
            expected_lines.append(json.dumps(expected_record, ensure_ascii=True))
            job_id += 1

    if not job_lines:
        raise SystemExit(
            "No input filenames matched expected pattern 'demo_<id>_<camera>.mp4'. "
            "Run export first or adjust --video-glob and naming."
        )

    jobs_path.write_text("\n".join(job_lines) + "\n", encoding="utf-8")
    expected_path.write_text("\n".join(expected_lines) + "\n", encoding="utf-8")
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "input_videos_dir": str(args.input_videos_dir),
        "video_glob": args.video_glob,
        "variants_per_demo": args.variants_per_demo,
        "prompt_profiles": prompt_profiles,
        "base_seed": args.base_seed,
        "remote_input_root": args.remote_input_root,
        "remote_output_root": args.remote_output_root,
        "source_video_files": len(input_videos),
        "unique_demo_ids": len(unique_demo_ids),
        "jobs_total": len(job_lines),
        "jobs_manifest": str(jobs_path),
        "expected_outputs_manifest": str(expected_path),
        "input_checksums": str(checksums_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Run bundle: {run_dir}")
    print(f"Jobs: {len(job_lines)}")
    print(f"Unique demos: {len(unique_demo_ids)}")
    print(f"Checksums: {checksums_path}")
    print(f"Metadata: {metadata_path}")


def parse_demo_filename(name: str) -> tuple[int, str] | None:
    """Extract (demo_id, camera_key) from demo filename."""
    match = DEMO_NAME_PATTERN.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def sha256_file(path: Path) -> str:
    """Compute SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

if __name__ == "__main__":
    main()
