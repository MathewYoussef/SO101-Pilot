#!/usr/bin/env python3
"""Validate transferred HDF5/video payload and emit manifest + report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py

from lib.hdf5_adapter import assert_required_structure, get_episode_group, iter_episode_views, validate_episode_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path("data/hdf5/incoming/demos.hdf5"),
        help="Incoming HDF5 file to validate.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/videos/incoming"),
        help="Root directory for transferred videos.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/episode_video_manifest.jsonl"),
        help="Output JSONL path for episode-to-video mapping.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/manifests/incoming_validation_report.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hdf5.exists():
        raise SystemExit(f"Missing HDF5: {args.hdf5}")
    if not args.video_root.exists():
        raise SystemExit(f"Missing video root: {args.video_root}")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hdf5": str(args.hdf5),
        "video_root": str(args.video_root),
        "episodes_total": 0,
        "episodes_with_issues": 0,
        "missing_video_files": 0,
        "unique_video_files": 0,
        "issues": [],
    }

    manifest_lines: list[str] = []
    all_video_paths: set[str] = set()
    issue_records: list[dict[str, object]] = []

    with h5py.File(args.hdf5, "r") as h5_file:
        assert_required_structure(h5_file)
        episodes = get_episode_group(h5_file)
        episode_views = list(iter_episode_views(h5_file, args.video_root))
        summary["episodes_total"] = len(episode_views)

        for episode_view in episode_views:
            episode = episodes[episode_view.episode_key]
            episode_issues = validate_episode_arrays(episode)
            video_exists = episode_view.video_path_resolved.exists()
            all_video_paths.add(str(episode_view.video_path_resolved))

            if not video_exists:
                episode_issues.append("video_file_missing")

            if episode_issues:
                issue_records.append(
                    {
                        "demo_id": episode_view.demo_id,
                        "episode_key": episode_view.episode_key,
                        "issues": episode_issues,
                    }
                )

            record = {
                "demo_id": episode_view.demo_id,
                "episode_key": episode_view.episode_key,
                "num_steps": episode_view.num_steps,
                "task_text": episode_view.task_text,
                "video_path_hdf5": episode_view.video_path_hdf5,
                "video_path_resolved": str(episode_view.video_path_resolved),
                "video_exists": video_exists,
                "from_timestamp": episode_view.from_timestamp,
                "to_timestamp": episode_view.to_timestamp,
                "fps": episode_view.fps,
                # Stable expected naming for downstream per-demo exports.
                "expected_export_name": f"demo_{episode_view.demo_id:04d}_front.mp4",
            }
            manifest_lines.append(json.dumps(record, ensure_ascii=True))

    summary["episodes_with_issues"] = len(issue_records)
    summary["issues"] = issue_records
    summary["unique_video_files"] = len(all_video_paths)
    summary["missing_video_files"] = sum(
        1 for line in manifest_lines if '"video_exists": false' in line
    )

    args.manifest_out.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    args.report_out.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote manifest: {args.manifest_out}")
    print(f"Wrote report:   {args.report_out}")
    print(f"Episodes:       {summary['episodes_total']}")
    print(f"Issues:         {summary['episodes_with_issues']}")

    if issue_records:
        print("Validation failed: dataset has issues.")
        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        raise
