#!/usr/bin/env python3
"""Export per-episode MP4 clips from canonical HDF5 for Transfer handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py

from lib.hdf5_adapter import assert_required_structure, iter_episode_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-hdf5",
        type=Path,
        default=Path("data/hdf5/demos.hdf5"),
        help="Canonical HDF5 path.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/videos/incoming"),
        help="Root for source videos referenced by the HDF5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/videos/original"),
        help="Directory to write exported per-episode MP4 files.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/transfer_export_manifest.jsonl"),
        help="Export manifest JSONL path.",
    )
    parser.add_argument(
        "--checksums-out",
        type=Path,
        default=Path("data/manifests/transfer_export_checksums.sha256"),
        help="SHA256 output for exported videos.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/manifests/transfer_export_report.json"),
        help="Export report JSON path.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg binary.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for export count (for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output videos if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan exports without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_hdf5.exists():
        raise SystemExit(f"Missing canonical HDF5: {args.input_hdf5}")
    if not args.video_root.exists():
        raise SystemExit(f"Missing video root: {args.video_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.checksums_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    checksums: list[str] = []

    source_videos: set[str] = set()
    exported_count = 0
    skipped_existing = 0
    missing_sources = 0
    failed_exports = 0

    with h5py.File(args.input_hdf5, "r") as h5_file:
        assert_required_structure(h5_file)
        episodes = list(iter_episode_views(h5_file, args.video_root))
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]

        for view in episodes:
            source_videos.add(str(view.video_path_resolved))
            camera_key = infer_camera_key(view.video_path_hdf5)
            output_name = f"demo_{view.demo_id:04d}_{camera_key}.mp4"
            output_path = args.output_dir / output_name
            status = "exported"
            error_message = None

            if not view.video_path_resolved.exists():
                status = "missing_source"
                missing_sources += 1
                error_message = f"Missing source video: {view.video_path_resolved}"
            elif output_path.exists() and not args.overwrite:
                status = "skipped_existing"
                skipped_existing += 1
            elif not args.dry_run:
                try:
                    clip_video_ffmpeg(
                        ffmpeg_bin=args.ffmpeg_bin,
                        input_video=view.video_path_resolved,
                        output_video=output_path,
                        start_sec=view.from_timestamp,
                        end_sec=view.to_timestamp,
                        fps=view.fps,
                    )
                    exported_count += 1
                except subprocess.CalledProcessError as exc:
                    failed_exports += 1
                    status = "failed"
                    error_message = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            else:
                exported_count += 1

            row = {
                "demo_id": view.demo_id,
                "episode_key": view.episode_key,
                "task_text": view.task_text,
                "source_video_path": str(view.video_path_resolved),
                "output_video_path": str(output_path),
                "output_video_name": output_name,
                "camera_key": camera_key,
                "fps": view.fps,
                "from_timestamp": view.from_timestamp,
                "to_timestamp": view.to_timestamp,
                "num_steps": view.num_steps,
                "status": status,
            }
            if error_message:
                row["error"] = error_message
            manifest_rows.append(row)

            if status in {"exported", "skipped_existing"} and output_path.exists() and not args.dry_run:
                checksums.append(f"{sha256_file(output_path)}  {output_path.name}")

    args.manifest_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    if checksums:
        args.checksums_out.write_text("\n".join(checksums) + "\n", encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_hdf5": str(args.input_hdf5),
        "video_root": str(args.video_root),
        "output_dir": str(args.output_dir),
        "episodes_considered": len(manifest_rows),
        "source_video_files": len(source_videos),
        "exported_count": exported_count,
        "skipped_existing": skipped_existing,
        "missing_sources": missing_sources,
        "failed_exports": failed_exports,
        "dry_run": args.dry_run,
        "manifest_out": str(args.manifest_out),
        "checksums_out": str(args.checksums_out),
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote manifest: {args.manifest_out}")
    print(f"Wrote report:   {args.report_out}")
    if checksums:
        print(f"Wrote checksums:{args.checksums_out}")
    print(f"Episodes considered: {report['episodes_considered']}")
    print(f"Exported: {report['exported_count']}")
    print(f"Skipped existing: {report['skipped_existing']}")
    print(f"Missing sources: {report['missing_sources']}")
    print(f"Failed exports: {report['failed_exports']}")

    if missing_sources > 0 or failed_exports > 0:
        raise SystemExit(1)


def clip_video_ffmpeg(
    ffmpeg_bin: str,
    input_video: Path,
    output_video: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> None:
    """Clip one episode video segment with deterministic re-encode."""
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd="invalid_duration",
            stderr=f"Invalid duration for clip: start={start_sec}, end={end_sec}".encode("utf-8"),
        )

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(input_video),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-r",
        f"{fps:.6f}",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def infer_camera_key(raw_video_path: str) -> str:
    """Infer stable camera key from source path."""
    path_lower = raw_video_path.lower()
    if "observation.images.front" in path_lower:
        return "front"
    if "wrist" in path_lower:
        return "wrist"
    # Fallback to final non-extension token.
    tail = Path(raw_video_path).stem
    return tail.replace(".", "_")


def sha256_file(path: Path) -> str:
    """Compute SHA256 checksum for file."""
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
