#!/usr/bin/env python3
"""Promote incoming HDF5 to canonical path for pipeline consumption.

This script intentionally implements the "promotion" path first:
`data/hdf5/incoming/demos.hdf5` -> `data/hdf5/demos.hdf5`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incoming-hdf5",
        type=Path,
        default=Path("data/hdf5/incoming/demos.hdf5"),
        help="Incoming HDF5 produced externally (already converted from LeRobot).",
    )
    parser.add_argument(
        "--canonical-hdf5",
        type=Path,
        default=Path("data/hdf5/demos.hdf5"),
        help="Canonical HDF5 path consumed by the rest of the pipeline.",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path("data/manifests/canonical_hdf5_metadata.json"),
        help="Metadata output with source path and sha256.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing canonical target if present.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.incoming_hdf5.exists():
        raise SystemExit(f"Missing incoming HDF5: {args.incoming_hdf5}")

    args.canonical_hdf5.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)

    source_hash = sha256_file(args.incoming_hdf5)
    source_size = args.incoming_hdf5.stat().st_size

    if args.canonical_hdf5.exists():
        target_hash = sha256_file(args.canonical_hdf5)
        if target_hash == source_hash:
            print(f"Canonical file already up to date: {args.canonical_hdf5}")
        elif not args.force:
            raise SystemExit(
                f"Target exists and differs from source. Use --force to overwrite: {args.canonical_hdf5}"
            )
        else:
            shutil.copy2(args.incoming_hdf5, args.canonical_hdf5)
            print(f"Overwrote canonical HDF5: {args.canonical_hdf5}")
    else:
        shutil.copy2(args.incoming_hdf5, args.canonical_hdf5)
        print(f"Created canonical HDF5: {args.canonical_hdf5}")

    canonical_hash = sha256_file(args.canonical_hdf5)
    meta = {
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "incoming_hdf5": str(args.incoming_hdf5),
        "canonical_hdf5": str(args.canonical_hdf5),
        "source_size_bytes": source_size,
        "source_sha256": source_hash,
        "canonical_sha256": canonical_hash,
        "content_match": source_hash == canonical_hash,
    }
    args.meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote metadata: {args.meta_out}")


if __name__ == "__main__":
    main()
