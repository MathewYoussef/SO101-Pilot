#!/usr/bin/env python3
"""Helpers for loading and validating SO101 HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


VIDEO_PATH_MARKER = "videos/"


@dataclass(frozen=True)
class EpisodeView:
    """Resolved view of one episode in the canonical HDF5."""

    episode_key: str
    demo_id: int
    num_steps: int
    task_text: str
    video_path_hdf5: str
    video_path_resolved: Path
    from_timestamp: float
    to_timestamp: float
    fps: float


def decode_scalar_string(raw: object) -> str:
    """Decode a scalar HDF5 string payload into Python str."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return decode_scalar_string(raw[()])
    return str(raw)


def resolve_video_path(video_root: Path, raw_video_path: str) -> Path:
    """Resolve a video path stored in HDF5 against a local root."""
    rel = raw_video_path
    if VIDEO_PATH_MARKER in rel:
        rel = rel.split(VIDEO_PATH_MARKER, 1)[1]
    return video_root / rel


def get_episode_group(dataset: h5py.File) -> h5py.Group:
    """Return the top-level episodes group for supported schemas."""
    if "episodes" in dataset:
        return dataset["episodes"]
    raise ValueError("Unsupported schema: expected top-level 'episodes' group.")


def required_episode_paths() -> tuple[str, ...]:
    """Required datasets/groups per episode for V0 loader compatibility."""
    return (
        "action/delta_eef",
        "obs/proprio",
        "obs/eef_pose",
        "meta/frame_index",
        "meta/timestamps",
        "meta/task_text",
        "meta/video/path",
        "meta/video/from_timestamp",
        "meta/video/to_timestamp",
        "meta/video/fps",
    )


def assert_required_structure(dataset: h5py.File) -> None:
    """Raise ValueError if required episode structure is missing."""
    episodes = get_episode_group(dataset)
    if len(episodes.keys()) == 0:
        raise ValueError("Dataset has zero episodes.")

    first_key = sorted(episodes.keys(), key=_sort_key)[0]
    first_episode = episodes[first_key]
    missing = [path for path in required_episode_paths() if path not in first_episode]
    if missing:
        raise ValueError(f"Episode '{first_key}' missing required paths: {missing}")


def iter_episode_views(dataset: h5py.File, video_root: Path) -> Iterator[EpisodeView]:
    """Yield episode views sorted by stable key ordering."""
    episodes = get_episode_group(dataset)
    for episode_key in sorted(episodes.keys(), key=_sort_key):
        episode = episodes[episode_key]
        action = episode["action"]["delta_eef"]
        task_text = decode_scalar_string(episode["meta"]["task_text"][()])
        raw_video = decode_scalar_string(episode["meta"]["video"]["path"][()])
        from_ts = float(episode["meta"]["video"]["from_timestamp"][()])
        to_ts = float(episode["meta"]["video"]["to_timestamp"][()])
        fps = float(episode["meta"]["video"]["fps"][()])

        demo_id = int(episode_key) if str(episode_key).isdigit() else _sort_key(episode_key)
        yield EpisodeView(
            episode_key=str(episode_key),
            demo_id=demo_id,
            num_steps=int(action.shape[0]),
            task_text=task_text,
            video_path_hdf5=raw_video,
            video_path_resolved=resolve_video_path(video_root, raw_video),
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            fps=fps,
        )


def validate_episode_arrays(episode: h5py.Group) -> list[str]:
    """Return a list of validation issues for one episode."""
    issues: list[str] = []
    delta_eef = episode["action"]["delta_eef"][...]
    proprio = episode["obs"]["proprio"][...]
    eef_pose = episode["obs"]["eef_pose"][...]
    frame_index = episode["meta"]["frame_index"][...]
    timestamps = episode["meta"]["timestamps"][...]

    length = len(delta_eef)
    if not (len(proprio) == len(eef_pose) == len(frame_index) == len(timestamps) == length):
        issues.append("length_mismatch")

    if not np.isfinite(delta_eef).all():
        issues.append("delta_eef_non_finite")
    if not np.isfinite(proprio).all():
        issues.append("proprio_non_finite")
    if not np.isfinite(eef_pose).all():
        issues.append("eef_pose_non_finite")
    if not np.all(np.diff(frame_index) >= 0):
        issues.append("frame_index_not_monotonic")
    if not np.all(np.diff(timestamps) >= 0):
        issues.append("timestamps_not_monotonic")
    if delta_eef.ndim != 2 or delta_eef.shape[1] != 6:
        issues.append("delta_eef_shape_not_Tx6")

    return issues


def _sort_key(key: str) -> int:
    """Stable sort helper for episode keys that may be numeric strings."""
    if str(key).isdigit():
        return int(key)
    return abs(hash(str(key)))
