# AGENTS.md — SO101-Pilot

## Mission
Build a repeatable sim‑to‑real pipeline for SO‑101 block stacking that aligns vision, action/proprio, and semantics, and produces stable artifacts end‑to‑end.

## Non‑Negotiables
- HDF5 is the canonical dataset format.
- Augmentation must preserve exact frame/action alignment.
- Actions are **ΔEEF + gripper**. Joints are debug/reference only.
- Evaluation is **closed‑loop rollouts** in Isaac Lab.
- Every step emits artifacts, run IDs, config snapshots, and checksums.

## Execution Order (Do Not Reorder)
1. LeRobot → HDF5 conversion
2. HDF5 → MP4 → Transfer → MP4 → HDF5 → merge
3. Reason → structured JSONL labels
4. Train multi‑head policy (actions + predicates)
5. Closed‑loop rollout evaluation

## Engineering Priorities
- Define schemas and I/O contracts before model work.
- Determinism: fixed seeds and reproducible runs.
- Instrumentation: always log, always save videos/metrics.
- Fail fast with dataset alignment checks and indexing tests.

## Anti‑Patterns to Avoid
- Training before dataset integrity is verified.
- Free‑form Reason captions (must be structured labels).
- Changing formats mid‑pipeline.
- Treating validation loss as success.

## Collaboration Hygiene
- Keep changes modular and documented.
- Prefer small, composable scripts with explicit inputs/outputs.
- Add tests for alignment and schema correctness as soon as possible.
