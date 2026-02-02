# SO101-Pilot

SO-101 Sim-to-Real Multimodal Policy V0 — a proof-of-concept pipeline for learning a block-stacking policy that aligns **vision**, **action/proprio**, and **semantics**. The goal is not peak performance; the goal is a repeatable, end-to-end system that produces stable artifacts.

## V0 Definition of Done
An engineer can run a short sequence and produce:
- `demos.hdf5` — canonical dataset converted from LeRobot
- `demos_aug.hdf5` — augmented dataset (Cosmos Transfer) merged with originals
- `reason_labels.jsonl` — structured semantic labels per episode (Cosmos Reason)
- `policy_v0.pt` — trained multi-head policy checkpoint
- `eval_metrics.json` + `eval_videos/*.mp4` — rollout evaluation outputs

## Pipeline (Non‑Negotiable Order)
1. **Data conversion:** LeRobot → HDF5
2. **Augmentation:** HDF5 → MP4 → Cosmos Transfer → MP4 → HDF5 → merge
3. **Semantics:** Cosmos Reason → structured JSONL labels
4. **Training:** multi‑head policy (actions + predicates)
5. **Evaluation:** Isaac Lab closed‑loop rollouts + videos + metrics

## Technical Thesis
- Control is modeled as **multimodal next‑action prediction** with an explicit preference axis.
- Policy predicts **ΔEEF + gripper** action chunks from vision + proprio (+ optional text).
- A secondary head predicts **task predicates/progress** (grasp, on‑top, success/failure mode).
- **Cosmos Transfer** broadens appearance while preserving alignment.
- **Cosmos Reason** provides structured teacher/audit signals; Isaac Lab ground truth is primary.

## Rules of Engagement
1. **Build the pipeline, not perfection.** Ugly policies are fine early.
2. **Canonicalize early.** HDF5 is the single source of truth.
3. **Preserve alignment at all costs.** Augmentations must stay in sync.
4. **Actions are ΔEEF + gripper.** Joints are debug/reference only.
5. **Reason is a teacher, not truth.** Use it for structure, not primary success.
6. **Evaluate with closed‑loop rollouts.** Loss curves don’t prove progress.
7. **Every step emits artifacts + checksums.** No manual steps.
8. **Minimize moving parts.** Single camera, short episodes, small configs.

## Success Criteria (V0)
If we can run the full loop and consistently obtain a merged augmented dataset, a trained checkpoint, rollout videos, and a stable metrics report across reruns — V0 is achieved.

## Status
Scaffold in progress. Next steps: define schemas, I/O contracts, run configs, and a minimal end‑to‑end script chain.
