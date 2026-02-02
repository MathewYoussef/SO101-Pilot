# SO101-Pilot

SO-101 Sim-to-Real Multimodal Policy V0 — a proof-of-concept pipeline for learning a block-stacking policy that aligns vision, action/proprio, and semantics. V0 success means integration and repeatability, not peak performance.

Note: Cosmos Transfer and Cosmos Reason will run on a separate machine from this repo's primary dev box.

## V0 Definition of Done
You are done with V0 when you can run an end-to-end sequence that produces:
- `data/hdf5/demos.hdf5` (canonical dataset from LeRobot)
- `data/hdf5/demos_aug.hdf5` (augmented dataset merged with originals)
- `data/semantics/reason_labels.jsonl` (structured semantics per episode)
- `outputs/checkpoints/policy_v0.pt` (multi-head policy checkpoint)
- `outputs/reports/eval_metrics.json` + `outputs/eval_videos/*.mp4`

## Non-Negotiable Order
1. LeRobot -> HDF5 conversion
2. HDF5 -> MP4 -> Cosmos Transfer -> MP4 -> HDF5 -> merge
3. Cosmos Reason -> structured JSONL labels
4. Train multi-head policy (actions + predicates)
5. Closed-loop rollout evaluation in Isaac Lab

## Core Rules
- HDF5 is the canonical dataset format.
- Visual augmentation must preserve action/state alignment.
- Actions are Delta-EEF + gripper. Joints are debug/reference only.
- Evaluation is closed-loop rollouts (loss curves do not prove progress).
- Every step emits artifacts, run IDs, config snapshots, and checksums.

## Repository Layout (V0 scaffold)
```
configs/
  dataset.yaml
  transfer.yaml
  reason.yaml
  model.yaml
  train.yaml
  eval.yaml

data/
  raw_lerobot/
  hdf5/
  videos/original/
  videos/augmented/
  semantics/

scripts/
  00_fetch_hf_dataset.py
  01_lerobot_to_hdf5.py
  02_export_mp4_from_hdf5.py
  03_run_transfer.py
  04_mp4_to_hdf5_augmented.py
  05_merge_hdf5.py
  06_run_reason_labels.py
  07_train_multimodal_policy.py
  08_eval_rollouts_isaaclab.py
  09_report.py

isaaclab_ext/
  robots/so101/
  tasks/so101_block_stack/

models/
  tokenizers.py
  multimodal_transformer.py
  heads.py

outputs/
  checkpoints/
  eval_videos/
  reports/
```

## Status
Scaffold only. No implementation yet. Next: fill in configs, define schemas, and wire each script end-to-end.
