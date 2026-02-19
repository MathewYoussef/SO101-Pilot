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
  semantics/  # reason_labels.jsonl, predicate_labels.jsonl

scripts/
  00_fetch_hf_dataset.py
  00a_validate_transfer.py
  01_lerobot_to_hdf5.py
  02_export_mp4_from_hdf5.py
  03_run_transfer.py
  03a_transfer_budget_estimator.py
  04_mp4_to_hdf5_augmented.py
  05_merge_hdf5.py
  06_run_reason_labels.py
  06a_generate_predicates_from_sim.py
  07_train_multimodal_policy.py
  08_eval_rollouts_isaaclab.py
  09_report.py
  lib/hdf5_adapter.py

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
Scaffold plus initial data-ingest hardening:
- `scripts/00a_validate_transfer.py` validates incoming HDF5/videos and writes manifest/report.
- `scripts/01_lerobot_to_hdf5.py` promotes incoming HDF5 to canonical `data/hdf5/demos.hdf5`.
- `scripts/lib/hdf5_adapter.py` provides one loader/validator path for current HDF5 schema.

## Immediate Data Bring-up Commands
Assuming files were transferred into `data/hdf5/incoming` and `data/videos/incoming`:

```bash
source /home/mathewyoussef/isaac-lab/env_isaaclab/bin/activate
python scripts/00a_validate_transfer.py
python scripts/01_lerobot_to_hdf5.py
```

## Cross-Machine Protocol (Transfer + Reason)
Use a single upload and single download per run cycle to keep lineage deterministic.

Dedicated pre-pilot intake path on this box:
`/home/mathewyoussef/lab-projects/SO101-Pilot/data/remote_handoff/prepilot_incoming`

Detailed handoff instructions:
`docs/cross_machine_handoff.md`

1. Export canonical HDF5 episodes to deterministic per-demo MP4 files
```bash
source /home/mathewyoussef/isaac-lab/env_isaaclab/bin/activate
python scripts/02_export_mp4_from_hdf5.py \
  --input-hdf5 data/hdf5/demos.hdf5 \
  --video-root data/videos/incoming \
  --output-dir data/videos/original
```

2. Build Transfer run bundle locally
```bash
python scripts/03_run_transfer.py \
  --input-videos-dir data/videos/original \
  --variants-per-demo 3 \
  --base-seed 1000
```

3. Estimate compute budget from pilot throughput
```bash
python scripts/03a_transfer_budget_estimator.py \
  --pilot-output-seconds <pilot_output_video_seconds> \
  --pilot-wall-seconds <pilot_wall_seconds> \
  --pilot-num-gpus 1 \
  --target-num-gpus 1
```

4. Run Cosmos Transfer on remote machine using generated run bundle:
- `data/manifests/transfer_runs/<run_id>/transfer_jobs.jsonl`
- `data/manifests/transfer_runs/<run_id>/transfer_input_checksums.sha256`
- `data/manifests/transfer_runs/<run_id>/requests/job_*.json`

5. Build Reason request set (all originals + stratified augmented subset)
```bash
python scripts/06_run_reason_labels.py \
  --original-manifest-jsonl data/manifests/episode_video_manifest.jsonl \
  --transfer-jobs-jsonl data/manifests/transfer_runs/<run_id>/transfer_jobs.jsonl \
  --augmented-sample-ratio 0.25 \
  --seed 0
```

6. Send `data/manifests/reason_requests.jsonl` to remote Reason machine, execute inference there, and return structured labels JSONL for local ingest/training.
