# Cross-Machine Handoff Protocol

This project uses a split execution model:
- local machine (`omni-sim-box`) for dataset integrity, HDF5 merge, training, evaluation
- remote machine for Cosmos Transfer/Reason inference

## Pre-pilot Intake Location (on omni-sim-box)

Use this exact root:

`/home/mathewyoussef/lab-projects/SO101-Pilot/data/remote_handoff/prepilot_incoming`

Subdirectories:
- `videos/`
- `manifests/`
- `logs/`

## Required Pre-pilot Deliverables

From the remote pre-pilot run, copy into the intake root:

1. generated video outputs (`videos/`)
2. run metadata + request/response manifests (`manifests/`)
3. model/runtime logs (`logs/`)
4. checksum file(s) for transferred artifacts (put in `manifests/`)

## Transfer Command Pattern

From the remote machine:

```bash
rsync -avhP --partial --append-verify -e ssh \
  /path/to/remote_prepilot/videos/ \
  mathewyoussef@10.201.212.240:/home/mathewyoussef/lab-projects/SO101-Pilot/data/remote_handoff/prepilot_incoming/videos/

rsync -avhP --partial --append-verify -e ssh \
  /path/to/remote_prepilot/manifests/ \
  mathewyoussef@10.201.212.240:/home/mathewyoussef/lab-projects/SO101-Pilot/data/remote_handoff/prepilot_incoming/manifests/

rsync -avhP --partial --append-verify -e ssh \
  /path/to/remote_prepilot/logs/ \
  mathewyoussef@10.201.212.240:/home/mathewyoussef/lab-projects/SO101-Pilot/data/remote_handoff/prepilot_incoming/logs/
```

## Next Stage After Pre-pilot Intake

After pre-pilot assets arrive and pass integrity checks:

1. finalize per-demo export package (`scripts/02_export_mp4_from_hdf5.py`)
2. build full transfer run bundle (`scripts/03_run_transfer.py`)
3. run budget sizing from pilot throughput (`scripts/03a_transfer_budget_estimator.py`)
4. pick `variants_per_demo` from measured budget
5. execute full remote transfer + reason runs
