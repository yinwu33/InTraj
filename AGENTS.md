# AGENTS Log

## 2026-04-08

- Reviewed `reference/SMART` and identified the core migration surface: `smart/model/smart.py`, `smart/modules/*`, `smart/datasets/preprocess.py`, `smart/transforms/target_builder.py`, and `data_preprocess_av2.py`.
- Added a reference-structured SMART port under `models/smart`, preserving module names and parameterization so the port can be checked directly against the reference implementation.
- Added an AV2 SMART datamodule and dataset under `datamodule/`, with lazy per-scenario preprocessing from raw AV2 logs into SMART base cache files and tokenization at load time.
- Added Hydra configs for SMART model and datamodule wiring: `configs/config_smart.yaml`, `configs/model/smart.yaml`, and `configs/datamodule/av2_smart.yaml`.
- Updated package exports so `main.py` can instantiate SMART through Hydra.
- Updated `pyproject.toml` and `uv.lock` to reflect SMART's direct `scipy` dependency and constrained `requires-python` to `3.10.x` so `uv` can resolve the pinned PyG wheel set.
- Fixed SMART integration issues discovered during porting:
  - converted absolute `smart.*` imports to package-relative imports under `models/smart`
  - made Waymo proto import optional in `models/smart/predictor.py` so AV2-only environments can import the model
  - added token-file path resolution with fallback to `reference/SMART/smart/tokens/*`
  - added repo-compatible logging keys (`train/loss`, `val/loss`, `test/loss`) and a `test_step`
  - added `create_scenario` support for the existing visualization callback
  - fixed the AV2 preprocess bug where `av_idx` was lost because the reference script wrote `av_index`
  - adapted the SMART target builder for AV2 and robust ego fallback selection
  - fixed a bf16 dtype mismatch in `models/smart/modules/agent_decoder.py`
- Verified syntax with `uv run python -m py_compile` on the SMART model and datamodule files.
- Verified the ported SMART parameter structure against `reference/SMART`: parameter count, `state_dict` key set, and tensor shapes all match exactly (`7,151,296` params, `818` keys, zero mismatches).
- Ran a manual real-batch SMART forward/training-step check and confirmed finite loss computation on `mini_train`.
- Ran `uv run python main.py --config-name config_smart` with `mini_train` for both train/val as a smoke test and confirmed end-to-end train/val execution.
- Ran a longer SMART smoke test (`limit_train_batches=8`, `limit_val_batches=2`) and observed `train/loss_step` decreasing from `7.72` to about `7.39`, with `train/loss_epoch=7.50` and `val/loss=7.45`.
- Ran `uv run python main.py --config-name config_smart mode=eval` from the saved checkpoint and confirmed the eval path executes successfully (`test/loss=7.42`, `test/minADE=7.67`, `test/minFDE=37.34` on the smoke batch).

## 2026-03-27

- Reviewed `main.py`, `models/simpl`, `models/vectornet`, current datamodules, and `reference/QCNet`.
- Installed original QCNet runtime dependencies with `uv`, including `torch-cluster` and `torch-scatter`, by updating `pyproject.toml` and syncing the environment.
- Replaced the earlier simplified QCNet-style rewrite with a reference-structured QCNet port under `models/qcnet`, preserving the original encoder/decoder/loss/metric module layout and parameterization.
- Reworked the AV2 QCNet datamodule to emit `torch_geometric` `HeteroData` and added a QCNet target builder, without modifying `data/`.
- Updated Hydra configs so `main.py --config-name config_qcnet` instantiates the reference-structured QCNet and datamodule correctly.
- Verified syntax with `uv run python -m py_compile` on the QCNet and datamodule files.
- Verified the ported QCNet parameter structure against `reference/QCNet`: parameter count, `state_dict` key set, and tensor shapes all match exactly (`7,714,501` params, `1009` keys, zero mismatches).
- Fixed integration issues discovered during smoke testing:
  - missing `Path` import in the QCNet AV2 dataset
  - duplicated AV2 map path join during preprocessing
  - device mismatch in `models/qcnet/__init__.py:create_scenario`
  - indexing bug in `utils/viz.py` exposed by QCNet batches
- Ran `uv run python main.py --config-name config_qcnet` with `mini_train` for both train/val as a fast smoke test and confirmed end-to-end train/val execution.
- Ran a longer `uv run` smoke test (`limit_train_batches=8`, `limit_val_batches=2`) and observed `train/loss_step` decreasing from `49.4` to about `31.4`, with `train/loss_epoch=35.2` and `val/loss=26.6`.
- Ran `uv run python main.py --config-name config_qcnet mode=eval` from the saved checkpoint and confirmed the eval path executes successfully (`test/loss=24.72` on the smoke batch).
