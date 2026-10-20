# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Training search support (issue #168 of the thesis repository)**:
  - `model-training/templates/Torchvision/Classification/EfficientNet`: validation runs every epoch (`val_dataloader`, `validation_step`, `val_loss`, `val_acc`, `val_f1`; `limit_val_batches=0` removed from `util.py`), the new `metrics_callback.py` writes one line per epoch to `/train/metrics.jsonl` (and repeats the last line with `"saved": true` once the model is written), and the image carries the label `agrigaia.metrics="jsonl/1"`. Early stopping still monitors `train_loss`.
  - `model-training/common/docker-entrypoint.sh`: when `nvidia-smi -L` works, GPU usage is sampled every 5 s into `/train/gpu.csv`.
  - `agri_gaia_backend/routers/train.py`: `GET /train/containers/{id}/metrics` (epoch figures and GPU samples, also for exited containers), `POST /train/containers/{id}/clone` (copy of a built container with another `train_config`, same image, no build; new column `train_containers.config_override`, Alembic revision `7d3a1c5e9b42`), and `DELETE /train/containers/{id}` keeps the image while another row uses it.

## [1.0.0] - 2026-06-15

### Changed

- **Python 3.11 → 3.12**: base image bumped to `python:3.12-bookworm` in both `Dockerfile` and `Dockerfile-dev`. (f05ff91)
- **NumPy 1 → 2**: `numpy` `~=1.23.1` → `~=2.4.1`. (f05ff91)
- **ONNX**: `onnx` `~=1.14.0` → `~=1.21.0`. (f05ff91)
- **Uvicorn**: `~=0.27.1` → `~=0.30.6` (initial Python 3.12 compatibility), then → `~=0.46.0`. (ea8eb6c, c5c1830)
- **requirements-parser**: `~=0.5.0` → `~=0.11.0` (Python 3.12 compatibility). (ea8eb6c)
- **Docker SDK**: `docker` `~=6.1.0` → `~=7.1.0`. (c5c1830)
- **requests**: `~=2.30.0` → `~=2.33.1`. (f05ff91, c5c1830)
- **python-dotenv**: `~=1.0.1` → `~=1.2.2`. (7249442)
- **xmltodict**: `~=0.13.0` → `~=1.0.4`. (7249442)
- **FiftyOne**: `~=0.25.2` → `~=1.14.1`. (7249442)
- **OpenAPI tooling**: `jsonschema` `~=4.17.3` → `~=4.23.0`; `openapi-spec-validator` `~=0.5.7` → `~=0.7.1`. Resolves the `'_io.BufferedReader' object has no attribute 'decode'` 500 error on Python 3.12 service-spec upload. (e1f206c)
- **Model training templates (EfficientNet, Mask R-CNN, YOLOv5)**:
  - `torch` `==1.13.0` → `~=2.8.0`.
  - `pytorch-lightning` `~=1.7.7` → `~=2.4.0`.
  - `ujson` `~=5.4.0` → `~=5.12.0`.
  - `scikit-learn` `~=1.1.2` → `~=1.5.0`. (c5c1830)
- **ONNX export schema** (`model-training/export/export.jsonschema`, `model-training/export/export.py`): removed the deprecated `training` and `operator_export_type` fields — neither is supported by the PyTorch 2.x ONNX exporter. Added a new `dynamo` boolean to opt into the `torch.export`-based exporter (default in `torch>=2.9`). Defaults to `false` because the dynamo exporter cannot trace data-dependent shapes in `batched_nms`, which means it cannot export torchvision detection models (Mask R-CNN, Faster R-CNN, RetinaNet, SSD, ...). (879e536)
- **Nuclio `nuctl deploy` command** (`agri_gaia_backend/routers/datasets.py`): now passes `--platform-config='{"attributes":{"network":"agri_gaia_network"}}'` so auto-annotation function containers attach to `agri_gaia_network` rather than Docker's default `bridge`. Without this override, CVAT cannot reach deployed functions (every annotation click returns `504`). Required by the Nuclio `1.16.4` bump in the platform repo, since `nuctl 1.16` initialises its platform config with an empty path and ignores any mounted `platform.yaml` for deploys triggered via the Docker socket. (1fdf293)
- **GPU-aware PyTorch CUDA wheel selection for training images**: added `get_torch_cuda_index()` to `agri_gaia_backend/util/common.py` (picks cu126 for Volta..Hopper or cu128 for Turing..Blackwell based on `nvidia-smi --query-gpu=compute_cap`). `agri_gaia_backend/routers/train.py::build_train_image` now forwards the selected index to the image builder as a `TORCH_CUDA_INDEX` build arg. All four template Dockerfiles (EfficientNet, Mask R-CNN, YOLOv5, YOLOv11) gained `ARG TORCH_CUDA_INDEX` (default `cu126`) and now install `torch~=2.11.0` / `torchvision~=0.26.0` from that index; the EfficientNet and Mask R-CNN `requirements.txt` files dropped their `torch` / `torchvision` entries (the Dockerfile RUN owns the install). First introduced Volta (V100/DGXS, CC 7.0) support to the training pipeline. (e2431b5)
- **PyTorch CUDA wheel index hard-pinned to cu126** for the YOLOv5 / YOLOv11 training images (`agri_gaia_backend/util/common.py::get_torch_cuda_index()`). The previous `nvidia-smi` auto-detection keyed off the **build** host's GPUs, which may not match the **run** host (the V100/DGXS, CC 7.0). cu128 dropped Volta (`sm_70`) as of torch 2.11; cu126 is the newest line that still ships `sm_70` kernels and covers CC 7.0–9.0 (Volta..Hopper). The YOLOv5 / YOLOv11 Dockerfiles now derive the local-version tag from the index URL and install `torch==<ver>+cu126` (with the explicit local version label) — a bare `torch==2.11.0` was a PEP 440 no-op against the cu128 wheel pre-installed in the Ultralytics base image, so the cu126 wheel was never actually swapped in. (26ca34e)
- **EDC connector URLs**: `agri_gaia_backend/services/edc/connector.py` and `.env` updated for the EDC service rename in the platform repo. `connector_data_url` now points at `https://edc.<base>` (was `edc-provider.<base>`), `connector_ids_url` at `https://edc-ids.<base>` (was `edc-provider-ids.<base>`); `PROVIDER_DATA_ENDPOINT` switched from `http://edc_provider:8182` to `http://edc:8182`. Driven by Let's Encrypt CN-length limits on the issued wildcard certificate. (14f7e36, eb7f1da)

### Fixed

- **Mask R-CNN template** migrated to the PyTorch Lightning 2.x trainer / callback API (updates in `maskrcnn.py` and `util.py`). (2b17c55)
- **Mask R-CNN training template** (`.../docker/maskrcnn.py`): guard `float(map_value)` with `map_value.numel() == 1` to handle multi-element tensors returned by `MeanAveragePrecision.compute()` under TorchMetrics 1.x. (879e536)
- **Mask R-CNN training template** (`.../docker/util.py`): `configure_strategy` now returns the string literal `"auto"` as a fallback (matching Lightning 2.x's accepted strategy keyword) instead of propagating an unknown strategy string back to the trainer. (879e536)
- **YOLOv5 / YOLOv11 export path** (`model-training/templates/Ultralytics/Object Detection/YOLOv{5,11}/docker/train.py`): now normalises the actual Ultralytics run directory (`<project>/<name>/weights/best.pt`) to the fixed location baked into `custom.json` (`/train/detect`). Ultralytics' `name` argument can drift from that fixed path, which previously broke both the ONNX export step and the backend's model retrieval. The fix runs whether or not export is enabled. (26ca34e)
