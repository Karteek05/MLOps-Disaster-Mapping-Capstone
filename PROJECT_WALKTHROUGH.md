# Disaster Damage Mapping — Project Walkthrough

## 1. Problem

Given a pre-disaster and post-disaster satellite image of the same location
(xBD/xView2-style pairs), predict a per-pixel building damage class:

| Class | Meaning |
|---|---|
| 0 | Background / no building |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |

This is a semantic segmentation task — the output is a full-resolution class
map, not a single label, so the model has to localize buildings and grade
their damage simultaneously.

## 2. Data pipeline

Source data is the xBD dataset: paired pre/post images plus JSON polygon
annotations (building footprints with a `damage_grade` property).

Pipeline stages, orchestrated by DVC (`dvc.yaml`, run via `dvc repro`):

1. **`prepare`** ([src/data_prep.py](src/data_prep.py)) — for each
   post-disaster JSON, rasterizes the building polygons into a single-channel
   PNG mask (pixel value = damage class), and stacks the pre/post RGB images
   into a 6-channel `.npy` array. Raw JSON polygons in, model-ready tensors
   out.
2. **`train`** ([src/train.py](src/train.py)) — trains the model (below) on
   the stacked arrays, logs params/metrics to MLflow, saves
   `models/unet_model.h5`.
3. **`evaluate`** ([src/evaluate.py](src/evaluate.py)) — scores the saved
   model against a held-out split, writes `eval_metrics.json`.

The train/val split logic lives once in
[src/data_loader.py](src/data_loader.py) (`list_train_val_files`, seeded
shuffle) and is shared by both `train.py` and `evaluate.py`, so evaluation is
guaranteed to score on data the model never trained on — not two
independently-implemented splits that could silently drift apart.

Data and model artifacts (`data/raw`, `data/processed`, `models/`) are
git-ignored and tracked with **DVC** instead, pointed at a **DagsHub** remote
— so the repo stays small while the ~17GB processed dataset and 300MB model
stay reproducible via `dvc pull`/`dvc repro`.

## 3. Model

A ResNet50-UNet ([src/model.py](src/model.py)):

- **Input**: 6 channels (3 pre + 3 post RGB, stacked), no normalization —
  the model is trained directly on raw `[0, 255]` pixel values, and
  inference must match that.
- **Encoder**: ResNet50, ImageNet-pretrained, frozen by default
  (`train.pretrained_trainable: false` in `params.yaml`) to keep
  memory/compute down on small GPUs. A `1x1` conv adapts the 6-channel input
  down to 3 channels so the pretrained ResNet50 weights still apply.
- **Decoder**: a standard UNet upsampling path with skip connections from
  four ResNet50 stages, transposed-conv upsampling, concatenation with the
  matching encoder feature map, then a conv block.
- **Output**: `img_size × img_size × 5` softmax — a per-pixel class
  distribution over the 5 damage classes.
- **Training resolution**: 256×256 (`params.yaml: train.img_size`), not the
  native 1024×1024 — sized to fit a 4GB GPU (e.g. GTX 1650). Mixed precision
  (`mixed_float16`) roughly halves activation memory.
- **Loss/metric**: sparse categorical crossentropy; IoU tracked via a custom
  `SparseMeanIoU` metric ([src/metrics.py](src/metrics.py)) built on a
  confusion matrix, since Keras doesn't ship a sparse-label IoU out of the
  box.

Latest logged results ([eval_metrics.json](eval_metrics.json)):
`eval_iou = 0.261`, `eval_accuracy = 0.956`, `eval_loss = 0.111` — the high
accuracy alongside a modest IoU is typical for this kind of class-imbalanced
segmentation task (background dominates most pixels; damage classes are the
hard, sparse ones the IoU actually stresses).

## 4. MLOps tooling

- **DVC** — versions data and model artifacts, defines the
  `prepare → train → evaluate` pipeline as a DAG (`dvc.yaml`/`dvc.lock`), so
  anyone can `dvc repro` and get byte-identical outputs given the same
  inputs. Remote storage is DagsHub.
- **MLflow** — tracks hyperparameters and metrics per training run
  (`mlflow.log_params`/`log_metric` in `train.py`). Deliberately does *not*
  log the full model binary to `mlruns/` — that's DVC's job, to avoid
  duplicating hundreds of MB per run.
- **CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) — on every
  push/PR to `main`: lint (`ruff`), unit tests (`pytest`), and a *smoke test*
  of the training code path (`SMOKE_TEST=true`) using a tiny model and
  random synthetic data. This validates the training script itself runs
  end-to-end without requiring the real ~17GB dataset or a GPU in CI — real
  training stays a manual, local/WSL2 step, with the resulting model
  committed via DVC.
- **Tests** ([tests/](tests)) — cover data prep, metrics, and model
  construction in isolation from the full pipeline.

## 5. Serving layer

Two front-ends over the same inference logic (resize → cast to float32 →
stack → `model.predict` → argmax → color-coded overlay blended over the
post-disaster image):

- **Gradio app** ([app.py](app.py)) — quick local demo, `python app.py`,
  drag-and-drop pre/post images in, get a blended damage-map image out.
- **FastAPI + Next.js app** ([backend/](backend), [frontend/](frontend)) —
  the production-shaped version. `backend/app/main.py` exposes `POST
  /predict` (multipart pre/post image upload → JSON with a base64 PNG
  overlay plus a per-class pixel breakdown, see
  [backend/app/schemas.py](backend/app/schemas.py)) and `GET /health`. The
  Next.js frontend ([frontend/src/](frontend/src)) is a drag-and-drop upload
  UI (`ImageDropzone.tsx`, `UploadPanel.tsx`) with a before/after result view
  (`ResultView.tsx`, `DamageLegend.tsx`).

Both share the same model-loading/inference code path conceptually;
`backend/app/inference.py` is effectively `app.py`'s logic adapted to
FastAPI's request/response cycle instead of Gradio's.

## 6. Deployment

- **Backend** — containerized ([backend/Dockerfile](backend/Dockerfile)),
  built from the repo root (needs `src/`, `params.yaml`, `models/` alongside
  `backend/`). Since the model is DVC-tracked (not committed to git), the
  image is built and pushed locally — where the model file already exists on
  disk — to a registry (Docker Hub), and deployed on **Render** from that
  prebuilt image rather than built from the GitHub repo directly.
- **Frontend** — deployed to **Vercel**, root directory `frontend/`, pointed
  at the Render backend URL via `NEXT_PUBLIC_API_BASE_URL`.
- CORS on the backend (`CORS_ORIGINS` env var,
  [backend/app/main.py:28](backend/app/main.py#L28)) is scoped to the
  deployed frontend origin.

## 7. Tech stack summary

| Layer | Tools |
|---|---|
| Model | TensorFlow/Keras (ResNet50-UNet) |
| Data/pipeline versioning | DVC + DagsHub remote |
| Experiment tracking | MLflow |
| CI | GitHub Actions (ruff, pytest, training smoke test) |
| Demo UI | Gradio |
| Production API | FastAPI |
| Production UI | Next.js, React, Tailwind |
| Backend deploy | Docker → Render |
| Frontend deploy | Vercel |

## 8. Repo layout

```
src/            data prep, model, data loader, metrics, train, evaluate
app.py          Gradio demo app
backend/        FastAPI inference service (Dockerfile, main.py, inference.py, schemas.py)
frontend/       Next.js UI
tests/          unit tests for data prep, metrics, model
dvc.yaml        prepare -> train -> evaluate pipeline
params.yaml     hyperparameters (img_size, epochs, lr, ...)
.github/workflows/ci.yml   lint + tests + training smoke test
```
