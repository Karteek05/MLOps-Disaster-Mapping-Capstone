# Disaster Damage Mapping

Semantic segmentation of building damage from paired pre/post-disaster
satellite images (xBD/xView2-style). A ResNet50-UNet takes a stacked
6-channel (pre + post) image and predicts a per-pixel damage class:
background, no damage, minor damage, major damage, destroyed.

## Project layout

- `src/data_prep.py` — turns raw xBD image/label pairs into stacked
  6-channel `.npy` inputs and damage-grade mask PNGs.
- `src/model.py` — the ResNet50-UNet architecture.
- `src/data_loader.py` — shared `tf.data` loading/splitting used by both
  training and evaluation, so evaluation always scores the model on the
  exact held-out split it never trained on.
- `src/metrics.py` — `SparseMeanIoU`, shared by train/evaluate/app.
- `src/train.py` — training entrypoint. Set `SMOKE_TEST=true` to run a
  fast synthetic pass (tiny model, random data, no pretrained-weight
  download) for validating the code path without real data or a GPU —
  this is what CI runs.
- `src/evaluate.py` — scores a trained model on the held-out split,
  writes `eval_metrics.json`.
- `app.py` — a Gradio app: upload a pre/post-disaster image pair, get a
  color-coded damage overlay back.
- `dvc.yaml` — the DVC pipeline: `prepare` → `train` → `evaluate`.

## Setup

```bash
pip install -r requirements.txt
dvc pull          # pulls data/raw, data/processed, models/ from the GCS remote
```

DVC's GCS remote credential lives outside the repo (`.dvc/config` points at
a local key file path) — it is intentionally never committed.

## Running the pipeline

```bash
dvc repro          # runs prepare -> train -> evaluate as needed
dvc dag             # visualize the pipeline stages
```

Hyperparameters (including `train.img_size`, the training resolution) live
in `params.yaml`.

## Running the app

```bash
python app.py
```

Requires a trained model at `models/unet_model.h5` (from `dvc pull` or a
local training run).

## GPU training on Windows (e.g. GTX 1650, 4GB VRAM)

TensorFlow dropped native Windows GPU support after 2.10 — `pip install
tensorflow` on Windows is CPU-only. To train with an NVIDIA GPU on Windows:

1. Enable WSL2 and install Ubuntu: `wsl --install -d Ubuntu` (admin
   PowerShell), then confirm `nvidia-smi` works inside the Ubuntu shell.
2. Install the VS Code "WSL" extension and reopen this project via
   Remote-WSL, working from a copy of the repo inside WSL's own
   filesystem (e.g. `~/dev/my_mlops_project`) rather than `/mnt/c/...`,
   for faster disk I/O against the processed dataset.
3. Inside WSL: `pip install "tensorflow[and-cuda]"` (pulls the matching
   CUDA/cuDNN automatically).

The model as configured is sized for that 4GB budget: `train.img_size` in
`params.yaml` defaults to 256 (not the native 1024) and the ResNet50
encoder is frozen by default (`train.pretrained_trainable: false`) —
resolution is the dominant memory cost, and freezing the encoder avoids
storing its gradients/optimizer state. Training also runs under mixed
precision (`mixed_float16`) to roughly halve activation memory (this
doesn't speed up compute on non-Tensor-Core GPUs like the GTX 1650, but
the memory savings help it fit). If 256 still runs out of memory, drop
`train.img_size` to 128.

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`: lint (`ruff`),
unit tests (`pytest`), and a smoke test of the training code path
(`SMOKE_TEST=true python -m src.train`) using tiny synthetic data. It does
**not** run real training — the real ~17GB dataset and multi-hour GPU
training stay local/manual; a trained model is committed via DVC by
whoever ran it.

## Testing locally

```bash
pytest -q
```

If TensorFlow fails to import under `pytest` with a native DLL load error
(seen on a Windows env with many unrelated packages installed, where
pytest's plugin-autoload scan trips something in TF's DLL loading) but
`python -c "import tensorflow"` works fine on its own, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
