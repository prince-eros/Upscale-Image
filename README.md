# Real-ESRGAN Upscaling Pipeline

Production-grade pipeline for preparing the `Original-indian_cultural` dataset for Sana 1.6B training.

## What It Does

- Preserves bucket structure for all 12 categories.
- Never upscales images with short-side >= 512.
- Upscales smaller images with bucket-aware model selection.
- Copies matching `.txt` metadata with every image.
- Computes LPIPS for every upscaled image.
- Moves rejected outputs (`LPIPS > 0.50`) to `rejected/`.
- Writes per-bucket CSVs, `summary.csv`, `visual_qa_samples.csv`, and `upscaling_notes.md`.
- Handles corrupt images gracefully and logs failures.

## Required Weights

Place official Real-ESRGAN models under:

```text
weights/
	RealESRGAN_x4plus.pth
	RealESRGAN_x4plus_anime_6B.pth
	RealESRGAN_x2plus.pth
```

## Environment Setup (uv)

```bash
uv init realesrgan_pipeline
cd realesrgan_pipeline
uv venv
.venv\Scripts\activate
uv pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
uv pip install basicsr realesrgan lpips opencv-python pillow tqdm pandas
uv pip install "numpy<2"
```

If you are using this existing workspace, you can skip `uv init` and run:

```bash
uv venv
.venv\Scripts\activate
uv sync
```

## CLI

### Test run (required first)

```bash
python pipeline.py --input Original-indian_cultural --output upcaled-indian_cultural --weights weights --test-limit 10
```

### Full run

```bash
python pipeline.py --input Original-indian_cultural --output upcaled-indian_cultural --weights weights --full-run
```

### Run one bucket only

```bash
python pipeline.py --input Original-indian_cultural --output upcaled-indian_cultural --weights weights --full-run --bucket 01_people_portraits
```

### Restart from scratch (delete existing output first)

```bash
python pipeline.py --input Original-indian_cultural --output upcaled-indian_cultural --weights weights --full-run --bucket 01_people_portraits --reset-output
```

### Optional flags

- `--fp32`: force fp32 inference.
- `--tile 512`: pre-enable tiled inference.
- `--tile-pad 10`: tile overlap.
- `--qa-sample-size 25`: random samples per bucket for manual QA.
- `--bucket <name>`: run only one bucket.
- `--reset-output`: remove previous output directory before running.

## Output Layout

```text
upcaled-indian_cultural/
	01_people_portraits/
		upscaled_512/
		original_512/
	...
	12_abstract_texture/
	quality_report/
		01_people_portraits.csv
		...
		summary.csv
		visual_qa_samples.csv
	rejected/
	upscaling_notes.md
```

## Bucket Rules Implemented

- `01_people_portraits`: `RealESRGAN_x4plus`; face-enhance flag when small faces are detected.
- `02_clothing_textiles`: `RealESRGAN_x4plus`, forced `scale=2`, all upscaled outputs flagged for manual QA.
- `03` to `10`: `RealESRGAN_x4plus` default flow.
- `11_art_design`: `RealESRGAN_x4plus_anime_6B` only.
- `12_abstract_texture`: `RealESRGAN_x4plus` default flow with QA emphasis.
- Very small images (`short-side < 128`): switch to `RealESRGAN_x2plus` / scale 2.

## Reliability Features

- fp16 by default when CUDA is available.
- Automatic fp32 retry on NaN/Inf and color-shift checks.
- Automatic tiled fallback (`512`, then `256`) on OOM.
- Skip already processed files to support resumable runs.
- Logs to `pipeline.log` by default.
