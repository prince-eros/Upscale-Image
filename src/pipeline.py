from __future__ import annotations

import argparse
import csv
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import (
    ART_BUCKET,
    BUCKET_MODELS,
    MIN_SHORT_SIDE,
    MODEL_X2,
    MODEL_X4,
    MODEL_X4_ANIME,
    PORTRAITS_BUCKET,
    TEXTILES_BUCKET,
    VERY_SMALL_THRESHOLD,
    PipelineConfig,
)
from .quality import LPIPSScorer, status_from_lpips
from .upscaler import RealESRGANUpscaler, detect_color_shift
from .utils import (
    build_stable_output_stem,
    copy_image_and_text,
    copy_text_pair,
    ensure_dir,
    estimate_small_face_present,
    image_resolution_str,
    iter_bucket_images,
    read_image_bgr,
    sample_items,
    set_reproducible_seed,
    short_side,
    write_image_bgr,
)

LOGGER = logging.getLogger("realesrgan_pipeline")


@dataclass
class Record:
    image_name: str
    original_resolution: str
    upscaled_resolution: str
    lpips_score: str
    status: str
    model_used: str
    scale_used: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-grade Real-ESRGAN upscaling pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset root")
    parser.add_argument("--weights", type=Path, default=Path("weights"), help="Directory containing model .pth files")
    parser.add_argument(
        "--bucket",
        type=str,
        choices=sorted(BUCKET_MODELS.keys()),
        help="Run only one specific bucket (for example: 01_people_portraits)",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete existing output directory before starting the run",
    )
    parser.add_argument("--fp32", action="store_true", help="Force fp32 inference")
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0 disables pre-tiling)")
    parser.add_argument("--tile-pad", type=int, default=10, help="Tile overlap padding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument(
        "--test-limit",
        type=int,
        default=10,
        help="Maximum images to process for test run. Ignored when --full-run is used.",
    )
    parser.add_argument("--full-run", action="store_true", help="Process full dataset")
    parser.add_argument("--qa-sample-size", type=int, default=25, help="Visual QA sample count per bucket")
    parser.add_argument("--log-file", type=Path, default=Path("pipeline.log"), help="Path to pipeline log file")
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_root=args.input,
        output_root=args.output,
        weights_dir=args.weights,
        bucket=args.bucket,
        reset_output=args.reset_output,
        fp32=args.fp32,
        tile=args.tile,
        tile_pad=args.tile_pad,
        seed=args.seed,
        test_limit=None if args.full_run else args.test_limit,
        full_run=args.full_run,
        qa_sample_size=args.qa_sample_size,
    )


def find_bucket_dirs(input_root: Path) -> list[Path]:
    dirs = [p for p in input_root.iterdir() if p.is_dir() and p.name in BUCKET_MODELS]
    dirs.sort(key=lambda p: p.name)
    return dirs


def filter_bucket_dirs(bucket_dirs: list[Path], selected_bucket: str | None) -> list[Path]:
    if selected_bucket is None:
        return bucket_dirs
    return [p for p in bucket_dirs if p.name == selected_bucket]


def write_bucket_csv(csv_path: Path, rows: list[Record]) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "original_resolution",
                "upscaled_resolution",
                "lpips_score",
                "status",
                "model_used",
                "scale_used",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary_csv(summary_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    ensure_dir(summary_path.parent)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "total",
                "upscaled",
                "copied",
                "flagged",
                "rejected",
                "errors",
                "avg_lpips",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def write_visual_qa_samples(path: Path, samples: dict[str, list[str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "sample_path"])
        writer.writeheader()
        for bucket, bucket_samples in samples.items():
            for sample in bucket_samples:
                writer.writerow({"bucket": bucket, "sample_path": sample})


def generate_notes(
    path: Path,
    summary_rows: list[dict[str, Any]],
    below_128_count: int,
    anime_stats: dict[str, float],
) -> None:
    total = sum(int(r["total"]) for r in summary_rows)
    total_upscaled = sum(int(r["upscaled"]) for r in summary_rows)
    total_rejected = sum(int(r["rejected"]) for r in summary_rows)
    success_rate = 0.0 if total == 0 else ((total - total_rejected) / total) * 100.0

    if summary_rows:
        highest_artifacts = sorted(summary_rows, key=lambda r: int(r["flagged"]) + int(r["rejected"]), reverse=True)
        artifact_text = ", ".join(r["bucket"] for r in highest_artifacts[:3])
    else:
        artifact_text = "N/A"

    avg_lpips_lines = []
    for row in summary_rows:
        avg_lpips_lines.append(f"- {row['bucket']}: {row['avg_lpips']}")

    anime_improvement = "Insufficient data"
    if anime_stats.get("count", 0) > 0:
        anime_improvement = (
            f"Anime model applied on {int(anime_stats['count'])} upscaled images; "
            f"avg LPIPS={anime_stats['avg_lpips']:.4f}."
        )

    content = "\n".join(
        [
            "# Upscaling Notes",
            "",
            "## Observations",
            f"- Buckets with most artifacts: {artifact_text}",
            "- Issues in clothing_textiles: Review flagged samples for texture hallucination and pattern repetition.",
            "- Issues in architecture: Review flagged samples for line bending and geometric distortion.",
            "",
            "## Model Evaluation",
            f"- Did anime model improve art_design? {anime_improvement}",
            "",
            "## Edge Cases",
            f"- Images below 128px: {below_128_count}",
            "",
            "## Statistics",
            f"- Total images processed: {total}",
            f"- Total upscaled: {total_upscaled}",
            f"- Total rejected: {total_rejected}",
            f"- Success rate: {success_rate:.2f}%",
            "- Avg LPIPS per bucket:",
            *avg_lpips_lines,
        ]
    )
    ensure_dir(path.parent)
    path.write_text(content + "\n", encoding="utf-8")


def choose_model_and_scale(bucket: str, image_short_side: int) -> tuple[str, int, str]:
    if bucket == ART_BUCKET:
        default_model = MODEL_X4_ANIME
    else:
        default_model = BUCKET_MODELS.get(bucket, MODEL_X4)

    if bucket == TEXTILES_BUCKET:
        return default_model, 2, "textiles_scale2"

    if image_short_side < VERY_SMALL_THRESHOLD:
        return MODEL_X2, 2, "very_small_scale2"

    return default_model, 4, "default"


def safe_move_to_rejected(src_img: Path, rejected_img: Path) -> None:
    ensure_dir(rejected_img.parent)
    if src_img.exists():
        shutil.move(str(src_img), str(rejected_img))
    src_txt = src_img.with_suffix(".txt")
    rejected_txt = rejected_img.with_suffix(".txt")
    if src_txt.exists():
        shutil.move(str(src_txt), str(rejected_txt))


def process_bucket(
    bucket_dir: Path,
    config: PipelineConfig,
    upscaler: RealESRGANUpscaler,
    scorer: LPIPSScorer,
    global_limit_state: dict[str, int | None],
) -> tuple[list[Record], dict[str, Any], list[str], int, list[float]]:
    bucket = bucket_dir.name
    output_bucket = config.output_root / bucket
    original_dir = output_bucket / "original_512"
    upscaled_dir = output_bucket / "upscaled_512"
    rejected_bucket = config.rejected_dir / bucket
    ensure_dir(original_dir)
    ensure_dir(upscaled_dir)
    ensure_dir(rejected_bucket)

    rows: list[Record] = []
    processed_paths_for_qa: list[str] = []
    below_128_count = 0
    bucket_lpips: list[float] = []

    images = iter_bucket_images(bucket_dir)
    for image_path in tqdm(images, desc=bucket, leave=False):
        remaining = global_limit_state.get("remaining")
        if remaining is not None and remaining <= 0:
            break

        try:
            image = read_image_bgr(image_path)
        except Exception as err:  # noqa: BLE001
            LOGGER.exception("Corrupted or unreadable image: %s", image_path)
            rows.append(
                Record(
                    image_name=str(image_path.name),
                    original_resolution="unknown",
                    upscaled_resolution="",
                    lpips_score="",
                    status=f"error:{type(err).__name__}",
                    model_used="",
                    scale_used="",
                )
            )
            if remaining is not None:
                global_limit_state["remaining"] = int(remaining) - 1
            continue

        stem = build_stable_output_stem(image_path, bucket_dir)
        ext = image_path.suffix.lower()
        img_short = short_side(image)

        original_res = image_resolution_str(image)
        if img_short >= MIN_SHORT_SIDE:
            dst = original_dir / f"{stem}{ext}"
            if dst.exists():
                status = "skipped_existing"
            else:
                copy_image_and_text(image_path, dst)
                status = "kept"
            rows.append(
                Record(
                    image_name=f"{stem}{ext}",
                    original_resolution=original_res,
                    upscaled_resolution=original_res,
                    lpips_score="",
                    status=status,
                    model_used="none",
                    scale_used="1",
                )
            )
            processed_paths_for_qa.append(str(dst.relative_to(config.output_root)))
            if remaining is not None:
                global_limit_state["remaining"] = int(remaining) - 1
            continue

        if img_short < VERY_SMALL_THRESHOLD:
            below_128_count += 1

        model_name, scale, reason = choose_model_and_scale(bucket, img_short)
        face_enhance = False
        if bucket == PORTRAITS_BUCKET and img_short < MIN_SHORT_SIDE:
            face_enhance = estimate_small_face_present(image, threshold_px=128)

        upscaled_target = upscaled_dir / f"{stem}{ext}"
        rejected_target = rejected_bucket / f"{stem}{ext}"
        if upscaled_target.exists() or rejected_target.exists():
            rows.append(
                Record(
                    image_name=f"{stem}{ext}",
                    original_resolution=original_res,
                    upscaled_resolution="existing",
                    lpips_score="",
                    status="skipped_existing",
                    model_used=model_name,
                    scale_used=str(scale),
                )
            )
            if remaining is not None:
                global_limit_state["remaining"] = int(remaining) - 1
            continue

        try:
            result = upscaler.upscale(
                image_bgr=image,
                model_name=model_name,
                requested_scale=scale,
                face_enhance=face_enhance,
            )
            if (not config.fp32) and detect_color_shift(image, result.image):
                # Retry in fp32 only when visible color drift is detected.
                retry = RealESRGANUpscaler(
                    weights_dir=config.weights_dir,
                    fp32=True,
                    tile=config.tile,
                    tile_pad=config.tile_pad,
                )
                result = retry.upscale(
                    image_bgr=image,
                    model_name=model_name,
                    requested_scale=scale,
                    face_enhance=face_enhance,
                )

            write_image_bgr(upscaled_target, result.image)
            copy_text_pair(image_path, upscaled_target)

            lpips_score = scorer.score(image, result.image)
            bucket_lpips.append(lpips_score)
            status = status_from_lpips(lpips_score)

            if bucket == TEXTILES_BUCKET and status == "kept":
                status = "flagged"

            if status == "rejected":
                safe_move_to_rejected(upscaled_target, rejected_target)
                processed_paths_for_qa.append(str(rejected_target.relative_to(config.output_root)))
                final_res = f"moved_to_rejected:{image_resolution_str(result.image)}"
            else:
                processed_paths_for_qa.append(str(upscaled_target.relative_to(config.output_root)))
                final_res = image_resolution_str(result.image)

            model_field = result.model_used
            if face_enhance:
                model_field += "+face_enhance"
            if result.used_fp32_retry:
                model_field += "+fp32_retry"
            if result.used_tiling_fallback:
                model_field += "+tile_fallback"
            if reason == "very_small_scale2":
                model_field += "+very_small"

            rows.append(
                Record(
                    image_name=f"{stem}{ext}",
                    original_resolution=original_res,
                    upscaled_resolution=final_res,
                    lpips_score=f"{lpips_score:.6f}",
                    status=status,
                    model_used=model_field,
                    scale_used=str(result.scale_used),
                )
            )
        except Exception as err:  # noqa: BLE001
            LOGGER.exception("Upscale failed: %s", image_path)
            rows.append(
                Record(
                    image_name=f"{stem}{ext}",
                    original_resolution=original_res,
                    upscaled_resolution="",
                    lpips_score="",
                    status=f"error:{type(err).__name__}",
                    model_used=model_name,
                    scale_used=str(scale),
                )
            )

        if remaining is not None:
            global_limit_state["remaining"] = int(remaining) - 1

    summary = {
        "bucket": bucket,
        "total": len(rows),
        "upscaled": sum(1 for r in rows if r.model_used not in {"", "none"}),
        "copied": sum(1 for r in rows if r.model_used == "none"),
        "flagged": sum(1 for r in rows if r.status == "flagged"),
        "rejected": sum(1 for r in rows if r.status == "rejected"),
        "errors": sum(1 for r in rows if r.status.startswith("error:")),
        "avg_lpips": f"{(sum(bucket_lpips) / len(bucket_lpips)):.6f}" if bucket_lpips else "",
    }
    return rows, summary, processed_paths_for_qa, below_128_count, bucket_lpips


def run_pipeline(config: PipelineConfig) -> int:
    if not config.input_root.exists():
        raise FileNotFoundError(f"Input dataset not found: {config.input_root}")

    if config.reset_output and config.output_root.exists():
        LOGGER.warning("Reset mode enabled; deleting output directory: %s", config.output_root)
        shutil.rmtree(config.output_root)

    ensure_dir(config.output_root)
    ensure_dir(config.quality_report_dir)
    ensure_dir(config.rejected_dir)

    set_reproducible_seed(config.seed)

    upscaler = RealESRGANUpscaler(
        weights_dir=config.weights_dir,
        fp32=config.fp32,
        tile=config.tile,
        tile_pad=config.tile_pad,
    )
    upscaler.validate_weights()

    scorer = LPIPSScorer()

    bucket_dirs = find_bucket_dirs(config.input_root)
    bucket_dirs = filter_bucket_dirs(bucket_dirs, config.bucket)
    if not bucket_dirs:
        if config.bucket:
            raise RuntimeError(f"Bucket not found in input root: {config.bucket}")
        raise RuntimeError("No valid bucket directories found in input root")

    global_limit_state: dict[str, int | None] = {"remaining": config.test_limit}
    summary_rows: list[dict[str, Any]] = []
    qa_samples: dict[str, list[str]] = {}
    below_128_total = 0
    art_lpips_values: list[float] = []

    for bucket_dir in bucket_dirs:
        remaining = global_limit_state.get("remaining")
        if remaining is not None and remaining <= 0:
            break

        rows, summary, qa_paths, below_128_count, bucket_lpips = process_bucket(
            bucket_dir=bucket_dir,
            config=config,
            upscaler=upscaler,
            scorer=scorer,
            global_limit_state=global_limit_state,
        )
        summary_rows.append(summary)
        below_128_total += below_128_count
        if bucket_dir.name == ART_BUCKET:
            art_lpips_values.extend(bucket_lpips)

        bucket_csv = config.quality_report_dir / f"{bucket_dir.name}.csv"
        write_bucket_csv(bucket_csv, rows)

        qa_samples[bucket_dir.name] = sample_items(qa_paths, config.qa_sample_size, seed=config.seed)

    summary_csv = config.quality_report_dir / "summary.csv"
    write_summary_csv(summary_csv, summary_rows)

    qa_manifest = config.quality_report_dir / "visual_qa_samples.csv"
    write_visual_qa_samples(qa_manifest, qa_samples)

    anime_stats = {
        "count": float(len(art_lpips_values)),
        "avg_lpips": (sum(art_lpips_values) / len(art_lpips_values)) if art_lpips_values else 0.0,
    }
    notes_path = config.output_root / "upscaling_notes.md"
    generate_notes(notes_path, summary_rows, below_128_total, anime_stats)

    LOGGER.info("Pipeline completed. Summary written to %s", summary_csv)
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    config = build_config(args)
    LOGGER.info("Running pipeline with config: %s", config)
    if config.test_limit is not None and not config.full_run:
        LOGGER.info("Test mode enabled with limit=%s images.", config.test_limit)
    else:
        LOGGER.info("Full run mode enabled.")

    if torch.cuda.is_available():
        LOGGER.info("CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        LOGGER.info("CUDA not available; running on CPU (slower).")

    if config.bucket:
        LOGGER.info("Single-bucket mode enabled: %s", config.bucket)

    return run_pipeline(config)
