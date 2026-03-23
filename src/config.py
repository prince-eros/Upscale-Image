from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_X4 = "RealESRGAN_x4plus"
MODEL_X4_ANIME = "RealESRGAN_x4plus_anime_6B"
MODEL_X2 = "RealESRGAN_x2plus"

MODEL_FILES = {
    MODEL_X4: "RealESRGAN_x4plus.pth",
    MODEL_X4_ANIME: "RealESRGAN_x4plus_anime_6B.pth",
    MODEL_X2: "RealESRGAN_x2plus.pth",
}

BUCKET_MODELS = {
    "01_people_portraits": MODEL_X4,
    "02_clothing_textiles": MODEL_X4,
    "03_architecture": MODEL_X4,
    "04_landscape_nature": MODEL_X4,
    "05_urban_street": MODEL_X4,
    "06_rural_village": MODEL_X4,
    "07_food_drink": MODEL_X4,
    "08_festivals_rituals": MODEL_X4,
    "09_objects_artifacts": MODEL_X4,
    "10_animals_wildlife": MODEL_X4,
    "11_art_design": MODEL_X4_ANIME,
    "12_abstract_texture": MODEL_X4,
}

PORTRAITS_BUCKET = "01_people_portraits"
TEXTILES_BUCKET = "02_clothing_textiles"
ART_BUCKET = "11_art_design"
MIN_SHORT_SIDE = 512
VERY_SMALL_THRESHOLD = 128
LPIPS_EXCELLENT = 0.20
LPIPS_GOOD = 0.35
LPIPS_FLAG = 0.50


@dataclass(frozen=True)
class PipelineConfig:
    input_root: Path
    output_root: Path
    weights_dir: Path
    bucket: str | None = None
    reset_output: bool = False
    fp32: bool = False
    tile: int = 0
    tile_pad: int = 10
    seed: int = 42
    test_limit: int | None = 10
    full_run: bool = False
    qa_sample_size: int = 25

    @property
    def quality_report_dir(self) -> Path:
        return self.output_root / "quality_report"

    @property
    def rejected_dir(self) -> Path:
        return self.output_root / "rejected"
