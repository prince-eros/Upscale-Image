from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

try:
    from gfpgan import GFPGANer
except Exception:  # noqa: BLE001
    GFPGANer = None

from .config import MODEL_FILES, MODEL_X2, MODEL_X4, MODEL_X4_ANIME


@dataclass
class UpscaleResult:
    image: np.ndarray
    model_used: str
    scale_used: int
    face_enhance_applied: bool
    used_fp32_retry: bool
    used_tiling_fallback: bool


class RealESRGANUpscaler:
    def __init__(
        self,
        weights_dir: Path,
        fp32: bool = False,
        tile: int = 0,
        tile_pad: int = 10,
    ) -> None:
        self.weights_dir = weights_dir
        self.default_fp32 = fp32
        self.default_tile = tile
        self.tile_pad = tile_pad
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache: dict[tuple[str, bool, int], RealESRGANer] = {}
        self._face_cache: dict[tuple[str, bool, int], Any] = {}
        self._gfpgan_weight = self._resolve_gfpgan_weight()

    def _resolve_gfpgan_weight(self) -> Path | None:
        candidates = [
            "GFPGANv1.4.pth",
            "GFPGANv1.3.pth",
            "GFPGANv1.2.pth",
        ]
        for name in candidates:
            p = self.weights_dir / name
            if p.exists():
                return p
        return None

    def validate_weights(self) -> None:
        missing = []
        hints: list[str] = []
        for model_key, filename in MODEL_FILES.items():
            model_path = self.weights_dir / filename
            if not model_path.exists():
                missing.append(f"{model_key}: {model_path}")

                # Common user mistake: ESRNet file downloaded instead of ESRGAN x4plus.
                if model_key == MODEL_X4:
                    esrnet_path = self.weights_dir / "RealESRNet_x4plus.pth"
                    if esrnet_path.exists():
                        hints.append(
                            "Found RealESRNet_x4plus.pth, but pipeline requires "
                            "RealESRGAN_x4plus.pth from official Real-ESRGAN releases."
                        )
        if missing:
            missing_text = "\n".join(missing)
            hint_text = ""
            if hints:
                hint_text = "\n\nHints:\n" + "\n".join(hints)
            raise FileNotFoundError(f"Missing Real-ESRGAN weights:\n{missing_text}{hint_text}")

    def upscale(
        self,
        image_bgr: np.ndarray,
        model_name: str,
        requested_scale: int,
        face_enhance: bool,
    ) -> UpscaleResult:
        use_fp32_retry = False
        used_tiling_fallback = False

        fp32_modes = [self.default_fp32]
        if not self.default_fp32:
            fp32_modes.append(True)

        tile_candidates = [self.default_tile]
        for fallback_tile in (512, 256):
            if fallback_tile not in tile_candidates:
                tile_candidates.append(fallback_tile)

        last_error: Exception | None = None
        for use_fp32 in fp32_modes:
            for tile in tile_candidates:
                try:
                    upsampler = self._get_or_create_upsampler(
                        model_name=model_name,
                        fp32=use_fp32,
                        tile=tile,
                    )
                    # First pass: texture/detail upscaling with RealESRGAN.
                    output, _ = upsampler.enhance(image_bgr, outscale=requested_scale)
                    # Second pass: optional face restoration on the already upscaled image.
                    if face_enhance:
                        output = self._enhance_faces(
                            image_bgr=output,
                            model_name=model_name,
                            fp32=use_fp32,
                            tile=tile,
                        )
                    if np.isnan(output).any() or np.isinf(output).any():
                        raise RuntimeError("NaN/Inf detected in output")

                    if use_fp32:
                        use_fp32_retry = use_fp32_retry or (not self.default_fp32)
                    if tile != self.default_tile:
                        used_tiling_fallback = True

                    return UpscaleResult(
                        image=output,
                        model_used=model_name,
                        scale_used=requested_scale,
                        face_enhance_applied=face_enhance,
                        used_fp32_retry=use_fp32_retry,
                        used_tiling_fallback=used_tiling_fallback,
                    )
                except RuntimeError as err:
                    last_error = err
                    msg = str(err).lower()
                    oom = "out of memory" in msg or "cuda error" in msg
                    nan_issue = "nan" in msg or "inf" in msg
                    if oom or nan_issue:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Upscaling failed with unknown error")

    def _get_or_create_upsampler(self, model_name: str, fp32: bool, tile: int) -> RealESRGANer:
        key = (model_name, fp32, tile)
        if key in self._cache:
            return self._cache[key]

        model, netscale = self._build_arch(model_name)
        model_path = self.weights_dir / MODEL_FILES[model_name]
        half = (not fp32) and self.device == "cuda"

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            model=model,
            tile=tile,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=half,
            gpu_id=0 if self.device == "cuda" else None,
        )
        self._cache[key] = upsampler
        return upsampler

    @staticmethod
    def _build_arch(model_name: str) -> tuple[RRDBNet, int]:
        if model_name == MODEL_X4:
            return (
                RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                4,
            )
        if model_name == MODEL_X4_ANIME:
            return (
                RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                4,
            )
        if model_name == MODEL_X2:
            return (
                RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                2,
            )
        raise ValueError(f"Unsupported model: {model_name}")

    def _enhance_faces(
        self,
        image_bgr: np.ndarray,
        model_name: str,
        fp32: bool,
        tile: int,
    ) -> np.ndarray:
        if GFPGANer is None or self._gfpgan_weight is None:
            return image_bgr

        face_enhancer = self._get_or_create_face_enhancer(
            model_name=model_name,
            fp32=fp32,
            tile=tile,
        )
        _cropped, _restored, output = face_enhancer.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        return output

    def _get_or_create_face_enhancer(
        self,
        model_name: str,
        fp32: bool,
        tile: int,
    ) -> Any:
        key = (model_name, fp32, tile)
        if key in self._face_cache:
            return self._face_cache[key]

        face_enhancer = GFPGANer(
            model_path=str(self._gfpgan_weight),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        self._face_cache[key] = face_enhancer
        return face_enhancer


def detect_color_shift(original_bgr: np.ndarray, upscaled_bgr: np.ndarray) -> bool:
    h, w = original_bgr.shape[:2]
    resized = cv2.resize(upscaled_bgr, (w, h), interpolation=cv2.INTER_AREA)
    o_mean = np.mean(original_bgr.astype(np.float32), axis=(0, 1))
    r_mean = np.mean(resized.astype(np.float32), axis=(0, 1))
    drift = np.abs(o_mean - r_mean)
    return bool(np.any(drift > 35.0))
