from __future__ import annotations

import cv2
import lpips
import numpy as np
import torch

from .config import LPIPS_EXCELLENT, LPIPS_FLAG, LPIPS_GOOD


class LPIPSScorer:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = lpips.LPIPS(net="alex").to(self.device)
        self.metric.eval()

    def score(self, original_bgr: np.ndarray, upscaled_bgr: np.ndarray) -> float:
        h, w = original_bgr.shape[:2]
        restored = cv2.resize(upscaled_bgr, (w, h), interpolation=cv2.INTER_AREA)

        t1 = self._to_tensor(original_bgr)
        t2 = self._to_tensor(restored)
        with torch.no_grad():
            value = self.metric(t1, t2)
        return float(value.item())

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        arr = image_rgb.astype(np.float32) / 255.0
        arr = (arr * 2.0) - 1.0
        chw = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        return tensor


def status_from_lpips(score: float) -> str:
    if score < LPIPS_EXCELLENT:
        return "kept"
    if LPIPS_EXCELLENT <= score < LPIPS_GOOD:
        return "kept"
    if LPIPS_GOOD <= score <= LPIPS_FLAG:
        return "flagged"
    return "rejected"
