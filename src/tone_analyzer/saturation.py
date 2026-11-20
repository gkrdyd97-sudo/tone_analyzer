# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

"""
③ 채도 (Saturation)

밝기(①)와 색 균형(②)을 맞춘 뒤, 벡터스코프의 퍼짐을 Reference 에 맞춘다.
DaVinci 의 Saturation 처럼 루마는 그대로 두고 색 성분만 키우거나 줄인다.

    y = luma + saturation * (rgb - luma)

기준
----
- 두 이미지 모두 얼굴이 있으면: **피부 채도 비율** (source = "skin")
  피부는 장면이 달라도 비교할 수 있는 공통 기준이라 화면 내용 차이에 강하다.
  (네온 간판이 많은 장면과 회색 벽 장면을 통째로 비교하면 채도가 엉뚱하게 움직인다)
- 아니면: 미드톤 픽셀의 채도 상위 25% 지점 비율 (source = "scene")

안전 범위 0.8 ~ 1.25. 벗어나면 원래 값(1) 쪽으로 줄인다 (primaries.limit_with_distrust).
"""

import numpy as np

from .primaries import LUMA_WEIGHTS, limit_with_distrust
from .skin_tone import srgb_to_lab


SAFE_RANGE = (0.8, 1.25)
WIDE_RANGE = (0.4, 2.5)


def apply_saturation(pixels, saturation):
    pixels = np.asarray(pixels, dtype=np.float64)
    luma = (pixels @ LUMA_WEIGHTS)[:, None]
    return np.clip(luma + float(saturation) * (pixels - luma), 0.0, 1.0)


def _chroma(pixels):
    lab = srgb_to_lab(np.asarray(pixels, dtype=np.float64))
    return np.hypot(lab[:, 1], lab[:, 2]), lab[:, 0]


def _scene_chroma(rgb_image, sample_count, random_generator):
    pixels = rgb_image.reshape(-1, 3)
    if len(pixels) > sample_count:
        pixels = pixels[random_generator.choice(len(pixels), sample_count, replace=False)]
    chroma, lightness = _chroma(pixels.astype(np.float64) / 255.0)
    midtones = (lightness > 20.0) & (lightness < 85.0)
    if midtones.sum() < 100:
        midtones = np.ones_like(midtones)
    return float(np.percentile(chroma[midtones], 75.0))


def estimate_saturation(
    target_rgb,
    reference_rgb,
    target_skin_samples=None,
    reference_skin_samples=None,
    safe_limits=True,
    strength=1.0,
    sample_count=60_000,
    random_seed=42,
):
    """
    Returns
    -------
    dict
        saturation  : 적용할 채도 배율 (안전 범위·강도 적용 후)
        diagnostics : source, 측정값, 넓은 범위 값, 제한 여부
    """

    random_generator = np.random.default_rng(random_seed)

    use_skin = (
        target_skin_samples is not None and len(target_skin_samples) >= 50
        and reference_skin_samples is not None and len(reference_skin_samples) >= 50
    )

    if use_skin:
        target_measure = float(np.median(_chroma(target_skin_samples)[0]))
        reference_measure = float(np.median(_chroma(reference_skin_samples)[0]))
        source = "skin"
    else:
        target_measure = _scene_chroma(target_rgb, sample_count, random_generator)
        reference_measure = _scene_chroma(reference_rgb, sample_count, random_generator)
        source = "scene"

    raw = float(np.clip(reference_measure / max(target_measure, 1e-3), *WIDE_RANGE))

    if target_measure < 2.0:
        # 거의 흑백 → 배율이 의미 없음
        raw = 1.0

    saturation, limited = limit_with_distrust(raw, *SAFE_RANGE) if safe_limits else (raw, False)
    saturation = float(saturation ** float(np.clip(strength, 0.0, 1.0)))

    return {
        "saturation": saturation,
        "diagnostics": {
            "source": source,
            "target_measure": target_measure,
            "reference_measure": reference_measure,
            "raw_saturation": raw,
            "limited": bool(limited),
        },
    }
