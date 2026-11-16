# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

"""
DaVinci Resolve 스타일 프라이머리(Lift / Gamma / Gain + Contrast / Pivot) 추정.

Target 이미지의 밝기 분포와 중성색 밸런스를 Reference 에 맞추는
프라이머리 값을 최적화로 찾는다.

적용 순서 (이 모듈의 정의)
-------------------------
    1. Contrast / Pivot   : y = (x - pivot) * contrast + pivot     (RGB 공통)
    2. Lift   (RGB 채널별) : y = y + lift * (1 - y)                 (흰색 고정, 검정 이동)
    3. Gain   (RGB 채널별) : y = y * gain                           (검정 고정, 흰색 이동)
    4. Gamma  (RGB 채널별) : y = y ** (1 / gamma)                   (1보다 크면 중간톤 밝게)

주의: DaVinci Resolve 내부 수식은 공개되어 있지 않으므로
여기서 구한 값은 '표준 LGG 공식 기준 근사값'이다.
Resolve 휠에 옮겨 넣을 때는 시작점으로 쓰고 눈으로 미세 조정하는 것을 권장한다.
"""

import numpy as np
from scipy.optimize import least_squares

from .image_io import validate_rgb_image


# Rec.709 루마 계수 (Resolve 의 루마 계산과 같은 기준)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

LUMA_QUANTILES = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)

NEUTRAL_PRIMARIES = {
    "contrast": 1.0,
    "pivot": 0.5,
    "lift": [0.0, 0.0, 0.0],
    "gamma": [1.0, 1.0, 1.0],
    "gain": [1.0, 1.0, 1.0],
}


def apply_primaries(pixels, primaries):
    """
    (N, 3) 0~1 float RGB 에 프라이머리를 적용한다.
    """

    x = np.asarray(pixels, dtype=np.float64)

    contrast = float(primaries["contrast"])
    pivot = float(primaries["pivot"])
    lift = np.asarray(primaries["lift"], dtype=np.float64)
    gamma = np.asarray(primaries["gamma"], dtype=np.float64)
    gain = np.asarray(primaries["gain"], dtype=np.float64)

    y = (x - pivot) * contrast + pivot
    y = np.clip(y, 0.0, 1.0)

    y = y + lift * (1.0 - y)
    y = y * gain
    y = np.clip(y, 0.0, 1.0)

    y = y ** (1.0 / gamma)

    return np.clip(y, 0.0, 1.0)


def apply_primaries_to_image(rgb_image, primaries):
    """
    (H, W, 3) uint8 RGB 이미지에 프라이머리를 적용한다.
    """

    validate_rgb_image(rgb_image)

    height, width = rgb_image.shape[:2]

    pixels = rgb_image.reshape(-1, 3).astype(np.float64) / 255.0
    output = apply_primaries(pixels, primaries)

    return (
        (output * 255.0).round().astype(np.uint8).reshape(height, width, 3)
    )


def _luma(pixels):
    return pixels @ LUMA_WEIGHTS


def _sample(rgb_image, sample_count, random_generator):
    pixels = rgb_image.reshape(-1, 3)
    count = min(sample_count, len(pixels))
    indices = random_generator.choice(len(pixels), size=count, replace=False)
    return pixels[indices].astype(np.float64) / 255.0


def _neutral_band_indices(pixels, luma, neutral_percentile=20.0):
    """
    섀도우 / 미드 / 하이라이트 각 밴드에서
    채도가 낮은(중성색에 가까운) 픽셀 인덱스를 고른다.

    밸런스는 '원래 회색이어야 할 것'의 색으로 판단해야
    빨간 드레스 같은 피사체 색에 끌려가지 않는다.
    """

    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    low, high = np.percentile(luma, [33.3, 66.7])

    bands = [
        luma <= low,
        (luma > low) & (luma < high),
        luma >= high,
    ]

    selected = []

    for band in bands:
        band_indices = np.flatnonzero(band)

        if len(band_indices) < 10:
            selected.append(band_indices)
            continue

        threshold = np.percentile(chroma[band_indices], neutral_percentile)
        selected.append(band_indices[chroma[band_indices] <= threshold])

    return selected


def _neutral_offsets(pixels, band_indices):
    """
    밴드별 중성 픽셀의 (R-Y, G-Y, B-Y) 중앙값. 형태 (3밴드, 3채널)
    """

    luma = _luma(pixels)
    offsets = []

    for indices in band_indices:
        if len(indices) == 0:
            offsets.append(np.zeros(3))
            continue

        offsets.append(
            np.median(pixels[indices] - luma[indices, None], axis=0)
        )

    return np.array(offsets)


def _quantile_anchor_indices(luma, quantiles, window=0.004):
    """
    각 분위수 근처 순위의 픽셀 인덱스 묶음.

    분위수를 매번 다시 정렬해서 계산하면 최적화 함수가 계단 모양이 되어
    기울기를 구할 수 없으므로, 처음에 정한 픽셀 묶음을 계속 추적한다.
    """

    order = np.argsort(luma)
    count = len(order)
    half_window = max(int(count * window), 5)

    anchors = []

    for quantile in quantiles:
        center = int(round(quantile / 100.0 * (count - 1)))
        start = max(center - half_window, 0)
        stop = min(center + half_window + 1, count)
        anchors.append(order[start:stop])

    return anchors


def _pack(primaries):
    return np.concatenate([
        [primaries["contrast"]],
        primaries["lift"],
        primaries["gamma"],
        primaries["gain"],
    ]).astype(np.float64)


def _unpack(vector, pivot):
    return {
        "contrast": float(vector[0]),
        "pivot": float(pivot),
        "lift": [float(v) for v in vector[1:4]],
        "gamma": [float(v) for v in vector[4:7]],
        "gain": [float(v) for v in vector[7:10]],
    }


def estimate_primaries(
    target_rgb,
    reference_rgb,
    sample_count=60_000,
    balance_strength=0.5,
    regularization=0.05,
    channel_regularization=0.4,
    random_seed=42,
):
    """
    Target -> Reference 프라이머리 값을 추정한다.

    맞추는 항목
    ----------
    1. 루마 분위수 (1, 5, 10, 25, 50, 75, 90, 95, 99%)
       -> 블랙 레벨, 대비, 중간톤 밝기, 화이트 레벨
    2. 섀도우 / 미드 / 하이라이트의 '중성색' 밸런스
       -> RGB 채널별 lift / gamma / gain 차이 (색온도/틴트)

    Parameters
    ----------
    balance_strength : 0~1
        밸런스를 Reference 쪽으로 얼마나 옮길지.
        0 이면 Target 의 밸런스를 유지하고 밝기/대비만 맞춘다.

    regularization : float
        값이 기본값(변화 없음)에서 과하게 멀어지지 않도록 하는 벌점.

    channel_regularization : float
        R / G / B 값이 서로 크게 벌어지지 않도록 하는 벌점.
        (밸런스는 프라이머리에서 가볍게, 룩의 색감은 다음 색 단계에서)

    Returns
    -------
    dict
        primaries : 추정된 프라이머리 값
        diagnostics : 맞추기 전/후 루마 분위수 등
    """

    validate_rgb_image(target_rgb)
    validate_rgb_image(reference_rgb)

    random_generator = np.random.default_rng(random_seed)

    target_pixels = _sample(target_rgb, sample_count, random_generator)
    reference_pixels = _sample(reference_rgb, sample_count, random_generator)

    target_luma = _luma(target_pixels)
    reference_luma = _luma(reference_pixels)

    # Pivot 은 Target 의 중간 밝기로 고정한다.
    # (중간톤 밝기는 고정한 채 대비만 바뀌도록)
    pivot = float(np.median(target_luma))

    reference_quantiles = np.percentile(reference_luma, LUMA_QUANTILES)
    anchors = _quantile_anchor_indices(target_luma, LUMA_QUANTILES)

    target_bands = _neutral_band_indices(target_pixels, target_luma)
    reference_bands = _neutral_band_indices(reference_pixels, reference_luma)

    target_offsets = _neutral_offsets(target_pixels, target_bands)
    reference_offsets = _neutral_offsets(reference_pixels, reference_bands)

    strength = float(np.clip(balance_strength, 0.0, 1.0))
    goal_offsets = (
        target_offsets + (reference_offsets - target_offsets) * strength
    )

    neutral_vector = _pack(NEUTRAL_PRIMARIES)

    def residuals(vector):
        primaries = _unpack(vector, pivot)
        output = apply_primaries(target_pixels, primaries)
        output_luma = _luma(output)

        tone_residual = np.array([
            output_luma[anchor].mean() for anchor in anchors
        ]) - reference_quantiles

        balance_residual = (
            _neutral_offsets(output, target_bands) - goal_offsets
        ).reshape(-1)

        regularization_residual = (
            (vector - neutral_vector) * regularization
        )

        lift_values = vector[1:4]
        log_gamma = np.log(vector[4:7])
        log_gain = np.log(vector[7:10])

        channel_residual = np.concatenate([
            lift_values - lift_values.mean(),
            log_gamma - log_gamma.mean(),
            log_gain - log_gain.mean(),
        ]) * channel_regularization

        return np.concatenate([
            tone_residual * 1.0,
            balance_residual * 2.0,
            regularization_residual,
            channel_residual,
        ])

    lower = np.array([0.5] + [-0.3] * 3 + [0.4] * 3 + [0.4] * 3)
    upper = np.array([2.0] + [0.3] * 3 + [2.5] * 3 + [2.5] * 3)

    solution = least_squares(
        residuals,
        neutral_vector,
        bounds=(lower, upper),
        diff_step=1e-4,
        max_nfev=300,
    )

    primaries = _unpack(solution.x, pivot)

    output = apply_primaries(target_pixels, primaries)

    diagnostics = {
        "luma_quantile_levels": LUMA_QUANTILES.tolist(),
        "reference_luma_quantiles": reference_quantiles.tolist(),
        "target_luma_quantiles_before": np.percentile(
            target_luma, LUMA_QUANTILES
        ).tolist(),
        "target_luma_quantiles_after": np.percentile(
            _luma(output), LUMA_QUANTILES
        ).tolist(),
        "reference_neutral_offsets": reference_offsets.tolist(),
        "target_neutral_offsets_before": target_offsets.tolist(),
        "target_neutral_offsets_after": _neutral_offsets(
            output, target_bands
        ).tolist(),
        "optimizer_cost": float(solution.cost),
        "optimizer_success": bool(solution.success),
    }

    return {
        "primaries": primaries,
        "diagnostics": diagnostics,
    }


def format_primaries_table(primaries):
    """
    UI / 로그 표시용 표 데이터.
    """

    return {
        "항목": ["Lift", "Gamma (지수)", "Gain"],
        "R": [
            primaries["lift"][0],
            primaries["gamma"][0],
            primaries["gain"][0],
        ],
        "G": [
            primaries["lift"][1],
            primaries["gamma"][1],
            primaries["gain"][1],
        ],
        "B": [
            primaries["lift"][2],
            primaries["gamma"][2],
            primaries["gain"][2],
        ],
        "Master(평균)": [
            float(np.mean(primaries["lift"])),
            float(np.mean(primaries["gamma"])),
            float(np.mean(primaries["gain"])),
        ],
    }
