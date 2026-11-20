# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

"""
① Lift / Gamma / Gain  →  ② Contrast / Pivot  추정.

컬러리스트의 작업 순서대로 '밝기 계조'만 다룬다.
(색 균형은 ③ 화이트밸런스 단계 white_balance.py 에서 따로 처리)

적용 순서와 수식 (값 범위 0~1, RGB 세 채널에 같은 값 적용 = 마스터 휠)
-------------------------------------------------------------------
    ① Lift   : y = x + lift * (1 - x)          흰색 고정, 검정 이동
       Gain  : y = y * gain                    검정 고정, 흰색 이동
       Gamma : y = y ** (1 / gamma)            지수. 1 = 변화 없음, < 1 이면 중간톤이 어두워짐
    ② Contrast / Pivot : y = (y - pivot) * contrast + pivot

주의: DaVinci Resolve 내부 수식은 공개되어 있지 않으므로
여기서 구한 값은 '표준 LGG 공식 기준 근사값'이다.
Resolve 에 옮겨 넣을 때는 시작점으로 쓰고 눈으로 미세 조정하는 것을 권장한다.
"""

import numpy as np
from scipy.optimize import least_squares

from .image_io import validate_rgb_image


# Rec.709 루마 계수 (Resolve 의 루마 계산과 같은 기준)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

LUMA_QUANTILES = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)

NEUTRAL_PRIMARIES = {
    "lift": 0.0,
    "gamma": 1.0,
    "gain": 1.0,
    "contrast": 1.0,
    "pivot": 0.435,
}


def apply_primaries(pixels, primaries):
    """
    (N, 3) 0~1 float RGB 에 ① LGG → ② Contrast/Pivot 을 적용한다.
    """

    x = np.asarray(pixels, dtype=np.float64)

    lift = float(primaries["lift"])
    gamma = float(primaries["gamma"])
    gain = float(primaries["gain"])
    contrast = float(primaries["contrast"])
    pivot = float(primaries["pivot"])

    # ① Lift / Gain / Gamma
    y = x + lift * (1.0 - x)
    y = np.clip(y * gain, 0.0, 1.0)
    y = y ** (1.0 / gamma)

    # ② Contrast / Pivot
    y = (y - pivot) * contrast + pivot

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


def luma(pixels):
    return np.asarray(pixels, dtype=np.float64) @ LUMA_WEIGHTS


def sample_pixels(rgb_image, sample_count, random_generator):
    pixels = rgb_image.reshape(-1, 3)
    count = min(sample_count, len(pixels))
    indices = random_generator.choice(len(pixels), size=count, replace=False)
    return pixels[indices].astype(np.float64) / 255.0


def _quantile_anchor_indices(luma_values, quantiles, window=0.004):
    """
    각 분위수 근처 순위의 픽셀 인덱스 묶음.

    분위수를 매번 다시 정렬해서 계산하면 최적화 함수가 계단 모양이 되어
    기울기를 구할 수 없으므로, 처음에 정한 픽셀 묶음을 계속 추적한다.
    """

    order = np.argsort(luma_values)
    count = len(order)
    half_window = max(int(count * window), 5)

    anchors = []

    for quantile in quantiles:
        center = int(round(quantile / 100.0 * (count - 1)))
        start = max(center - half_window, 0)
        stop = min(center + half_window + 1, count)
        anchors.append(order[start:stop])

    return anchors


PARAMETER_ORDER = ["lift", "gamma", "gain", "contrast"]


def _pack(primaries):
    return np.array(
        [primaries[name] for name in PARAMETER_ORDER],
        dtype=np.float64,
    )


def _unpack(vector, pivot):
    primaries = {
        name: float(value)
        for name, value in zip(PARAMETER_ORDER, vector)
    }
    primaries["pivot"] = float(pivot)
    return primaries


# 추정 허용 범위
#                       lift   gamma  gain  contrast
WIDE_LOWER = np.array([-0.15, 0.3,   0.3,  0.5])
WIDE_UPPER = np.array([0.3,   3.0,   2.5,  2.0])

# 안전 범위: 두 이미지의 '내용 차이'(검은 옷이 많은 레퍼런스 등)를 노출 차이로
# 착각해 계조를 크게 비트는 것을 막는다. 실제 노출 차이는 카메라 앱에서는 EV(센서)로 처리.
SAFE_LOWER = np.array([-0.04, 0.8,  0.85, 0.85])
SAFE_UPPER = np.array([0.04,  1.25, 1.15, 1.2])


def blend_primaries(primaries, strength):
    """
    ①② 강도. 0 = 변화 없음, 1 = 추정값 그대로.
    곱하는 값(gamma, gain, contrast)은 지수로, 더하는 값(lift)은 선형으로 섞는다.
    """

    strength = float(np.clip(strength, 0.0, 1.0))

    return {
        "lift": float(primaries["lift"]) * strength,
        "gamma": float(primaries["gamma"]) ** strength,
        "gain": float(primaries["gain"]) ** strength,
        "contrast": float(primaries["contrast"]) ** strength,
        "pivot": float(primaries["pivot"]),
    }


def estimate_primaries(
    target_rgb,
    reference_rgb,
    sample_count=60_000,
    regularization=0.05,
    separation_ratio=0.4,
    random_seed=42,
    safe_limits=True,
    strength=1.0,
):
    """
    Target -> Reference 의 ① LGG, ② Contrast/Pivot 값을 추정한다.

    루마 분위수 (1, 5, 10, 25, 50, 75, 90, 95, 99%) 를 맞춘다.
    -> 블랙 레벨, 중간톤 밝기, 화이트 레벨, 대비

    separation_ratio : 이웃한 분위수 간격이 원래 간격의 이 비율보다
    좁아지지 않게 한다. (블랙 크러시 / 화이트 클리핑 방지)

    Pivot 은 Reference 의 중간 밝기(루마 중앙값)로 고정한다.
    (대비를 조절해도 '레퍼런스의 중간톤 위치'는 유지되도록)

    safe_limits : True 면 SAFE 범위 안에서만 맞춘다.
        범위 밖이 필요했다면(= 넓은 범위로 맞춘 값이 SAFE 밖) diagnostics["limited"] 에 표시.
    strength : ①② 강도 0~1 (blend_primaries)

    Returns
    -------
    dict
        primaries   : {"lift", "gamma", "gain", "contrast", "pivot"}  (안전 범위·강도 적용 후)
        diagnostics : 맞추기 전/후 루마 분위수, 넓은 범위 추정값(raw_primaries), 제한 여부
    """

    validate_rgb_image(target_rgb)
    validate_rgb_image(reference_rgb)

    random_generator = np.random.default_rng(random_seed)

    target_pixels = sample_pixels(target_rgb, sample_count, random_generator)
    reference_pixels = sample_pixels(reference_rgb, sample_count, random_generator)

    target_luma = luma(target_pixels)
    reference_luma = luma(reference_pixels)

    pivot = float(np.median(reference_luma))

    reference_quantiles = np.percentile(reference_luma, LUMA_QUANTILES)
    anchors = _quantile_anchor_indices(target_luma, LUMA_QUANTILES)

    neutral_vector = _pack(NEUTRAL_PRIMARIES)

    target_steps = np.diff(np.array([
        target_luma[anchor].mean() for anchor in anchors
    ]))
    reference_steps = np.diff(reference_quantiles)

    def residuals(vector):
        output_luma = luma(
            apply_primaries(target_pixels, _unpack(vector, pivot))
        )

        output_quantiles = np.array([
            output_luma[anchor].mean() for anchor in anchors
        ])

        tone_residual = output_quantiles - reference_quantiles

        regularization_residual = (
            (vector - neutral_vector) * regularization
        )

        # 계조 간격 유지: 이웃한 분위수 사이 간격이
        # 원래 간격의 일정 비율보다 좁아지면 벌점 (블랙/화이트 뭉개짐 방지)
        output_steps = np.diff(output_quantiles)
        minimum_steps = np.minimum(target_steps, reference_steps) * separation_ratio

        # 간격이 원래 좁은 섀도우 구간도 똑같이 보호되도록 '비율'로 계산한다.
        separation_residual = (
            np.maximum(minimum_steps - output_steps, 0.0)
            / (minimum_steps + 1e-4)
        ) * 0.15

        return np.concatenate([
            tone_residual,
            regularization_residual,
            separation_residual,
        ])

    def solve(lower, upper):
        return least_squares(
            residuals,
            np.clip(neutral_vector, lower, upper),
            bounds=(lower, upper),
            diff_step=1e-4,
            max_nfev=300,
        )

    wide_solution = solve(WIDE_LOWER, WIDE_UPPER)
    raw_primaries = _unpack(wide_solution.x, pivot)

    if safe_limits:
        lower, upper = SAFE_LOWER, SAFE_UPPER
        solution = solve(lower, upper)
    else:
        lower, upper = WIDE_LOWER, WIDE_UPPER
        solution = wide_solution

    # 넓은 범위 추정값이 안전 범위를 벗어난 항목 = 제한된 항목
    limited = [
        name
        for name, value, low, high in zip(PARAMETER_ORDER, wide_solution.x, SAFE_LOWER, SAFE_UPPER)
        if value < low - 1e-3 or value > high + 1e-3
    ] if safe_limits else []

    primaries = blend_primaries(_unpack(solution.x, pivot), strength)

    at_bounds = [
        name
        for name, value, low, high in zip(
            PARAMETER_ORDER, solution.x, lower, upper
        )
        if np.isclose(value, low, atol=1e-3) or np.isclose(value, high, atol=1e-3)
    ]

    diagnostics = {
        "luma_quantile_levels": LUMA_QUANTILES.tolist(),
        "reference_luma_quantiles": reference_quantiles.tolist(),
        "target_luma_quantiles_before": np.percentile(
            target_luma, LUMA_QUANTILES
        ).tolist(),
        "target_luma_quantiles_after": np.percentile(
            luma(apply_primaries(target_pixels, primaries)),
            LUMA_QUANTILES,
        ).tolist(),
        "parameters_at_bounds": at_bounds,
        "raw_primaries": raw_primaries,
        "safe_limits": bool(safe_limits),
        "limited": limited,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "optimizer_cost": float(solution.cost),
        "optimizer_success": bool(solution.success),
    }

    return {
        "primaries": primaries,
        "diagnostics": diagnostics,
    }


def format_primaries_table(primaries):
    """
    UI / 로그 표시용 표 데이터 (컬러리스트 작업 순서).
    """

    return {
        "단계": [
            "① Lift",
            "① Gamma (지수)",
            "① Gain",
            "② Contrast",
            "② Pivot",
        ],
        "값": [
            primaries["lift"],
            primaries["gamma"],
            primaries["gain"],
            primaries["contrast"],
            primaries["pivot"],
        ],
        "변화 없음": [0.0, 1.0, 1.0, 1.0, "-"],
    }
