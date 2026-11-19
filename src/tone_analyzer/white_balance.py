# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

"""
③ 화이트밸런스 (Temp / Tint) 추정.

LGG / Contrast 로 계조를 맞춘 이미지에서 '원래 회색이어야 할' 저채도 픽셀(중성색)을 찾고

- mode="neutral"   (기본): 중성색을 실제 회색(Lab a = b = 0) 쪽으로 보정한다.
                           컬러리스트의 밸런스 작업. 룩의 색감은 ④ 가 담당한다.
- mode="reference"       : 중성색을 Reference 의 중성색 쪽으로 옮긴다.
                           (Reference 의 회색에 섞인 룩 색감까지 따라간다)

수식
----
    gain_R = exp(+temp),  gain_G = exp(-tint),  gain_B = exp(-temp)
    gains  = gains / luma(gains)        # 밝기는 유지 (루마 = 1)
    y      = x * gains

    temp  > 0 : 따뜻하게 (R↑ B↓)      temp  < 0 : 차갑게
    tint  > 0 : 마젠타 쪽 (G↓)        tint  < 0 : 그린 쪽

DaVinci 의 Temp(K) / Tint 숫자와 단위가 다르므로,
Resolve 에는 함께 표시되는 RGB Gain 비율(예: Primaries 의 Gain R/G/B)로
옮기거나 눈으로 맞추는 것을 권장한다.
"""

import numpy as np
from scipy.optimize import least_squares

from .image_io import validate_rgb_image
from .primaries import LUMA_WEIGHTS, luma, sample_pixels


NEUTRAL_WHITE_BALANCE = {
    "temp": 0.0,
    "tint": 0.0,
}


def white_balance_gains(temp, tint):
    gains = np.exp(np.array([temp, -tint, -temp], dtype=np.float64))
    return gains / (gains @ LUMA_WEIGHTS)


def apply_white_balance(pixels, white_balance):
    """
    (N, 3) 0~1 float RGB 에 화이트밸런스를 적용한다.
    """

    gains = white_balance_gains(
        white_balance["temp"],
        white_balance["tint"],
    )

    return np.clip(
        np.asarray(pixels, dtype=np.float64) * gains,
        0.0,
        1.0,
    )


_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def _lab_ab(pixels):
    """
    sRGB(0~1) -> CIE Lab 의 (a, b).

    OpenCV 의 float Lab 변환은 내부 근사 때문에 값이 1/64 단위로 끊겨
    미세한 변화를 다루는 최적화에 쓸 수 없으므로 정확한 수식으로 계산한다.
    """

    rgb = np.clip(np.asarray(pixels, dtype=np.float64), 0.0, 1.0)

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    xyz = (linear @ _RGB_TO_XYZ.T) / _WHITE_D65

    delta = 6.0 / 29.0
    f = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3.0 * delta ** 2) + 4.0 / 29.0,
    )

    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])

    return np.stack([a, b], axis=1)


def _neutral_mask(pixels, neutral_percentile=20.0, exclude_weights=None):
    """
    중성색 후보: 너무 어둡거나 밝지 않고(루마 20~97%, 거의 흰색 제외),
    상대 채도((max - min) / max)가 낮은 하위 neutral_percentile % 픽셀.

    절대 채도(max - min)를 쓰면 어두운 픽셀이 무조건 '무채색'으로 뽑혀
    밸런스가 섀도우 노이즈에 끌려가므로 상대 채도를 쓴다.
    exclude_weights 가 주어지면(피부 등) 0.3 이상인 픽셀은 후보에서 뺀다.
    """

    luma_values = luma(pixels)
    low, high = np.percentile(luma_values, [20.0, 97.0])
    in_range = (luma_values > low) & (luma_values < high) & (pixels.max(axis=1) < 0.985)

    if exclude_weights is not None:
        in_range &= exclude_weights < 0.3

    maximum = pixels.max(axis=1)
    saturation = (maximum - pixels.min(axis=1)) / (maximum + 1e-4)
    threshold = np.percentile(saturation[in_range], neutral_percentile)

    return in_range & (saturation <= threshold)


def estimate_white_balance(
    target_rgb,
    reference_rgb,
    strength=0.7,
    mode="neutral",
    skin_cluster=None,
    sample_count=60_000,
    random_seed=42,
):
    """
    Target(①② 적용 후) -> Reference 의 Temp / Tint 를 추정한다.

    Parameters
    ----------
    strength : 0~1
        목표 쪽으로 얼마나 옮길지. neutral 모드에서 1 이면 중성색을 완전히 회색으로.
        조명의 의도된 색(텅스텐의 따뜻함 등)을 조금 남기도록 기본 0.7.

    mode : "neutral" | "reference"

    skin_cluster : 피부 색 범위. 피부는 중성색 후보에서 뺀다.
    """

    validate_rgb_image(target_rgb)
    validate_rgb_image(reference_rgb)

    random_generator = np.random.default_rng(random_seed)

    target_pixels = sample_pixels(target_rgb, sample_count, random_generator)
    reference_pixels = sample_pixels(reference_rgb, sample_count, random_generator)

    exclude = None

    if skin_cluster is not None:
        from .skin_tone import cluster_weights, srgb_to_lab

        exclude = cluster_weights(srgb_to_lab(target_pixels), skin_cluster)

    target_neutrals = target_pixels[_neutral_mask(target_pixels, exclude_weights=exclude)]
    reference_neutrals = reference_pixels[_neutral_mask(reference_pixels)]

    target_ab = np.median(_lab_ab(target_neutrals), axis=0)
    reference_ab = np.median(_lab_ab(reference_neutrals), axis=0)

    strength = float(np.clip(strength, 0.0, 1.0))

    if mode == "reference":
        destination_ab = reference_ab
    elif mode == "neutral":
        destination_ab = np.zeros(2)
    else:
        raise ValueError("mode는 'neutral' 또는 'reference' 여야 합니다.")

    goal_ab = target_ab + (destination_ab - target_ab) * strength

    def residuals(vector):
        balanced = apply_white_balance(
            target_neutrals,
            {"temp": vector[0], "tint": vector[1]},
        )
        return np.concatenate([
            np.median(_lab_ab(balanced), axis=0) - goal_ab,
            vector * 0.5,       # 과한 보정 억제
        ])

    solution = least_squares(
        residuals,
        np.zeros(2),
        bounds=(np.array([-0.4, -0.4]), np.array([0.4, 0.4])),
        diff_step=1e-3,
        max_nfev=200,
    )

    white_balance = {
        "temp": float(solution.x[0]),
        "tint": float(solution.x[1]),
    }

    balanced_ab = np.median(
        _lab_ab(apply_white_balance(target_neutrals, white_balance)),
        axis=0,
    )

    white_balance["gains"] = white_balance_gains(
        white_balance["temp"],
        white_balance["tint"],
    ).tolist()

    diagnostics = {
        "neutral_ab_reference": reference_ab.tolist(),
        "neutral_ab_before": target_ab.tolist(),
        "neutral_ab_goal": goal_ab.tolist(),
        "neutral_ab_after": balanced_ab.tolist(),
        "neutral_pixel_count": int(len(target_neutrals)),
        "mode": mode,
    }

    return {
        "white_balance": white_balance,
        "diagnostics": diagnostics,
    }


def format_white_balance_table(white_balance):
    gains = white_balance["gains"]

    return {
        "항목": [
            "③ Temp (+따뜻 / −차갑)",
            "③ Tint (+마젠타 / −그린)",
            "RGB Gain 비율 R",
            "RGB Gain 비율 G",
            "RGB Gain 비율 B",
        ],
        "값": [
            white_balance["temp"],
            white_balance["tint"],
            gains[0],
            gains[1],
            gains[2],
        ],
    }
