# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

"""
⑤ 스킨톤 교정.

④ 컬러 그레이딩까지 끝난 색에서 피부로 추정되는 색만 골라
색상(hue)을 목표 쪽으로 당기고, 채도를 조절한다.
DaVinci 에서 피부를 퀄리파이어로 잡고 Hue / Sat 을 만지는 작업에 해당한다.

목표 색상
--------
- "line"      : 벡터스코프 스킨톤 라인 (YIQ 의 I 축 방향, Lab 색상각 약 62도)
- "reference" : Reference 이미지 속 피부의 색상 (피부가 충분히 없으면 스킨톤 라인)

LUT 격자점 단위로 적용되므로 .cube 에 그대로 반영되고 실시간 조절된다.
"""

import cv2
import numpy as np

from .image_io import validate_rgb_image


# ─────────────────────────────────────────
# 정확한 sRGB <-> Lab 변환 (OpenCV float Lab 은 1/64 단위로 끊김)
# ─────────────────────────────────────────

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])

_DELTA = 6.0 / 29.0


def srgb_to_lab(pixels):
    rgb = np.clip(np.asarray(pixels, dtype=np.float64), 0.0, 1.0)

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    xyz = (linear @ _RGB_TO_XYZ.T) / _WHITE_D65

    f = np.where(
        xyz > _DELTA ** 3,
        np.cbrt(xyz),
        xyz / (3.0 * _DELTA ** 2) + 4.0 / 29.0,
    )

    return np.stack([
        116.0 * f[:, 1] - 16.0,
        500.0 * (f[:, 0] - f[:, 1]),
        200.0 * (f[:, 1] - f[:, 2]),
    ], axis=1)


def lab_to_srgb(lab):
    lab = np.asarray(lab, dtype=np.float64)

    fy = (lab[:, 0] + 16.0) / 116.0
    fx = fy + lab[:, 1] / 500.0
    fz = fy - lab[:, 2] / 200.0

    f = np.stack([fx, fy, fz], axis=1)

    xyz = np.where(
        f > _DELTA,
        f ** 3,
        3.0 * _DELTA ** 2 * (f - 4.0 / 29.0),
    ) * _WHITE_D65

    linear = np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, 1.0)

    rgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear ** (1.0 / 2.4) - 0.055,
    )

    return np.clip(rgb, 0.0, 1.0)


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def skin_weights_from_lab(lab):
    """
    피부색일 가능성(0~1). color_matching.compute_skin_weights 와 같은 기준.
    """

    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    hue_weight = (
        _smoothstep(-15.0, 5.0, hue)
        * (1.0 - _smoothstep(65.0, 80.0, hue))
    )

    chroma_weight = (
        _smoothstep(3.0, 7.0, chroma)
        * (1.0 - _smoothstep(45.0, 60.0, chroma))
    )

    lightness_weight = (
        _smoothstep(20.0, 35.0, lab[:, 0])
        * (1.0 - _smoothstep(90.0, 97.0, lab[:, 0]))
    )

    return hue_weight * chroma_weight * lightness_weight


# ─────────────────────────────────────────
# 스킨톤 라인
# ─────────────────────────────────────────

def _skin_line_hue():
    """
    YIQ 의 I 축(벡터스코프 스킨톤 라인) 방향 색의 Lab 색상각.
    """

    y, i = 0.55, 0.07
    rgb = np.array([[y + 0.956 * i, y - 0.272 * i, y - 1.106 * i]])
    lab = srgb_to_lab(rgb)[0]
    return float(np.degrees(np.arctan2(lab[2], lab[1])))


SKIN_LINE_HUE = _skin_line_hue()


def _circular_weighted_mean(hue_degrees, weights):
    radians = np.radians(hue_degrees)
    return float(np.degrees(np.arctan2(
        (np.sin(radians) * weights).sum(),
        (np.cos(radians) * weights).sum(),
    )))


def measure_skin(
    rgb_image,
    sample_count=200_000,
    min_skin_ratio=0.005,
    random_seed=42,
):
    """
    이미지 속 피부의 평균 색상각 / 채도와 피부 비율을 잰다.

    Returns
    -------
    dict
        hue, chroma : 피부 가중 평균 (피부가 너무 적으면 None)
        skin_ratio  : 피부 가중치 평균
    """

    validate_rgb_image(rgb_image)

    pixels = rgb_image.reshape(-1, 3)
    random_generator = np.random.default_rng(random_seed)

    if len(pixels) > sample_count:
        pixels = pixels[
            random_generator.choice(len(pixels), sample_count, replace=False)
        ]

    lab = srgb_to_lab(pixels.astype(np.float64) / 255.0)
    weights = skin_weights_from_lab(lab)
    # 확실한 피부 위주로 측정
    weights = np.where(weights > 0.5, weights, 0.0)

    skin_ratio = float(weights.mean())

    if skin_ratio < min_skin_ratio:
        return {"hue": None, "chroma": None, "skin_ratio": skin_ratio}

    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    return {
        "hue": _circular_weighted_mean(hue, weights),
        "chroma": float((chroma * weights).sum() / weights.sum()),
        "skin_ratio": skin_ratio,
    }


def resolve_target_hue(mode, reference_skin):
    """
    교정 목표 색상각과 실제로 쓰인 기준을 돌려준다.
    """

    if mode == "reference" and reference_skin.get("hue") is not None:
        return reference_skin["hue"], "reference"

    return SKIN_LINE_HUE, "line"


# ─────────────────────────────────────────
# 교정
# ─────────────────────────────────────────

def correct_skin_pixels(
    pixels,
    target_hue,
    hue_strength,
    saturation,
    weights=None,
):
    """
    (N, 3) 0~1 RGB 에서 피부색만 색상을 target_hue 쪽으로 당기고 채도를 조절한다.

    hue_strength : 0~1. 피부의 색상 차이를 이 비율만큼 줄인다.
    saturation   : 피부 채도 배율. 1 = 그대로.
    weights      : 피부 가중치를 직접 줄 때 (LUT 격자에서 넓힌 가중치 등)
    """

    lab = srgb_to_lab(pixels)

    if weights is None:
        weights = skin_weights_from_lab(lab)

    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    # 가장 짧은 방향의 각도 차이 (-180 ~ 180)
    hue_difference = (target_hue - hue + 180.0) % 360.0 - 180.0

    amount = np.clip(weights, 0.0, 1.0)

    new_hue = np.radians(
        hue + hue_difference * float(np.clip(hue_strength, 0.0, 1.0)) * amount
    )
    new_chroma = chroma * (1.0 + (float(saturation) - 1.0) * amount)

    corrected = lab.copy()
    corrected[:, 1] = new_chroma * np.cos(new_hue)
    corrected[:, 2] = new_chroma * np.sin(new_hue)

    return lab_to_srgb(corrected)


def correction_qualifier(lab, skin_chroma=None):
    """
    교정용 피부 퀄리파이어 (보호용보다 좁다).

    - 색상: 분홍 쪽 하한을 좁혀(-5~8도) 입술 중앙 같은 분홍빛을 덜 잡는다.
    - 채도: 이미지의 실제 피부 채도(skin_chroma)를 기준으로
      1.6배를 넘으면 줄이기 시작해 2.4배에서 0 이 된다. (입술, 립 컬러 제외)
    """

    weights = skin_weights_from_lab(lab)

    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    weights = weights * _smoothstep(-5.0, 8.0, hue)

    if skin_chroma is not None and skin_chroma > 0:
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        weights = weights * (
            1.0 - _smoothstep(1.6 * skin_chroma, 2.4 * skin_chroma, chroma)
        )

    return weights


def build_skin_correction_lut(
    lut_input,
    target_hue,
    hue_strength,
    saturation,
    skin_chroma=None,
):
    """
    ⑤ 스킨톤 교정 LUT (노드 4).

    skin_chroma : ④ 까지 적용한 이미지의 피부 평균 채도.
                  이보다 훨씬 진한 색(입술 등)은 교정하지 않는다.

    """

    lut_input = np.asarray(lut_input, dtype=np.float64)

    if hue_strength <= 0.0 and abs(saturation - 1.0) < 1e-6:
        return lut_input.astype(np.float32)

    # 피부 '보호'와 달리 가중치를 이웃 격자로 넓히지 않는다.
    # 넓히면 바로 옆 색인 입술까지 끌려가고, 교정은 경계가 부드러워도 괜찮다.
    weights = correction_qualifier(srgb_to_lab(lut_input), skin_chroma)

    return correct_skin_pixels(
        lut_input,
        target_hue=target_hue,
        hue_strength=hue_strength,
        saturation=saturation,
        weights=weights,
    ).astype(np.float32)


# ─────────────────────────────────────────
# 벡터스코프 (확인용)
# ─────────────────────────────────────────

def render_vectorscope(
    rgb_image,
    size=360,
    target_hue=None,
):
    """
    Rec.709 Cb/Cr 벡터스코프 이미지를 만든다. (uint8 RGB)

    - 노란 선: 스킨톤 라인 (YIQ I 축)
    - 주황 점선: 현재 교정 목표 (target_hue 가 스킨톤 라인과 다를 때)
    """

    validate_rgb_image(rgb_image)

    pixels = rgb_image.reshape(-1, 3).astype(np.float64) / 255.0

    if len(pixels) > 400_000:
        step = len(pixels) // 400_000 + 1
        pixels = pixels[::step]

    y = pixels @ np.array([0.2126, 0.7152, 0.0722])
    cb = (pixels[:, 2] - y) / 1.8556
    cr = (pixels[:, 0] - y) / 1.5748

    # Cb/Cr ±0.3 범위를 화면에 맞춤 (일반적인 영상의 분포가 잘 보이도록 확대)
    scale = (size / 2 - 4) / 0.3
    center = size / 2

    xs = np.clip((center + cb * scale).astype(int), 0, size - 1)
    ys = np.clip((center - cr * scale).astype(int), 0, size - 1)

    histogram = np.zeros((size, size), dtype=np.float64)
    np.add.at(histogram, (ys, xs), 1.0)

    intensity = np.log1p(histogram)
    if intensity.max() > 0:
        intensity /= intensity.max()

    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 1] = (intensity * 230).astype(np.uint8)
    image[..., 0] = (intensity * 120).astype(np.uint8)
    image[..., 2] = (intensity * 120).astype(np.uint8)

    graticule = (70, 70, 70)
    cv2.circle(image, (int(center), int(center)), int(scale * 0.3), graticule, 1)
    cv2.circle(image, (int(center), int(center)), int(scale * 0.15), graticule, 1)
    cv2.line(image, (0, int(center)), (size, int(center)), graticule, 1)
    cv2.line(image, (int(center), 0), (int(center), size), graticule, 1)

    def draw_hue_line(lab_hue, color, dashed=False):
        # Lab 색상각 -> 대표 색 -> Cb/Cr 방향
        lab = np.array([[60.0, 30 * np.cos(np.radians(lab_hue)), 30 * np.sin(np.radians(lab_hue))]])
        rgb = lab_to_srgb(lab)[0]
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
        direction = np.array([(rgb[2] - luma) / 1.8556, (rgb[0] - luma) / 1.5748])
        direction /= np.linalg.norm(direction) + 1e-9
        end = (
            int(center + direction[0] * scale * 0.45),
            int(center - direction[1] * scale * 0.45),
        )
        if dashed:
            for t in np.linspace(0, 1, 12)[::2]:
                start_point = (
                    int(center + (end[0] - center) * t),
                    int(center + (end[1] - center) * t),
                )
                stop_point = (
                    int(center + (end[0] - center) * min(t + 1 / 11, 1)),
                    int(center + (end[1] - center) * min(t + 1 / 11, 1)),
                )
                cv2.line(image, start_point, stop_point, color, 1, cv2.LINE_AA)
        else:
            cv2.line(image, (int(center), int(center)), end, color, 1, cv2.LINE_AA)

    draw_hue_line(SKIN_LINE_HUE, (230, 200, 60))

    if target_hue is not None and abs(target_hue - SKIN_LINE_HUE) > 1.0:
        draw_hue_line(target_hue, (255, 140, 40), dashed=True)

    return image
