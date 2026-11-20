# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

"""
레터박스 / 필러박스(검은 띠) 자동 감지.

영화 캡처의 위아래(또는 좌우) 검은 띠가 통계에 '블랙'으로 섞이면
①② 섀도우 추정과 ④ 섀도우 색이 틀어진다. 분석 전에 잘라낸다.

규칙
----
1. 가장자리 2줄의 밝기 중앙값을 그쪽 '띠 색'으로 잡는다. (0 이든 16 이든)
   어둡지 않으면 그쪽에는 띠가 없다.
2. 줄(행/열)마다 '띠 줄' 판정
   - 밝기 중앙값이 띠 색과 거의 같고                (± LEVEL_TOLERANCE)
   - 띠 색보다 밝은 픽셀이 BRIGHT_FRACTION_MAX 이하이고 (자막·재생바는 성겨서 통과)
   - 밝은 픽셀을 뺀 나머지가 고르다                 (UNIFORM_STD_MAX)
3. 가장자리에서 안쪽으로 훑다가 '띠가 아닌 줄'이 CONFIRM_LINES 줄 연속이면 경계.
4. 검증
   - 경계 계단: 경계 안쪽이 띠 색보다 확실히 밝아야 한다. (밤하늘처럼 서서히 밝아지면 자르지 않음)
   - 한쪽만 띠면 신뢰도를 낮춘다. 위아래 대칭은 요구하지 않는다.
     (Mac 스크린샷은 메뉴바·노치 때문에 위 178 / 아래 112 처럼 비대칭인 경우가 실제로 있다)
   - 잘라낸 비율이 알려진 화면비(2.39, 1.85, 16:9 ...)에 가까우면 신뢰도 가산.
5. 경계에서 몇 픽셀 더 안쪽까지 잘라 압축 번짐을 뺀다.

결과는 자를 영역 + 신뢰도. 이미지는 crop_letterbox 로 따로 자른다.
"""

import numpy as np

from .image_io import validate_rgb_image


DARK_MAX = 24.0 / 255.0             # 띠 색으로 인정하는 최대 밝기
LEVEL_TOLERANCE = 3.0 / 255.0       # 띠 줄 중앙값과 띠 색의 허용 차이
BRIGHT_OFFSET = 15.0 / 255.0        # 띠 색보다 이만큼 밝으면 '밝은 픽셀' (자막 등)
BRIGHT_FRACTION_MAX = 0.2           # 한 줄에서 밝은 픽셀 허용 비율
UNIFORM_STD_MAX = 2.5 / 255.0       # 밝은 픽셀을 뺀 나머지의 표준편차 상한
CONFIRM_LINES = 3                   # 띠가 아닌 줄이 이만큼 연속이면 경계
MIN_STEP = 8.0 / 255.0              # 경계 안쪽이 띠 색보다 최소 이만큼 밝아야 함
MAX_CROP_FRACTION = 0.6             # 한 축에서 이보다 많이 잘리면 의심
ASPECT_TOLERANCE = 0.02

KNOWN_ASPECTS = {
    "2.39:1": 2.39,
    "2.35:1": 2.35,
    "2.20:1": 2.20,
    "2:1": 2.00,
    "1.85:1": 1.85,
    "16:9": 16 / 9,
    "1.66:1": 1.66,
    "3:2": 1.5,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "4:5": 0.8,
    "3:4": 0.75,
    "9:16": 9 / 16,
}

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _luma_image(rgb_image):
    return (rgb_image.astype(np.float32) / 255.0) @ _LUMA


def _line_is_bar(lines, level):
    """
    lines : (줄 수, 샘플 수) 루마. 각 줄이 띠 줄인지 (줄 수,) bool.
    """

    median = np.median(lines, axis=1)
    bright = lines > level + BRIGHT_OFFSET
    bright_fraction = bright.mean(axis=1)

    # 밝은 픽셀(자막 등)은 띠 색으로 바꿔서 고른지 본다.
    rest = np.where(bright, level, lines)
    rest_std = rest.std(axis=1)

    return (
        (np.abs(median - level) <= LEVEL_TOLERANCE)
        & (bright_fraction <= BRIGHT_FRACTION_MAX)
        & (rest_std <= UNIFORM_STD_MAX)
    )


def _scan_side(lines):
    """
    lines : 가장자리부터 안쪽 순서의 (줄 수, 샘플 수) 루마.
    Returns (띠 두께, 띠 색, 경계 계단 크기). 띠가 없으면 두께 0.
    """

    level = float(np.median(lines[:2]))

    if level > DARK_MAX:
        return 0, level, 0.0

    is_bar = _line_is_bar(lines, level)

    count = len(is_bar)
    position = 0

    while position < count:
        window = is_bar[position:position + CONFIRM_LINES]
        if len(window) == CONFIRM_LINES and not window.any():
            break
        if len(window) < CONFIRM_LINES and not window.any():
            break
        position += 1

    if position >= count:
        # 전부 띠 → 이미지 전체가 어두움
        return count, level, 0.0

    thickness = position

    if thickness == 0:
        return 0, level, 0.0

    inner = lines[thickness:thickness + 8]
    step = float(max(
        inner.mean() - level,
        (np.percentile(inner, 90) - level) / 3.0,
    ))

    return thickness, level, step


def _nearest_aspect(aspect):
    name, value = min(KNOWN_ASPECTS.items(), key=lambda item: abs(item[1] - aspect) / item[1])
    return name, abs(value - aspect) / value <= ASPECT_TOLERANCE


def detect_letterbox(rgb_image, max_samples_per_line=1024):
    """
    검은 띠를 찾는다.

    Returns
    -------
    dict
        top, bottom, left, right : 잘라낼 픽셀 수 (안전 마진 포함)
        width, height            : 잘라낸 뒤 크기
        confidence               : "high" | "low" | "none"
        aspect, aspect_name, aspect_match
        bar_level                : 띠 색 (0~1)
        warnings                 : 한국어 안내 목록
    """

    validate_rgb_image(rgb_image)

    height, width = rgb_image.shape[:2]
    luma = _luma_image(rgb_image)

    column_step = max(1, width // max_samples_per_line)
    row_step = max(1, height // max_samples_per_line)

    rows = luma[:, ::column_step]          # (height, samples)
    columns = luma[::row_step, :].T        # (width, samples)

    sides = {
        "top": _scan_side(rows),
        "bottom": _scan_side(rows[::-1]),
        "left": _scan_side(columns),
        "right": _scan_side(columns[::-1]),
    }

    warnings = []
    crop = {name: 0 for name in sides}
    levels = []

    # 이미지 전체가 어두운 경우 (페이드 아웃 등): 자르지 않는다.
    if sides["top"][0] >= height or sides["left"][0] >= width:
        warnings.append("이미지 거의 전체가 검은색입니다. 너무 어두운 레퍼런스는 룩을 제대로 잡기 어렵습니다.")
        return _result(width, height, crop, "none", None, warnings)

    weak_sides = []

    for name, (thickness, level, step) in sides.items():
        if thickness == 0:
            continue
        if step < MIN_STEP:
            weak_sides.append(name)
            continue
        crop[name] = thickness
        levels.append(level)

    if weak_sides:
        warnings.append(
            "가장자리가 어둡지만 검은 띠로 보기 어려워 자르지 않았습니다 "
            f"({', '.join(_SIDE_NAMES[name] for name in weak_sides)}). 어두운 하늘이나 그림자일 수 있습니다."
        )

    for axis, (first, second), size in (
        ("세로", ("top", "bottom"), height),
        ("가로", ("left", "right"), width),
    ):
        if crop[first] + crop[second] > size * MAX_CROP_FRACTION:
            warnings.append(f"{axis} 방향으로 너무 많이 잘려서 자르지 않았습니다.")
            crop[first] = crop[second] = 0

    if not any(crop.values()):
        return _result(width, height, crop, "none", None, warnings)

    # 안전 마진: 압축 번짐 제거
    for name in crop:
        if crop[name] > 0:
            size = height if name in ("top", "bottom") else width
            crop[name] += 2 + int(round(size * 0.002))

    content_width = width - crop["left"] - crop["right"]
    content_height = height - crop["top"] - crop["bottom"]
    # 화면비 판정은 마진 전 크기로
    raw_aspect = (
        (width - sides["left"][0] * (crop["left"] > 0) - sides["right"][0] * (crop["right"] > 0))
        / (height - sides["top"][0] * (crop["top"] > 0) - sides["bottom"][0] * (crop["bottom"] > 0))
    )
    _, aspect_match = _nearest_aspect(raw_aspect)

    vertical_pair = (crop["top"] > 0) == (crop["bottom"] > 0)
    horizontal_pair = (crop["left"] > 0) == (crop["right"] > 0)

    if (vertical_pair and horizontal_pair) or aspect_match:
        confidence = "high"
    else:
        confidence = "low"
        warnings.append("한쪽에만 검은 띠가 있습니다. 자른 영역을 확인해 주세요.")

    result = _result(width, height, crop, confidence, float(np.mean(levels)), warnings)
    result["aspect"] = float(raw_aspect)
    result["aspect_name"], result["aspect_match"] = _nearest_aspect(raw_aspect)
    result["width"], result["height"] = int(content_width), int(content_height)

    return result


_SIDE_NAMES = {"top": "위", "bottom": "아래", "left": "왼쪽", "right": "오른쪽"}


def _result(width, height, crop, confidence, bar_level, warnings):
    return {
        "top": int(crop["top"]),
        "bottom": int(crop["bottom"]),
        "left": int(crop["left"]),
        "right": int(crop["right"]),
        "width": int(width - crop["left"] - crop["right"]),
        "height": int(height - crop["top"] - crop["bottom"]),
        "confidence": confidence,
        "aspect": float((width - crop["left"] - crop["right"]) / (height - crop["top"] - crop["bottom"])),
        "aspect_name": None,
        "aspect_match": False,
        "bar_level": bar_level,
        "warnings": warnings,
    }


def crop_letterbox(rgb_image, crop):
    """detect_letterbox 결과대로 자른다."""

    height, width = rgb_image.shape[:2]

    return rgb_image[
        crop["top"]:height - crop["bottom"],
        crop["left"]:width - crop["right"],
    ]


def remove_letterbox(rgb_image):
    """감지 + 자르기. (잘린 이미지, 감지 결과)"""

    crop = detect_letterbox(rgb_image)

    if crop["confidence"] == "none":
        return rgb_image, crop

    return crop_letterbox(rgb_image, crop), crop
