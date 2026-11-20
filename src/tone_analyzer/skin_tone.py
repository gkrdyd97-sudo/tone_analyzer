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


# ─────────────────────────────────────────
# 얼굴 인식 기반 피부 샘플링
# ─────────────────────────────────────────

FACE_MODEL_PATH = (
    __import__("pathlib").Path(__file__).resolve().parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


def detect_faces(rgb_image, max_side=1280, score_threshold=0.6):
    """
    YuNet 으로 얼굴을 찾는다. 원본 좌표계의 (x, y, w, h, score) 목록을 돌려준다.
    모델이나 OpenCV 기능이 없으면 빈 목록 (색 기반 추정으로 대체된다).
    """

    validate_rgb_image(rgb_image)

    if not FACE_MODEL_PATH.exists() or not hasattr(cv2, "FaceDetectorYN"):
        return []

    height, width = rgb_image.shape[:2]
    scale = min(1.0, max_side / max(height, width))

    small = cv2.resize(
        rgb_image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    detector = cv2.FaceDetectorYN.create(
        str(FACE_MODEL_PATH),
        "",
        (bgr.shape[1], bgr.shape[0]),
        score_threshold=score_threshold,
    )

    _, faces = detector.detect(bgr)

    if faces is None:
        return []

    return [
        (
            float(face[0] / scale),
            float(face[1] / scale),
            float(face[2] / scale),
            float(face[3] / scale),
            float(face[14]),
        )
        for face in faces
    ]


def sample_face_pixels(rgb_image, faces, max_samples=8000, random_seed=42):
    """
    얼굴 상자 안쪽 타원(볼·이마·코 부위)에서 피부 픽셀을 뽑는다.

    - 눈썹·눈·머리카락(아주 어두운 픽셀)과 배경(피부와 동떨어진 색)을 거르고
    - 중앙값에서 크게 벗어난 색(입술, 그림자 끝)을 한 번 더 걸러낸다.

    Returns
    -------
    (N, 3) 0~1 RGB. 얼굴이 없거나 샘플이 너무 적으면 None.
    """

    if not faces:
        return None

    height, width = rgb_image.shape[:2]
    collected = []

    for x, y, w, h, _ in faces:
        center_x = x + w * 0.5
        center_y = y + h * 0.55
        radius_x = w * 0.32
        radius_y = h * 0.36

        x0, x1 = int(max(center_x - radius_x, 0)), int(min(center_x + radius_x + 1, width))
        y0, y1 = int(max(center_y - radius_y, 0)), int(min(center_y + radius_y + 1, height))

        if x1 - x0 < 3 or y1 - y0 < 3:
            continue

        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        inside = (
            ((grid_x - center_x) / radius_x) ** 2
            + ((grid_y - center_y) / radius_y) ** 2
        ) <= 1.0

        collected.append(rgb_image[y0:y1, x0:x1][inside])

    if not collected:
        return None

    pixels = np.concatenate(collected).astype(np.float64) / 255.0

    lab = srgb_to_lab(pixels)
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    # 너무 어둡거나(눈·눈썹·머리카락) 날아간 픽셀, 무채색 제외
    keep = (lab[:, 0] > 15.0) & (lab[:, 0] < 97.0) & (chroma > 2.0)
    pixels, lab = pixels[keep], lab[keep]

    if len(pixels) < 50:
        return None

    # 중앙값 기준 a, b 이상치 제거 (입술, 배경 섞임)
    median_ab = np.median(lab[:, 1:], axis=0)
    distance = np.linalg.norm(lab[:, 1:] - median_ab, axis=1)
    limit = np.median(distance) * 2.5 + 1.0
    pixels = pixels[distance <= limit]

    if len(pixels) < 50:
        return None

    if len(pixels) > max_samples:
        random_generator = np.random.default_rng(random_seed)
        pixels = pixels[
            random_generator.choice(len(pixels), max_samples, replace=False)
        ]

    return pixels


def color_based_skin_pixels(rgb_image, max_samples=8000, random_seed=42):
    """
    얼굴을 찾지 못했을 때의 대체: 색 기준 피부 가중치가 높은 픽셀.
    """

    pixels = rgb_image.reshape(-1, 3).astype(np.float64) / 255.0
    random_generator = np.random.default_rng(random_seed)

    if len(pixels) > 200_000:
        pixels = pixels[random_generator.choice(len(pixels), 200_000, replace=False)]

    weights = skin_weights_from_lab(srgb_to_lab(pixels))
    pixels = pixels[weights > 0.5]

    if len(pixels) < 50:
        return None

    if len(pixels) > max_samples:
        pixels = pixels[random_generator.choice(len(pixels), max_samples, replace=False)]

    return pixels


def skin_cluster_from_pixels(pixels):
    """
    피부 샘플의 색 범위(퀄리파이어)를 만든다.

    Returns
    -------
    dict 또는 None
        hue, hue_width : 색상 중심과 허용 폭(도)
        chroma, chroma_low, chroma_high
        lightness_low, lightness_high
    """

    if pixels is None or len(pixels) < 50:
        return None

    lab = srgb_to_lab(pixels)
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    center = _circular_weighted_mean(hue, np.ones(len(hue)))
    deviation = np.abs((hue - center + 180.0) % 360.0 - 180.0)

    # 얼굴 박스에는 그림자 끝, 손, 머리카락 경계가 섞이므로
    # 양 끝을 잘라낸 '중심부' 분포로 범위를 잡는다.
    chroma_low, chroma_mid, chroma_high = np.percentile(chroma, [20, 50, 80])
    lightness_low, lightness_high = np.percentile(lab[:, 0], [10, 90])

    return {
        "hue": float(center),
        "hue_width": float(np.clip(np.percentile(deviation, 75) * 1.6, 8.0, 20.0)),
        "chroma": float(chroma_mid),
        "chroma_low": float(max(chroma_low * 0.75, 0.0)),
        "chroma_high": float(chroma_high * 1.2 + 1.5),
        "lightness_low": float(max(lightness_low - 6.0, 0.0)),
        "lightness_high": float(min(lightness_high + 6.0, 100.0)),
        "sample_count": int(len(pixels)),
    }


def cluster_weights(lab, cluster):
    """
    피부 색 범위(cluster) 안에 있을수록 1 에 가까운 부드러운 가중치.
    """

    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    deviation = np.abs((hue - cluster["hue"] + 180.0) % 360.0 - 180.0)

    width = cluster["hue_width"]
    hue_weight = 1.0 - _smoothstep(width * 0.6, width, deviation)

    chroma_weight = (
        _smoothstep(cluster["chroma_low"] - 3.0, cluster["chroma_low"], chroma)
        * (1.0 - _smoothstep(cluster["chroma_high"], cluster["chroma_high"] + 6.0, chroma))
    )

    lightness_weight = (
        _smoothstep(cluster["lightness_low"] - 8.0, cluster["lightness_low"], lab[:, 0])
        * (1.0 - _smoothstep(cluster["lightness_high"], cluster["lightness_high"] + 8.0, lab[:, 0]))
    )

    return hue_weight * chroma_weight * lightness_weight


def skin_mask_from_cluster(rgb_image, cluster):
    """
    (H, W) uint8 피부 마스크 (UI 확인용).
    """

    height, width = rgb_image.shape[:2]

    if cluster is None:
        weights = skin_weights_from_lab(
            srgb_to_lab(rgb_image.reshape(-1, 3).astype(np.float64) / 255.0)
        )
    else:
        weights = cluster_weights(
            srgb_to_lab(rgb_image.reshape(-1, 3).astype(np.float64) / 255.0),
            cluster,
        )

    return (weights.reshape(height, width) * 255.0).astype(np.uint8)


def skin_leakage(rgb_image, cluster, faces, max_side=640):
    """
    피부 색 범위에 해당하는 영역 중 '얼굴 밖'의 비율 (이미지 전체 대비).

    값이 크면 배경에 피부와 같은 색이 많다는 뜻이고,
    LUT 는 위치를 구분하지 못하므로 피부 보호/교정이 배경에도 적용된다.
    """

    if cluster is None:
        return None

    height, width = rgb_image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    small = cv2.resize(
        rgb_image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    weights = cluster_weights(
        srgb_to_lab(small.reshape(-1, 3).astype(np.float64) / 255.0),
        cluster,
    ).reshape(small.shape[:2])

    face_mask = np.zeros(small.shape[:2], dtype=bool)

    for x, y, w, h, _ in faces or []:
        # 얼굴 아래 목/손까지 여유 있게 포함
        x0 = int(max((x - w * 0.3) * scale, 0))
        x1 = int(min((x + w * 1.3) * scale, small.shape[1]))
        y0 = int(max((y - h * 0.2) * scale, 0))
        y1 = int(min((y + h * 1.8) * scale, small.shape[0]))
        face_mask[y0:y1, x0:x1] = True

    return float((weights * ~face_mask).mean())


def measure_skin_samples(rgb_image, faces=None):
    """
    피부 샘플과 정보를 얻는다. 얼굴 인식 우선, 실패하면 색 기반.

    Returns
    -------
    (info, samples)
        info    : {"source": "face"|"color"|None, "face_count", "hue", "chroma"}
        samples : (N, 3) 0~1 RGB 또는 None
    """

    if faces is None:
        faces = detect_faces(rgb_image)

    samples = sample_face_pixels(rgb_image, faces)
    source = "face"

    if samples is None:
        samples = color_based_skin_pixels(rgb_image)
        source = "color" if samples is not None else None

    cluster = skin_cluster_from_pixels(samples)

    info = {
        "source": source,
        "face_count": int(len(faces)),
        "hue": cluster["hue"] if cluster else None,
        "chroma": cluster["chroma"] if cluster else None,
        "lightness": (
            float(np.median(srgb_to_lab(samples)[:, 0])) if samples is not None else None
        ),
    }

    return info, samples


MIN_RELIABLE_SKIN_CHROMA = 6.0


def is_reliable_skin(skin_info):
    """
    측정한 피부가 목표로 쓸 만한지.
    얼굴에서 잰 것이 아니거나, 얼굴이 너무 어둡거나(후드, 역광) 무채색에 가까우면 색상각을 믿을 수 없다.
    """

    # 얼굴에서 잰 피부만 믿는다. 색 기준 추정(얼굴 없음)은 나무·벽일 수 있다.
    return (
        skin_info is not None
        and skin_info.get("source", "face") == "face"
        and skin_info.get("hue") is not None
        and (skin_info.get("chroma") or 0.0) >= MIN_RELIABLE_SKIN_CHROMA
    )


def resolve_target_hue(mode, reference_skin):
    """
    교정 목표 색상각과 실제로 쓰인 기준을 돌려준다.
    Reference 피부를 믿을 수 없으면 스킨톤 라인을 쓴다.
    """

    if mode == "reference" and is_reliable_skin(reference_skin):
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
    lightness_scale=0.0,
    lightness_weights=None,
):
    """
    (N, 3) 0~1 RGB 에서 피부색만 색상을 target_hue 쪽으로 당기고 채도·밝기를 조절한다.

    hue_strength    : 0~1. 피부의 색상 차이를 이 비율만큼 줄인다.
    saturation      : 피부 채도 배율. 1 = 그대로.
    weights         : 피부 가중치를 직접 줄 때 (LUT 격자에서 넓힌 가중치 등)
    lightness_scale   : 피부 밝기 배율 - 1 (예: -0.1 = 10% 어둡게). 비율로 바꿔서 피부 결(질감)을 유지한다.
    lightness_weights : 밝기용 가중치. 색상용보다 넓고 부드러운 키를 써야 얼굴 안에서 얼룩이 생기지 않는다.
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
    if lightness_weights is None:
        lightness_weights = amount
    corrected[:, 0] = lab[:, 0] * (1.0 + float(lightness_scale) * np.clip(lightness_weights, 0.0, 1.0))
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


# ⑤ 피부 밝기는 작게만 움직인다. 큰 노출 차이는 ① 프라이머리의 몫이고,
# 퀄리파이어로 밝기를 크게 바꾸면 키 경계에서 얼룩이 생긴다.
MAX_SKIN_LIGHTNESS_SCALE = 0.12


def skin_lightness_weights(lut_input, cluster):
    """
    밝기용 피부 키: 색상용 범위보다 넓게 잡고 LUT 격자에서 한 칸 넓힌다(3x3x3).
    얼굴 안의 모든 피부가 거의 같은 가중치를 받고, 경계는 피부 바깥에서 부드럽게 빠지도록.
    """

    from .lut import _dilate_3d

    wide = dict(cluster)
    wide["hue_width"] = min(cluster["hue_width"] * 1.5, 30.0)
    wide["chroma_low"] = max(cluster["chroma_low"] - 4.0, 0.0)
    wide["chroma_high"] = cluster["chroma_high"] + 8.0
    wide["lightness_low"] = max(cluster["lightness_low"] - 12.0, 0.0)
    wide["lightness_high"] = min(cluster["lightness_high"] + 12.0, 100.0)

    lut_input = np.asarray(lut_input, dtype=np.float64)
    size = int(round(len(lut_input) ** (1.0 / 3.0)))

    weights = cluster_weights(srgb_to_lab(lut_input), wide).reshape(size, size, size)
    return _dilate_3d(weights).reshape(-1)


def build_skin_correction_lut(
    lut_input,
    target_hue,
    hue_strength,
    saturation,
    skin_chroma=None,
    skin_cluster=None,
    lightness_scale=0.0,
):
    """
    ⑤ 스킨톤 교정 LUT. 퀄리파이어로 피부만 잡아 색상·채도·밝기를 맞춘다.

    lightness_scale : 피부 밝기 배율 - 1. 밝기는 넓고 부드러운 키(skin_lightness_weights)로 적용한다.

    skin_cluster : ④ 이후 실제 얼굴 피부의 색 범위. 있으면 이 범위만 교정한다.
                   (없으면 넓은 색 기준 퀄리파이어 사용)

    skin_chroma : ④ 까지 적용한 이미지의 피부 평균 채도.
                  이보다 훨씬 진한 색(입술 등)은 교정하지 않는다.

    """

    lut_input = np.asarray(lut_input, dtype=np.float64)

    if hue_strength <= 0.0 and abs(saturation - 1.0) < 1e-6 and abs(lightness_scale) < 1e-6:
        return lut_input.astype(np.float32)

    # 피부 '보호'와 달리 가중치를 이웃 격자로 넓히지 않는다.
    # 넓히면 바로 옆 색인 입술까지 끌려가고, 교정은 경계가 부드러워도 괜찮다.
    if skin_cluster is not None:
        weights = cluster_weights(srgb_to_lab(lut_input), skin_cluster)
    else:
        weights = correction_qualifier(srgb_to_lab(lut_input), skin_chroma)

    return correct_skin_pixels(
        lut_input,
        target_hue=target_hue,
        hue_strength=hue_strength,
        saturation=saturation,
        weights=weights,
        lightness_scale=lightness_scale,
        lightness_weights=(
            skin_lightness_weights(lut_input, skin_cluster)
            if (skin_cluster is not None and abs(lightness_scale) > 1e-6) else None
        ),
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
