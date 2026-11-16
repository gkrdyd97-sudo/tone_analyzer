# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

import cv2
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .image_io import validate_rgb_image


def resize_target_to_reference(
    target_rgb,
    reference_rgb,
):
    """
    Target 이미지를 Reference와 같은 크기로 변경한다.
    """

    validate_rgb_image(
        target_rgb
    )


    validate_rgb_image(
        reference_rgb
    )


    reference_height, reference_width = (
        reference_rgb.shape[:2]
    )


    resized_target_rgb = cv2.resize(
        target_rgb,
        (
            reference_width,
            reference_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


    return resized_target_rgb


def _random_rotation(random_generator):
    """
    3x3 무작위 직교 회전 행렬을 만든다.
    """

    matrix = random_generator.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))

    return q


def transfer_color_distribution(
    target_samples,
    reference_samples,
    iteration_count=20,
    random_seed=42,
):
    """
    IDT(Iterative Distribution Transfer, Pitié 2005) 방식으로
    target 픽셀들의 RGB 분포를 reference 분포에 맞춘다.

    두 이미지의 '같은 좌표'가 아니라 '색 분포'를 맞추므로
    내용이 다른 사진끼리도 사용할 수 있다.

    매 반복마다 RGB 공간을 무작위로 회전시킨 뒤,
    각 축에서 1D 분위수(quantile) 매칭을 수행한다.
    """

    random_generator = np.random.default_rng(random_seed)

    current = target_samples.astype(np.float64).copy()
    reference = reference_samples.astype(np.float64)

    quantile_levels = np.linspace(0.0, 1.0, 1001)

    for _ in range(iteration_count):

        rotation = _random_rotation(random_generator)

        current_projected = current @ rotation
        reference_projected = reference @ rotation

        for axis in range(3):

            current_quantiles = np.quantile(
                current_projected[:, axis],
                quantile_levels,
            )

            reference_quantiles = np.quantile(
                reference_projected[:, axis],
                quantile_levels,
            )

            # 동일 값 구간에서 interp가 깨지지 않도록 단조 증가 보장
            current_quantiles = np.maximum.accumulate(
                current_quantiles
                + np.arange(len(current_quantiles)) * 1e-9
            )

            current_projected[:, axis] = np.interp(
                current_projected[:, axis],
                current_quantiles,
                reference_quantiles,
            )

        current = current_projected @ rotation.T

    return np.clip(current, 0.0, 1.0).astype(np.float32)


def _rgb_to_lab(samples):
    """(N, 3) 0~1 RGB -> (N, 3) Lab (L: 0~100, a/b: 약 -128~127)"""
    image = samples.reshape(-1, 1, 3).astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def _lab_to_rgb(samples):
    """(N, 3) Lab -> (N, 3) 0~1 RGB"""
    image = samples.reshape(-1, 1, 3).astype(np.float32)
    rgb = cv2.cvtColor(image, cv2.COLOR_Lab2RGB).reshape(-1, 3)
    return np.clip(rgb, 0.0, 1.0)


def transfer_color_reinhard(
    target_samples,
    reference_samples,
    preserve_luminance=True,
):
    """
    Reinhard(2001) 색 전이: Lab 각 채널의 평균/표준편차를 맞춘다.
    전역적인 색 이동만 하므로 과하게 튀지 않는다.
    """

    target_lab = _rgb_to_lab(target_samples)
    reference_lab = _rgb_to_lab(reference_samples)

    channels = [1, 2] if preserve_luminance else [0, 1, 2]
    result_lab = target_lab.copy()

    for channel in channels:
        t_mean = target_lab[:, channel].mean()
        t_std = target_lab[:, channel].std() + 1e-6
        r_mean = reference_lab[:, channel].mean()
        r_std = reference_lab[:, channel].std()

        result_lab[:, channel] = (
            (target_lab[:, channel] - t_mean) * (r_std / t_std) + r_mean
        )

    return _lab_to_rgb(result_lab).astype(np.float32)


def _sample_pixels(
    rgb_image,
    sample_count,
    random_generator,
):
    """
    이미지에서 픽셀을 무작위로 뽑아 0~1 범위 (N, 3) 배열로 반환한다.
    """

    pixel_list = (
        rgb_image.reshape(-1, 3).astype(np.float32) / 255.0
    )

    sample_count = min(sample_count, len(pixel_list))

    indices = random_generator.choice(
        len(pixel_list),
        size=sample_count,
        replace=False,
    )

    return pixel_list[indices], indices


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_tonal_weights(
    lightness,
    shadow_percentile=20.0,
    highlight_percentile=80.0,
):
    """
    밝기(L, 0~100)로 섀도우 / 미드톤 / 하이라이트 소프트 가중치를 만든다.

    analysis.make_tonal_masks 와 같은 퍼센타일 기준을 쓰되,
    경계에서 색이 끊기지 않도록 부드럽게 전환한다.

    Returns
    -------
    (N, 3) 배열. 열 순서: shadow, midtone, highlight. 행의 합은 1.
    """

    shadow_threshold, highlight_threshold = np.percentile(
        lightness,
        [shadow_percentile, highlight_percentile],
    )

    width = max(
        (highlight_threshold - shadow_threshold) * 0.35,
        2.0,
    )

    shadow = 1.0 - _smoothstep(
        shadow_threshold - width,
        shadow_threshold + width,
        lightness,
    )

    highlight = _smoothstep(
        highlight_threshold - width,
        highlight_threshold + width,
        lightness,
    )

    midtone = np.clip(1.0 - shadow - highlight, 0.0, 1.0)

    weights = np.stack([shadow, midtone, highlight], axis=1)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8

    return weights.astype(np.float32)


def _weighted_mean_std(values, weights):
    weight_sum = weights.sum() + 1e-8
    mean = (values * weights).sum() / weight_sum
    variance = (((values - mean) ** 2) * weights).sum() / weight_sum
    return mean, np.sqrt(variance)


def transfer_color_reinhard_tonal(
    target_samples,
    reference_samples,
    preserve_luminance=True,
    min_std_ratio=0.4,
    max_std_ratio=2.5,
):
    """
    톤 영역별(Split toning) Reinhard 색 전이.

    섀도우 / 미드톤 / 하이라이트 각각에서
    Lab 평균과 표준편차를 따로 맞춘 뒤,
    target 픽셀의 영역 가중치로 섞는다.
    예) 섀도우는 청록으로, 하이라이트는 따뜻하게 가는 룩을 재현할 수 있다.
    """

    target_lab = _rgb_to_lab(target_samples)
    reference_lab = _rgb_to_lab(reference_samples)

    target_weights = compute_tonal_weights(target_lab[:, 0])
    reference_weights = compute_tonal_weights(reference_lab[:, 0])

    channels = [1, 2] if preserve_luminance else [0, 1, 2]
    result_lab = target_lab.copy()

    for channel in channels:

        blended_channel = np.zeros(len(target_lab), dtype=np.float64)

        for region in range(3):

            t_mean, t_std = _weighted_mean_std(
                target_lab[:, channel],
                target_weights[:, region],
            )

            r_mean, r_std = _weighted_mean_std(
                reference_lab[:, channel],
                reference_weights[:, region],
            )

            # 픽셀 수가 적은 영역에서 값이 폭주하지 않도록 비율 제한
            std_ratio = np.clip(
                r_std / (t_std + 1e-6),
                min_std_ratio,
                max_std_ratio,
            )

            region_result = (
                (target_lab[:, channel] - t_mean) * std_ratio + r_mean
            )

            blended_channel += region_result * target_weights[:, region]

        result_lab[:, channel] = blended_channel

    return _lab_to_rgb(result_lab).astype(np.float32)


def compute_skin_weights(rgb_samples):
    """
    피부색일 가능성(0~1)을 Lab 색상각 / 채도 / 밝기로 추정한다.

    - 색상각 h = atan2(b, a): 피부는 대략 0~65도 (분홍~주황~노랑 기운)
    - 채도 C = sqrt(a^2 + b^2): 너무 무채색이거나 너무 원색이면 제외
    - 밝기 L: 너무 어둡거나 하얗게 날아간 영역 제외

    얼굴 인식 없이 색만으로 판단하므로
    주황빛 배경 등도 일부 포함될 수 있다.
    """

    lab = _rgb_to_lab(rgb_samples)

    lightness = lab[:, 0]
    a_channel = lab[:, 1]
    b_channel = lab[:, 2]

    hue = np.degrees(np.arctan2(b_channel, a_channel))
    chroma = np.sqrt(a_channel ** 2 + b_channel ** 2)

    # 보정된 광고/영화 이미지는 피부가 분홍 쪽(색상각 0도 부근)으로
    # 눌려 있는 경우가 많아 범위를 넓게 잡는다.
    hue_weight = (
        _smoothstep(-15.0, 5.0, hue)
        * (1.0 - _smoothstep(65.0, 80.0, hue))
    )

    chroma_weight = (
        _smoothstep(3.0, 7.0, chroma)
        * (1.0 - _smoothstep(45.0, 60.0, chroma))
    )

    lightness_weight = (
        _smoothstep(20.0, 35.0, lightness)
        * (1.0 - _smoothstep(90.0, 97.0, lightness))
    )

    return (
        hue_weight * chroma_weight * lightness_weight
    ).astype(np.float32)


def prepare_pixel_pairs(
    target_rgb,
    reference_rgb,
    max_sample_count=100_000,
    random_seed=42,
    transfer_strength=1.0,
    method="reinhard",
    preserve_luminance=True,
    tonal_regions=True,
    skin_protection=0.6,
):
    """
    LUT 학습용 (입력 RGB, 목표 RGB) 쌍을 생성한다.

    [중요] Target과 Reference는 서로 다른 사진이므로
    같은 좌표의 픽셀끼리는 아무 관계가 없다.
    따라서 다음 방식으로 '실제로 대응되는' 쌍을 만든다.

    1. Target / Reference 에서 각각 독립적으로 픽셀 샘플링
    2. Target 샘플의 색 분포를 Reference 분포로 전이
    3. (Target 원본 픽셀, 같은 픽셀의 전이 결과)를 학습 쌍으로 사용

    method : "reinhard"(기본, 평균/분산 매칭 - 안정적)
             "idt"(분포 전체 매칭 - 강하지만 과해질 수 있음)

    tonal_regions : True면 섀도우/미드톤/하이라이트별로 따로 전이
                    (reinhard 에서만 사용)

    preserve_luminance : True면 밝기는 1D 톤 커브 결과를 유지하고
    색(Lab의 a, b)만 Reference에 맞춘다.

    skin_protection : 0~1. 피부색으로 추정되는 픽셀은
    이 비율만큼 전이를 약하게 한다. (0이면 보호 안 함)

    transfer_strength : 0~1. 보통 1.0으로 학습하고,
    강도 조절은 lut.blend_lut_with_identity 로 실시간 처리한다.
    """

    if max_sample_count < 1:
        raise ValueError(
            "max_sample_count는 1 이상이어야 합니다."
        )

    validate_rgb_image(target_rgb)
    validate_rgb_image(reference_rgb)

    random_generator = np.random.default_rng(random_seed)

    target_samples, sample_indices = _sample_pixels(
        target_rgb,
        max_sample_count,
        random_generator,
    )

    reference_pool, _ = _sample_pixels(
        reference_rgb,
        max_sample_count,
        random_generator,
    )

    if method == "reinhard":

        if tonal_regions:
            transferred_samples = transfer_color_reinhard_tonal(
                target_samples=target_samples,
                reference_samples=reference_pool,
                preserve_luminance=preserve_luminance,
            )
        else:
            transferred_samples = transfer_color_reinhard(
                target_samples=target_samples,
                reference_samples=reference_pool,
                preserve_luminance=preserve_luminance,
            )

    elif method == "idt":
        transferred_samples = transfer_color_distribution(
            target_samples=target_samples,
            reference_samples=reference_pool,
            random_seed=random_seed,
        )

        if preserve_luminance:
            # 밝기(L)는 1D 톤 커브 결과를 유지하고 색(a, b)만 가져온다.
            transferred_lab = _rgb_to_lab(transferred_samples)
            transferred_lab[:, 0] = _rgb_to_lab(target_samples)[:, 0]
            transferred_samples = _lab_to_rgb(transferred_lab)

    else:
        raise ValueError(
            "method는 'reinhard' 또는 'idt' 여야 합니다."
        )

    # 픽셀별 전이 강도: 피부로 추정되는 픽셀은 약하게
    strength = float(np.clip(transfer_strength, 0.0, 1.0))
    skin_weights = compute_skin_weights(target_samples)

    per_pixel_strength = (
        strength
        * (1.0 - float(np.clip(skin_protection, 0.0, 1.0)) * skin_weights)
    )[:, None]

    desired_samples = (
        target_samples * (1.0 - per_pixel_strength)
        + transferred_samples * per_pixel_strength
    ).astype(np.float32)

    return {
        "resized_target_rgb": target_rgb,
        "target_samples": target_samples,
        # 학습 목표값: 각 target 픽셀이 '되어야 할' 색
        "reference_samples": desired_samples,
        "reference_pool": reference_pool,
        "sample_indices": sample_indices,
        "skin_ratio": float(skin_weights.mean()),
    }


def compute_skin_mask(rgb_image):
    """
    (H, W, 3) uint8 RGB 이미지의 피부 가중치 마스크(0~255 uint8)를 만든다.
    UI에서 피부 보호 영역을 확인할 때 사용한다.
    """

    height, width = rgb_image.shape[:2]
    weights = compute_skin_weights(
        rgb_image.reshape(-1, 3).astype(np.float32) / 255.0
    )

    return (weights.reshape(height, width) * 255.0).astype(np.uint8)


def train_rgb_model(
    target_samples,
    reference_samples,
    n_neighbors=8,
):
    """
    Target RGB -> 목표 RGB 변환 모델을 학습한다.

    3차 다항식 + Ridge 를 사용해
    부드럽고(밴딩 없음) 비선형 색 이동도 표현할 수 있는
    매핑을 만든다. LUT의 빈 영역도 자연스럽게 채워진다.

    n_neighbors는 기존 pipeline과의 호환성을 위해 남겨두었다.
    """

    if target_samples.ndim != 2 or target_samples.shape[1] != 3:
        raise ValueError(
            "target_samples는 (N, 3) 배열이어야 합니다."
        )

    if reference_samples.ndim != 2 or reference_samples.shape[1] != 3:
        raise ValueError(
            "reference_samples는 (N, 3) 배열이어야 합니다."
        )

    if len(target_samples) != len(reference_samples):
        raise ValueError(
            "Target과 Reference 샘플 개수가 같아야 합니다."
        )

    model = make_pipeline(
        PolynomialFeatures(degree=3),
        Ridge(alpha=1e-2),
    )

    model.fit(
        target_samples,
        reference_samples,
    )

    return model


def evaluate_rgb_model(
    target_samples,
    reference_samples,
    n_neighbors=8,
    test_size=0.2,
    random_seed=42,
):
    """
    RGB 변환 모델을 학습하고 성능을 평가한다.
    """

    target_train, target_test, reference_train, reference_test = (
        train_test_split(
            target_samples,
            reference_samples,
            test_size=test_size,
            random_state=random_seed,
        )
    )


    model = train_rgb_model(
        target_samples=target_train,
        reference_samples=reference_train,
        n_neighbors=n_neighbors,
    )


    reference_prediction = model.predict(
        target_test,
    )


    reference_prediction = np.clip(
        reference_prediction,
        0.0,
        1.0,
    )


    mean_absolute_error_value = (
        mean_absolute_error(
            reference_test,
            reference_prediction,
        )
    )


    return {
        "model": model,
        "target_train": target_train,
        "target_test": target_test,
        "reference_train": reference_train,
        "reference_test": reference_test,
        "reference_prediction": reference_prediction,
        "mean_absolute_error": float(
            mean_absolute_error_value
        ),
        "mean_absolute_error_8bit": float(
            mean_absolute_error_value * 255.0
        ),
    }