import cv2
import numpy as np

from .image_io import validate_rgb_image


def rgb_to_lab(
    rgb_image,
):
    """
    RGB 이미지를 Lab 이미지로 변환한다.

    Parameters
    ----------
    rgb_image : np.ndarray
        RGB 이미지

        형태:
        (높이, 너비, 3)

        자료형:
        uint8

    Returns
    -------
    lab_image : np.ndarray
        OpenCV 형식의 Lab 이미지

        형태:
        (높이, 너비, 3)

        자료형:
        uint8
    """

    validate_rgb_image(
        rgb_image
    )


    lab_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2LAB,
    )


    return lab_image


def extract_lightness(
    rgb_image,
):
    """
    RGB 이미지에서 밝기 채널을 추출한다.

    OpenCV의 Lab L 채널은
    0~255 범위로 저장된다.

    이 함수는 이를 다시
    사람이 이해하기 쉬운 0~100 범위로 변환한다.

    Parameters
    ----------
    rgb_image : np.ndarray
        RGB 이미지

    Returns
    -------
    lightness : np.ndarray
        밝기 배열

        형태:
        (높이, 너비)

        값의 범위:
        0~100

        자료형:
        float32
    """

    lab_image = rgb_to_lab(
        rgb_image
    )


    lightness_channel = (
        lab_image[:, :, 0]
    )


    lightness = (
        lightness_channel.astype(
            np.float32
        )
        / 255.0
        * 100.0
    )


    return lightness


def make_tonal_masks(
    lightness,
    shadow_percentile=20.0,
    highlight_percentile=80.0,
):
    """
    밝기 배열을 Shadow, Midtone, Highlight로 나눈다.

    Parameters
    ----------
    lightness : np.ndarray
        0~100 범위의 밝기 배열

    shadow_percentile : float
        Shadow의 상한 백분위

    highlight_percentile : float
        Highlight의 하한 백분위

    Returns
    -------
    masks : dict
        톤 영역 마스크와 기준값을 담은 딕셔너리
    """

    if lightness.ndim != 2:
        raise ValueError(
            "lightness는 2차원 배열이어야 합니다."
        )


    if not (
        0.0
        <= shadow_percentile
        < highlight_percentile
        <= 100.0
    ):
        raise ValueError(
            "백분위 값의 순서가 올바르지 않습니다."
        )


    shadow_threshold, highlight_threshold = (
        np.percentile(
            lightness,
            [
                shadow_percentile,
                highlight_percentile,
            ],
        )
    )


    shadow_mask = (
        lightness <= shadow_threshold
    )


    midtone_mask = (
        (lightness > shadow_threshold)
        & (lightness < highlight_threshold)
    )


    highlight_mask = (
        lightness >= highlight_threshold
    )


    masks = {
        "shadow": shadow_mask,
        "midtone": midtone_mask,
        "highlight": highlight_mask,
        "shadow_threshold": float(
            shadow_threshold
        ),
        "highlight_threshold": float(
            highlight_threshold
        ),
    }


    return masks


def analyze_tonal_region(
    lightness,
    mask,
    region_name,
):
    """
    하나의 톤 영역에 대한 밝기 통계를 계산한다.

    Parameters
    ----------
    lightness : np.ndarray
        0~100 범위의 밝기 배열

    mask : np.ndarray
        특정 영역을 나타내는 bool 배열

    region_name : str
        영역 이름

    Returns
    -------
    statistics : dict
        영역별 통계 결과
    """

    if lightness.shape != mask.shape:
        raise ValueError(
            "lightness와 mask의 크기가 "
            "같아야 합니다."
        )


    selected_pixels = lightness[mask]


    if len(selected_pixels) == 0:
        raise ValueError(
            f"{region_name} 영역에 픽셀이 없습니다."
        )


    pixel_count = len(
        selected_pixels
    )


    total_pixel_count = lightness.size


    statistics = {
        "region": region_name,
        "pixel_count": int(
            pixel_count
        ),
        "pixel_ratio": float(
            pixel_count / total_pixel_count
        ),
        "mean": float(
            selected_pixels.mean()
        ),
        "std": float(
            selected_pixels.std()
        ),
        "min": float(
            selected_pixels.min()
        ),
        "max": float(
            selected_pixels.max()
        ),
    }


    return statistics


def analyze_all_tonal_regions(
    lightness,
    masks,
):
    """
    Shadow, Midtone, Highlight를
    한 번에 분석한다.

    Parameters
    ----------
    lightness : np.ndarray
        0~100 범위의 밝기 배열

    masks : dict
        make_tonal_masks()가 반환한 마스크

    Returns
    -------
    region_statistics : dict
        세 톤 영역의 통계 결과
    """

    region_statistics = {}


    region_statistics["shadow"] = (
        analyze_tonal_region(
            lightness=lightness,
            mask=masks["shadow"],
            region_name="shadow",
        )
    )


    region_statistics["midtone"] = (
        analyze_tonal_region(
            lightness=lightness,
            mask=masks["midtone"],
            region_name="midtone",
        )
    )


    region_statistics["highlight"] = (
        analyze_tonal_region(
            lightness=lightness,
            mask=masks["highlight"],
            region_name="highlight",
        )
    )


    return region_statistics


def calculate_contrast(
    lightness,
    masks,
):
    """
    이미지의 대비 관련 지표를 계산한다.

    Parameters
    ----------
    lightness : np.ndarray
        0~100 범위의 밝기 배열

    masks : dict
        make_tonal_masks()가 반환한 마스크

    Returns
    -------
    contrast : dict
        대비 지표
    """

    percentile_05, percentile_95 = (
        np.percentile(
            lightness,
            [5.0, 95.0],
        )
    )


    shadow_pixels = (
        lightness[masks["shadow"]]
    )


    highlight_pixels = (
        lightness[masks["highlight"]]
    )


    if len(shadow_pixels) == 0:
        raise ValueError(
            "Shadow 영역에 픽셀이 없습니다."
        )


    if len(highlight_pixels) == 0:
        raise ValueError(
            "Highlight 영역에 픽셀이 없습니다."
        )


    shadow_mean = (
        shadow_pixels.mean()
    )


    highlight_mean = (
        highlight_pixels.mean()
    )


    contrast = {
        "rms": float(
            lightness.std()
        ),
        "percentile": float(
            percentile_95 - percentile_05
        ),
        "shadow_mean": float(
            shadow_mean
        ),
        "highlight_mean": float(
            highlight_mean
        ),
        "gap": float(
            highlight_mean - shadow_mean
        ),
    }


    return contrast