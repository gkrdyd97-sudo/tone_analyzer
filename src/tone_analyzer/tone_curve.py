import cv2
import numpy as np

from .image_io import validate_rgb_image


def create_tone_curve(
    target_lightness,
    reference_lightness,
    point_count=101,
):
    """
    Target 밝기를 Reference 밝기로 변환하는
    1D 톤 커브를 생성한다.

    Parameters
    ----------
    target_lightness : np.ndarray
        Target의 밝기 배열

        형태:
        (높이, 너비)

        값의 범위:
        0~100

    reference_lightness : np.ndarray
        Reference의 밝기 배열

        형태:
        (높이, 너비)

        값의 범위:
        0~100

    point_count : int
        백분위 기준점의 개수

        기본값:
        101개

    Returns
    -------
    tone_curve : dict
        Target 밝기와 Reference 밝기의
        대응 관계를 담은 딕셔너리
    """

    if target_lightness.ndim != 2:
        raise ValueError(
            "target_lightness는 2차원 배열이어야 합니다."
        )


    if reference_lightness.ndim != 2:
        raise ValueError(
            "reference_lightness는 2차원 배열이어야 합니다."
        )


    if point_count < 2:
        raise ValueError(
            "point_count는 2 이상이어야 합니다."
        )


    # 0~100 사이의 백분위 기준 생성
    percentiles = np.linspace(
        0.0,
        100.0,
        point_count,
    )


    # 두 이미지의 같은 백분위 밝기 계산
    target_points = np.percentile(
        target_lightness,
        percentiles,
    )


    reference_points = np.percentile(
        reference_lightness,
        percentiles,
    )


    # 어두운 이미지에서는 같은 밝기값이
    # 여러 번 반복될 수 있다.
    #
    # np.interp()에 넣기 위해
    # Target 기준점을 중복 제거한다.
    target_points_unique, unique_indices = (
        np.unique(
            target_points,
            return_index=True,
        )
    )


    # Target 기준점을 제거한 동일한 위치를
    # Reference에서도 선택한다.
    reference_points_unique = (
        reference_points[unique_indices]
    )


    tone_curve = {
        "percentiles": percentiles,
        "target_points": target_points_unique,
        "reference_points": reference_points_unique,
    }


    return tone_curve


def apply_tone_curve(
    lightness,
    tone_curve,
):
    """
    1D 톤 커브를 밝기 배열에 적용한다.

    Parameters
    ----------
    lightness : np.ndarray
        입력 밝기 배열

        값의 범위:
        0~100

    tone_curve : dict
        create_tone_curve()가 반환한 톤 커브

    Returns
    -------
    mapped_lightness : np.ndarray
        변환된 밝기 배열

        값의 범위:
        0~100
    """

    target_points = (
        tone_curve["target_points"]
    )


    reference_points = (
        tone_curve["reference_points"]
    )


    if len(target_points) < 2:
        raise ValueError(
            "톤 커브 기준점이 충분하지 않습니다."
        )


    # Target 밝기값을 Reference 밝기값으로 변환
    mapped_lightness = np.interp(
        lightness,
        target_points,
        reference_points,
    )


    # 출력 밝기 범위를 0~100으로 제한
    mapped_lightness = np.clip(
        mapped_lightness,
        0.0,
        100.0,
    ).astype(np.float32)


    return mapped_lightness


def apply_tone_curve_to_rgb(
    rgb_image,
    tone_curve,
):
    """
    RGB 이미지에 1D 톤 커브를 적용한다.

    RGB 전체를 직접 바꾸지 않고
    Lab의 L 채널만 변환한다.

    Parameters
    ----------
    rgb_image : np.ndarray
        RGB 이미지

        형태:
        (높이, 너비, 3)

        자료형:
        uint8

    tone_curve : dict
        create_tone_curve()가 반환한 톤 커브

    Returns
    -------
    output_rgb_image : np.ndarray
        톤 커브가 적용된 RGB 이미지

        형태:
        (높이, 너비, 3)

        자료형:
        uint8
    """

    validate_rgb_image(
        rgb_image
    )


    # RGB를 Lab으로 변환
    lab_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2LAB,
    )


    # OpenCV Lab의 L 채널 추출
    lightness_channel = (
        lab_image[:, :, 0]
    )


    # 0~255를 0~100으로 변환
    lightness = (
        lightness_channel.astype(
            np.float32
        )
        / 255.0
        * 100.0
    )


    # 1D 톤 커브 적용
    mapped_lightness = apply_tone_curve(
        lightness=lightness,
        tone_curve=tone_curve,
    )


    # 0~100을 다시 0~255로 변환
    mapped_lightness_channel = (
        mapped_lightness
        / 100.0
        * 255.0
    ).round().astype(np.uint8)


    # 변환된 L 채널로 교체
    output_lab_image = lab_image.copy()

    output_lab_image[:, :, 0] = (
        mapped_lightness_channel
    )


    # Lab을 RGB로 변환
    output_rgb_image = cv2.cvtColor(
        output_lab_image,
        cv2.COLOR_LAB2RGB,
    )


    return output_rgb_image

def apply_tone_curve_to_pixels(
    pixels,
    tone_curve,
):
    """
    (N, 3) 0~1 float RGB 에 1D 톤 커브(Lab L 기준)를 적용한다.
    3D LUT 격자점처럼 이미지가 아닌 색 목록에 쓰기 위한 함수.
    """

    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 3)

    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2Lab)

    lab[:, :, 0] = apply_tone_curve(
        lightness=lab[:, :, 0],
        tone_curve=tone_curve,
    )

    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

    return np.clip(rgb.reshape(-1, 3), 0.0, 1.0)
