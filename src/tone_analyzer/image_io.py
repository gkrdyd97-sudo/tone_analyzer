from pathlib import Path

import cv2
import numpy as np


def validate_image_path(
    image_path,
):
    """
    이미지 경로가 올바른지 확인한다.

    Parameters
    ----------
    image_path : str 또는 Path
        확인할 이미지 파일 경로

    Returns
    -------
    path : Path
        확인이 끝난 Path 객체
    """

    path = Path(image_path)


    if not path.exists():
        raise FileNotFoundError(
            f"이미지 파일이 존재하지 않습니다: {path}"
        )


    if not path.is_file():
        raise ValueError(
            f"이미지 경로가 파일이 아닙니다: {path}"
        )


    return path


def load_rgb_image(
    image_path,
):
    """
    이미지 파일을 읽어서 RGB NumPy 배열로 반환한다.

    Parameters
    ----------
    image_path : str 또는 Path
        이미지 파일 경로

    Returns
    -------
    rgb_image : np.ndarray
        RGB 이미지 배열

        형태:
        (높이, 너비, 3)

        자료형:
        uint8

        값의 범위:
        0~255
    """

    path = validate_image_path(
        image_path
    )


    # OpenCV는 기본적으로 BGR 순서로 이미지를 읽는다.
    bgr_image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )


    if bgr_image is None:
        raise ValueError(
            f"이미지를 읽을 수 없습니다: {path}"
        )


    # OpenCV의 BGR 순서를 RGB 순서로 변환한다.
    rgb_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2RGB,
    )


    return rgb_image


def validate_rgb_image(
    rgb_image,
):
    """
    RGB 이미지 배열의 형태와 자료형을 확인한다.

    Parameters
    ----------
    rgb_image : np.ndarray
        확인할 RGB 이미지 배열

    Returns
    -------
    None
    """

    if not isinstance(
        rgb_image,
        np.ndarray,
    ):
        raise TypeError(
            "이미지는 NumPy 배열이어야 합니다."
        )


    if rgb_image.ndim != 3:
        raise ValueError(
            "이미지는 높이, 너비, 채널의 "
            "3차원 배열이어야 합니다."
        )


    if rgb_image.shape[2] != 3:
        raise ValueError(
            "RGB 이미지의 채널 수는 3이어야 합니다."
        )


    if rgb_image.dtype != np.uint8:
        raise TypeError(
            "현재 함수는 uint8 이미지를 사용합니다."
        )


def save_rgb_image(
    rgb_image,
    output_path,
):
    """
    RGB NumPy 배열을 이미지 파일로 저장한다.

    Parameters
    ----------
    rgb_image : np.ndarray
        저장할 RGB 이미지

        형태:
        (높이, 너비, 3)

        자료형:
        uint8

        값의 범위:
        0~255

    output_path : str 또는 Path
        저장할 파일 경로

    Returns
    -------
    output_path : Path
        저장된 파일 경로
    """

    validate_rgb_image(
        rgb_image
    )


    output_path = Path(
        output_path
    )


    # 출력 폴더가 없으면 자동으로 생성한다.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # OpenCV로 저장하기 전에 RGB를 BGR로 변환한다.
    bgr_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )


    saved = cv2.imwrite(
        str(output_path),
        bgr_image,
    )


    if not saved:
        raise IOError(
            f"이미지를 저장하지 못했습니다: "
            f"{output_path}"
        )


    return output_path