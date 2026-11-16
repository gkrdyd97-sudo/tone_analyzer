# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

from pathlib import Path

import numpy as np


def build_lut_from_model(
    model,
    lut_size,
):
    """
    학습된 RGB 변환 모델로 3D LUT를 생성한다.

    Parameters
    ----------
    model
        Target RGB를 Reference RGB로 변환하는
        학습된 모델

    lut_size : int
        LUT 한 축의 크기

        예:
        17이면 17 x 17 x 17
        33이면 33 x 33 x 33

    Returns
    -------
    lut_input : np.ndarray
        3D LUT에 입력되는 RGB 좌표

        형태:
        (lut_size ** 3, 3)

    lut_output : np.ndarray
        모델이 계산한 출력 RGB 값

        형태:
        (lut_size ** 3, 3)
    """

    if lut_size < 2:
        raise ValueError(
            "lut_size는 2 이상이어야 합니다."
        )


    # RGB 각 채널에서 사용할 값 생성
    lut_axis = np.linspace(
        0.0,
        1.0,
        lut_size,
        dtype=np.float32,
    )


    # R, G, B의 모든 조합 생성
    red_grid, green_grid, blue_grid = np.meshgrid(
        lut_axis,
        lut_axis,
        lut_axis,
        indexing="ij",
    )


    # 3차원 RGB 격자를
    # (픽셀 개수, RGB 채널) 형태로 변경
    lut_input = np.stack(
        [
            red_grid.reshape(-1),
            green_grid.reshape(-1),
            blue_grid.reshape(-1),
        ],
        axis=1,
    )


    # 학습된 모델로 모든 RGB 격자점 변환
    lut_output = model.predict(
        lut_input,
    )


    # 출력값을 0~1 범위로 제한
    lut_output = np.clip(
        lut_output,
        0.0,
        1.0,
    ).astype(np.float32)


    return lut_input, lut_output


def apply_3d_lut(
    rgb_image,
    lut_output,
    lut_size,
):
    """
    RGB 이미지에 3D LUT를 적용한다.

    Parameters
    ----------
    rgb_image : np.ndarray
        RGB 이미지

        형태:
        (높이, 너비, 3)

        값의 범위:
        0~255

    lut_output : np.ndarray
        3D LUT 출력값

        형태:
        (lut_size ** 3, 3)

        값의 범위:
        0~1

    lut_size : int
        LUT 한 축의 크기

    Returns
    -------
    output_rgb_image : np.ndarray
        LUT가 적용된 RGB 이미지

        형태:
        (높이, 너비, 3)

        자료형:
        uint8
    """

    # 입력 이미지가 3차원인지 확인
    if rgb_image.ndim != 3:
        raise ValueError(
            "RGB 이미지는 3차원 배열이어야 합니다."
        )


    # 마지막 차원이 RGB 3채널인지 확인
    if rgb_image.shape[2] != 3:
        raise ValueError(
            "RGB 이미지의 마지막 차원은 3이어야 합니다."
        )


    # LUT 데이터 개수 확인
    expected_lut_count = lut_size ** 3

    if len(lut_output) != expected_lut_count:
        raise ValueError(
            "lut_output의 개수가 "
            "lut_size와 맞지 않습니다."
        )


    image_height, image_width = rgb_image.shape[:2]

    output_pixels = apply_3d_lut_to_pixels(
        pixels=rgb_image.reshape(-1, 3).astype(np.float32) / 255.0,
        lut_output=lut_output,
        lut_size=lut_size,
    )


    # 원래 이미지 형태로 복원
    output_rgb_image = output_pixels.reshape(
        image_height,
        image_width,
        3,
    )


    # 0~255 범위의 uint8 이미지로 변환
    output_rgb_image = (
        output_rgb_image * 255.0
    ).round().astype(np.uint8)


    return output_rgb_image


def save_cube_file(
    lut_output,
    lut_size,
    output_path,
    title="Tone Analyzer 3D LUT",
):
    """
    3D LUT를 .cube 파일로 저장한다.

    Parameters
    ----------
    lut_output : np.ndarray
        LUT 출력 RGB 값

        형태:
        (lut_size ** 3, 3)

        값의 범위:
        0~1

    lut_size : int
        LUT 한 축의 크기

    output_path : str 또는 Path
        저장할 .cube 파일 경로

    title : str
        LUT 파일 제목
    """

    # LUT 데이터 개수 확인
    expected_lut_count = lut_size ** 3

    if len(lut_output) != expected_lut_count:
        raise ValueError(
            "lut_output의 개수가 "
            "lut_size와 맞지 않습니다."
        )


    # Path 객체로 변환
    output_path = Path(output_path)


    # 저장 폴더가 없으면 생성
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # .cube 파일의 헤더 생성
    cube_lines = []

    cube_lines.append(
        f'TITLE "{title}"'
    )

    cube_lines.append(
        f"LUT_3D_SIZE {lut_size}"
    )

    cube_lines.append(
        "DOMAIN_MIN 0.0 0.0 0.0"
    )

    cube_lines.append(
        "DOMAIN_MAX 1.0 1.0 1.0"
    )


    # .cube 표준은 R이 가장 빠르게 변하는 순서(R -> G -> B)다.
    # 내부 LUT는 indexing="ij"(B가 가장 빠름)이므로 축 순서를 바꿔 저장한다.
    cube_ordered_output = (
        np.asarray(lut_output)
        .reshape(lut_size, lut_size, lut_size, 3)
        .transpose(2, 1, 0, 3)
        .reshape(-1, 3)
    )

    # LUT 출력값을 한 줄씩 저장
    for red, green, blue in cube_ordered_output:

        cube_lines.append(
            f"{red:.6f} "
            f"{green:.6f} "
            f"{blue:.6f}"
        )


    # 줄바꿈으로 전체 파일 내용 결합
    cube_text = "\n".join(
        cube_lines
    ) + "\n"


    # 파일 저장
    output_path.write_text(
        cube_text,
        encoding="utf-8",
    )


    return output_path

def blend_lut_with_identity(
    lut_input,
    lut_output,
    strength,
):
    """
    LUT 강도를 조절한다. (재학습 없이 실시간 처리)

    strength = 0 이면 원본 그대로(identity),
    strength = 1 이면 학습된 LUT 그대로.
    """

    strength = float(np.clip(strength, 0.0, 1.0))

    blended = (
        np.asarray(lut_input, dtype=np.float32) * (1.0 - strength)
        + np.asarray(lut_output, dtype=np.float32) * strength
    )

    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def make_cube_text(
    lut_output,
    lut_size,
    title="Tone Analyzer 3D LUT",
):
    """
    .cube 파일 내용을 문자열로 만든다. (파일 저장 없이 다운로드용)
    """

    cube_ordered_output = (
        np.asarray(lut_output)
        .reshape(lut_size, lut_size, lut_size, 3)
        .transpose(2, 1, 0, 3)
        .reshape(-1, 3)
    )

    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {lut_size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]

    lines.extend(
        f"{red:.6f} {green:.6f} {blue:.6f}"
        for red, green, blue in cube_ordered_output
    )

    return "\n".join(lines) + "\n"


def _dilate_3d(volume):
    """3x3x3 최대값 필터 (scipy 없이 numpy로 구현)"""

    padded = np.pad(volume, 1, mode="edge")
    size_r, size_g, size_b = volume.shape
    result = volume.copy()

    for dr in range(3):
        for dg in range(3):
            for db in range(3):
                result = np.maximum(
                    result,
                    padded[
                        dr:dr + size_r,
                        dg:dg + size_g,
                        db:db + size_b,
                    ],
                )

    return result


def protect_skin_in_lut(
    lut_input,
    lut_output,
    skin_protection,
):
    """
    LUT 격자점 단위로 피부톤을 보호한다.

    각 격자점의 '입력 색'이 피부색일 가능성(0~1)을 계산하고,
    그 비율 * skin_protection 만큼 출력을 원래 색(identity)으로 되돌린다.

    학습 단계가 아니라 LUT에 직접 적용하므로
    - 다항식 모델의 평활화 때문에 효과가 흐려지지 않고
    - .cube 파일에도 그대로 반영되며
    - 재학습 없이 실시간으로 조절할 수 있다.
    """

    from .color_matching import compute_skin_weights

    protection = float(np.clip(skin_protection, 0.0, 1.0))

    lut_input = np.asarray(lut_input, dtype=np.float32)
    lut_output = np.asarray(lut_output, dtype=np.float32)

    if protection <= 0.0:
        return lut_output.copy()

    lut_size = int(round(len(lut_input) ** (1.0 / 3.0)))

    skin_weights = compute_skin_weights(lut_input).reshape(
        lut_size, lut_size, lut_size
    )

    # 피부색은 RGB 공간에서 얇은 영역이라, 픽셀이 속한 격자 칸의
    # 8개 꼭짓점 중 일부만 피부로 판정되면 보간 과정에서 보호가 흐려진다.
    # 그래서 가중치를 이웃 격자로 한 칸 넓힌다(3x3x3 최대값 필터).
    skin_weights = _dilate_3d(skin_weights).reshape(-1)

    keep_original = (
        protection * skin_weights
    )[:, None]

    protected = (
        lut_output * (1.0 - keep_original)
        + lut_input * keep_original
    )

    return np.clip(protected, 0.0, 1.0).astype(np.float32)


def apply_3d_lut_to_pixels(
    pixels,
    lut_output,
    lut_size,
):
    """
    (N, 3) 0~1 float RGB 목록에 3D LUT를 삼선형 보간으로 적용한다.
    LUT끼리 합성할 때(격자점 -> 다른 LUT)도 사용한다.
    """

    pixels = np.clip(
        np.asarray(pixels, dtype=np.float32),
        0.0,
        1.0,
    )







    # LUT를 다시 3차원 구조로 복원
    lut_cube = np.asarray(lut_output, dtype=np.float32).reshape(
        lut_size,
        lut_size,
        lut_size,
        3,
    )


    # 각 픽셀이 LUT 공간에서 어느 위치에 있는지 계산
    lut_position = (
        pixels * (lut_size - 1)
    )


    # 주변 격자의 아래쪽 인덱스
    lower_index = np.floor(
        lut_position
    ).astype(np.int32)


    # 주변 격자의 위쪽 인덱스
    upper_index = lower_index + 1


    # 인덱스가 LUT 범위를 벗어나지 않도록 제한
    lower_index = np.clip(
        lower_index,
        0,
        lut_size - 1,
    )

    upper_index = np.clip(
        upper_index,
        0,
        lut_size - 1,
    )


    # 현재 픽셀이 격자 사이에서
    # 어느 위치에 있는지 계산
    interpolation_weight = (
        lut_position - lower_index
    )


    # R, G, B의 아래쪽 인덱스
    r0 = lower_index[:, 0]
    g0 = lower_index[:, 1]
    b0 = lower_index[:, 2]


    # R, G, B의 위쪽 인덱스
    r1 = upper_index[:, 0]
    g1 = upper_index[:, 1]
    b1 = upper_index[:, 2]


    # R, G, B 방향의 보간 가중치
    wr = interpolation_weight[:, 0]
    wg = interpolation_weight[:, 1]
    wb = interpolation_weight[:, 2]


    # 현재 픽셀 주변의 8개 LUT 값 가져오기
    c000 = lut_cube[r0, g0, b0]
    c001 = lut_cube[r0, g0, b1]
    c010 = lut_cube[r0, g1, b0]
    c011 = lut_cube[r0, g1, b1]

    c100 = lut_cube[r1, g0, b0]
    c101 = lut_cube[r1, g0, b1]
    c110 = lut_cube[r1, g1, b0]
    c111 = lut_cube[r1, g1, b1]


    # 3차원 선형 보간
    output_pixels = (
        c000
        * (1 - wr)[:, None]
        * (1 - wg)[:, None]
        * (1 - wb)[:, None]

        + c001
        * (1 - wr)[:, None]
        * (1 - wg)[:, None]
        * wb[:, None]

        + c010
        * (1 - wr)[:, None]
        * wg[:, None]
        * (1 - wb)[:, None]

        + c011
        * (1 - wr)[:, None]
        * wg[:, None]
        * wb[:, None]

        + c100
        * wr[:, None]
        * (1 - wg)[:, None]
        * (1 - wb)[:, None]

        + c101
        * wr[:, None]
        * (1 - wg)[:, None]
        * wb[:, None]

        + c110
        * wr[:, None]
        * wg[:, None]
        * (1 - wb)[:, None]

        + c111
        * wr[:, None]
        * wg[:, None]
        * wb[:, None]
    )


    # RGB 결과를 0~1 범위로 제한
    output_pixels = np.clip(
        output_pixels,
        0.0,
        1.0,
    )


    return np.clip(output_pixels, 0.0, 1.0).astype(np.float32)


def compose_with_tone_stage(
    tone_stage_nodes,
    color_lut_output,
    lut_size,
):
    """
    [톤 단계 -> 색 LUT] 를 하나의 3D LUT로 합친다.

    tone_stage_nodes : 각 격자점 입력 색에 톤 단계(프라이머리 등)를
                       적용한 결과. 형태 (lut_size**3, 3)

    원본 소스에 이 LUT 하나만 걸면 앱 결과와 같은 색이 나온다.
    """

    return apply_3d_lut_to_pixels(
        pixels=tone_stage_nodes,
        lut_output=color_lut_output,
        lut_size=lut_size,
    )
