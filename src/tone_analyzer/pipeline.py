# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

from pathlib import Path

import numpy as np

from .analysis import (
    calculate_contrast,
    analyze_all_tonal_regions,
    extract_lightness,
    make_tonal_masks,
)

from .color_matching import (
    evaluate_rgb_model,
    prepare_pixel_pairs,
    train_rgb_model,
)

from .image_io import (
    load_rgb_image,
    save_rgb_image,
)

from .lut import (
    apply_3d_lut,
    blend_lut_with_identity,
    build_lut_from_model,
    compose_with_tone_stage,
    protect_skin_in_lut,
    save_cube_file,
)

from .primaries import (
    apply_primaries,
    estimate_primaries,
)

from .skin_tone import (
    SKIN_LINE_HUE,
    build_skin_correction_lut,
    measure_skin,
    resolve_target_hue,
)

from .white_balance import (
    apply_white_balance,
    estimate_white_balance,
)

from .result_io import (
    save_analysis_summary,
)

from .tone_curve import (
    apply_tone_curve_to_pixels,
    create_tone_curve,
)


def build_final_luts(
    lut_input,
    tone_stage_nodes,
    color_lut_raw,
    lut_size,
    transfer_strength,
    skin_protection,
    primaries_nodes=None,
    white_balance_nodes=None,
    skin_target_hue=SKIN_LINE_HUE,
    skin_hue_strength=0.0,
    skin_saturation=1.0,
    skin_chroma=None,
):
    """
    현재 강도 / 피부 보호 설정으로 내보낼 LUT 3종을 만든다.
    (앱에서 슬라이더를 움직일 때도 이 함수로 즉시 다시 계산한다)

    Returns
    -------
    dict
        primaries     : ①② LGG + Contrast/Pivot 만        (노드 1)
        white_balance : ③ 화이트밸런스만                    (노드 2)
        tone          : ①②③ 를 합친 것
        color         : ④ 컬러 그레이딩 (+ 피부 보호, 강도) (노드 3)
        skin          : ⑤ 스킨톤 교정                       (노드 4)
        combined      : ①~⑤ 전부 (원본 소스에 하나만 거는 LUT)
    """

    protected = protect_skin_in_lut(
        lut_input=lut_input,
        lut_output=color_lut_raw,
        skin_protection=skin_protection,
    )

    color_lut = blend_lut_with_identity(
        lut_input=lut_input,
        lut_output=protected,
        strength=transfer_strength,
    )

    skin_lut = build_skin_correction_lut(
        lut_input=lut_input,
        target_hue=skin_target_hue,
        hue_strength=skin_hue_strength,
        saturation=skin_saturation,
        skin_chroma=skin_chroma,
    )

    # ①②③ → ④ → ⑤ 를 하나로 합친다.
    combined_lut = compose_with_tone_stage(
        tone_stage_nodes=compose_with_tone_stage(
            tone_stage_nodes=tone_stage_nodes,
            color_lut_output=color_lut,
            lut_size=lut_size,
        ),
        color_lut_output=skin_lut,
        lut_size=lut_size,
    )

    identity = np.asarray(lut_input, dtype=np.float32)

    return {
        "primaries": (
            np.asarray(primaries_nodes, dtype=np.float32)
            if primaries_nodes is not None else identity
        ),
        "white_balance": (
            np.asarray(white_balance_nodes, dtype=np.float32)
            if white_balance_nodes is not None else identity
        ),
        "tone": np.asarray(tone_stage_nodes, dtype=np.float32),
        "color": color_lut,
        "skin": skin_lut,
        "combined": combined_lut,
    }


def run_tone_matching_pipeline(
    reference_path,
    target_path,
    output_directory,
    lut_size=33,
    max_sample_count=100_000,
    n_neighbors=64,
    color_method="reinhard",
    transfer_strength=0.8,
    preserve_luminance=True,
    tonal_regions=True,
    skin_protection=0.85,
    tone_method="primaries",
    balance_strength=0.5,
    skin_target_mode="line",
    skin_hue_strength=0.5,
    skin_saturation=1.0,
):
    """
    전체 톤 매칭 파이프라인.

    순서 (컬러리스트 작업 순서 / DaVinci 노드 구조)
    ---------------------------------------------
    ① Lift / Gamma / Gain   ┐ tone_method="primaries" (기본)
    ② Contrast / Pivot      ┘ tone_method="histogram" 이면 히스토그램 매칭 커브로 대체
    ③ 화이트밸런스 (Temp / Tint, 강도 = balance_strength 0~1)
    ④ 컬러 그레이딩 (Reference 색감, 3D LUT) + 피부 보호 / 강도
    ⑤ 스킨톤 교정 (피부 색상을 스킨톤 라인 또는 Reference 피부 쪽으로, 채도 조절)
       skin_target_mode = "line" | "reference"

    내보내는 .cube
    -------------
    - target_to_reference_{N}.cube             : ①~⑤ 전부 (원본 소스에 하나만 걸기)
    - target_to_reference_{N}_1_primaries.cube : ①② (노드 1)
    - target_to_reference_{N}_2_wb.cube        : ③   (노드 2)
    - target_to_reference_{N}_3_color.cube     : ④   (노드 3)
    - target_to_reference_{N}_4_skin.cube      : ⑤   (노드 4)
    """

    reference_path = Path(reference_path)
    target_path = Path(target_path)
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    # 1. 이미지 읽기

    reference_rgb = load_rgb_image(reference_path)
    target_rgb = load_rgb_image(target_path)


    # 2. 밝기 분석

    reference_lightness = extract_lightness(reference_rgb)
    target_lightness = extract_lightness(target_rgb)

    reference_masks = make_tonal_masks(reference_lightness)
    target_masks = make_tonal_masks(target_lightness)

    reference_regions = analyze_all_tonal_regions(
        lightness=reference_lightness,
        masks=reference_masks,
    )

    target_regions = analyze_all_tonal_regions(
        lightness=target_lightness,
        masks=target_masks,
    )

    reference_contrast = calculate_contrast(
        lightness=reference_lightness,
        masks=reference_masks,
    )

    target_contrast = calculate_contrast(
        lightness=target_lightness,
        masks=target_masks,
    )

    # 히스토그램 커브는 분석용으로 항상 계산해 둔다.
    tone_curve = create_tone_curve(
        target_lightness=target_lightness,
        reference_lightness=reference_lightness,
        point_count=101,
    )


    # 3. 톤 단계: ① LGG → ② Contrast/Pivot (또는 히스토그램 커브)

    primaries_result = None

    if tone_method == "primaries":

        primaries_result = estimate_primaries(
            target_rgb=target_rgb,
            reference_rgb=reference_rgb,
        )

        primaries = primaries_result["primaries"]

        def primaries_stage(pixels):
            return apply_primaries(pixels, primaries)

    elif tone_method == "histogram":

        def primaries_stage(pixels):
            return apply_tone_curve_to_pixels(pixels, tone_curve)

    else:
        raise ValueError(
            "tone_method는 'primaries' 또는 'histogram' 이어야 합니다."
        )

    height, width = target_rgb.shape[:2]

    def apply_stage_to_image(stage, rgb_image):
        output = stage(rgb_image.reshape(-1, 3).astype(np.float32) / 255.0)
        return (
            np.asarray(output).reshape(height, width, 3) * 255.0
        ).round().astype(np.uint8)

    target_primaries_rgb = apply_stage_to_image(
        primaries_stage,
        target_rgb,
    )


    # ③ 화이트밸런스 (①② 결과 기준으로 추정)

    white_balance_result = estimate_white_balance(
        target_rgb=target_primaries_rgb,
        reference_rgb=reference_rgb,
        strength=balance_strength,
    )

    white_balance = white_balance_result["white_balance"]

    def white_balance_stage(pixels):
        return apply_white_balance(pixels, white_balance)

    def tone_stage(pixels):
        return white_balance_stage(primaries_stage(pixels))

    target_tone_matched_rgb = apply_stage_to_image(
        white_balance_stage,
        target_primaries_rgb,
    )


    # 4. 색 단계 학습 쌍 생성
    #    (같은 좌표 픽셀끼리 짝짓지 않고,
    #     target 색 분포를 reference 분포로 전이한 결과를 목표값으로 사용)

    pair_data = prepare_pixel_pairs(
        target_rgb=target_tone_matched_rgb,
        reference_rgb=reference_rgb,
        max_sample_count=max_sample_count,
        random_seed=42,
        # 강도 1.0(완전 전이)으로 학습하고,
        # 실제 강도는 LUT 단계에서 identity 와 섞어 조절한다.
        transfer_strength=1.0,
        method=color_method,
        preserve_luminance=preserve_luminance,
        tonal_regions=tonal_regions,
        # 피부 보호는 학습이 아니라 LUT 격자점에 직접 적용한다.
        skin_protection=0.0,
    )

    target_samples = pair_data["target_samples"]
    reference_samples = pair_data["reference_samples"]


    # 5. 색 모델 평가 / 학습

    evaluation_result = evaluate_rgb_model(
        target_samples=target_samples,
        reference_samples=reference_samples,
        n_neighbors=n_neighbors,
        test_size=0.2,
        random_seed=42,
    )

    rgb_model = train_rgb_model(
        target_samples=target_samples,
        reference_samples=reference_samples,
        n_neighbors=n_neighbors,
    )


    # 6. LUT 생성

    lut_input, color_lut_raw = build_lut_from_model(
        model=rgb_model,
        lut_size=lut_size,
    )

    primaries_nodes = np.asarray(
        primaries_stage(lut_input), dtype=np.float32
    )
    white_balance_nodes = np.asarray(
        white_balance_stage(lut_input), dtype=np.float32
    )
    tone_stage_nodes = np.asarray(
        tone_stage(lut_input), dtype=np.float32
    )

    # ⑤ 스킨톤 교정 목표
    #    교정 범위(퀄리파이어)는 ④ 까지 적용한 이미지의 실제 피부 채도를 기준으로 한다.
    graded_luts = build_final_luts(
        lut_input=lut_input,
        tone_stage_nodes=tone_stage_nodes,
        color_lut_raw=color_lut_raw,
        lut_size=lut_size,
        transfer_strength=transfer_strength,
        skin_protection=skin_protection,
    )

    target_skin = measure_skin(
        apply_3d_lut(
            rgb_image=target_tone_matched_rgb,
            lut_output=graded_luts["color"],
            lut_size=lut_size,
        )
    )

    reference_skin = measure_skin(reference_rgb)
    skin_target_hue, skin_target_used = resolve_target_hue(
        skin_target_mode,
        reference_skin,
    )

    final_luts = build_final_luts(
        lut_input=lut_input,
        tone_stage_nodes=tone_stage_nodes,
        color_lut_raw=color_lut_raw,
        lut_size=lut_size,
        transfer_strength=transfer_strength,
        skin_protection=skin_protection,
        primaries_nodes=primaries_nodes,
        white_balance_nodes=white_balance_nodes,
        skin_target_hue=skin_target_hue,
        skin_hue_strength=skin_hue_strength,
        skin_saturation=skin_saturation,
        skin_chroma=target_skin["chroma"],
    )


    # 7. 결과 이미지 (톤 단계 결과에 색 LUT 적용)

    target_graded_rgb = apply_3d_lut(
        rgb_image=target_tone_matched_rgb,
        lut_output=final_luts["color"],
        lut_size=lut_size,
    )

    target_3d_lut_rgb = apply_3d_lut(
        rgb_image=target_graded_rgb,
        lut_output=final_luts["skin"],
        lut_size=lut_size,
    )


    # 8. 저장

    tone_matched_image_path = output_directory / "target_tone_matched.png"
    final_image_path = output_directory / "target_3d_lut_matched.png"
    analysis_json_path = output_directory / "analysis_summary.json"

    cube_paths = {
        "combined": output_directory / f"target_to_reference_{lut_size}.cube",
        "primaries": output_directory / f"target_to_reference_{lut_size}_1_primaries.cube",
        "white_balance": output_directory / f"target_to_reference_{lut_size}_2_wb.cube",
        "color": output_directory / f"target_to_reference_{lut_size}_3_color.cube",
        "skin": output_directory / f"target_to_reference_{lut_size}_4_skin.cube",
    }

    cube_titles = {
        "combined": "Target to Reference (1-5 all)",
        "primaries": f"Target to Reference (1-2 primaries: {tone_method})",
        "white_balance": "Target to Reference (3 white balance)",
        "color": "Target to Reference (4 color grading)",
        "skin": "Target to Reference (5 skin tone)",
    }

    save_rgb_image(
        rgb_image=target_tone_matched_rgb,
        output_path=tone_matched_image_path,
    )

    save_rgb_image(
        rgb_image=target_3d_lut_rgb,
        output_path=final_image_path,
    )

    for key, path in cube_paths.items():
        save_cube_file(
            lut_output=final_luts[key],
            lut_size=lut_size,
            output_path=path,
            title=cube_titles[key],
        )


    # 9. 전체 결과

    result = {
        "reference_path": str(reference_path.resolve()),
        "target_path": str(target_path.resolve()),
        "tone_matched_image_path": str(tone_matched_image_path.resolve()),
        "final_image_path": str(final_image_path.resolve()),
        "cube_path": str(cube_paths["combined"].resolve()),
        "cube_paths": {
            key: str(path.resolve()) for key, path in cube_paths.items()
        },
        "analysis_json_path": str(analysis_json_path.resolve()),
        "lut_size": int(lut_size),
        "sample_count": int(len(target_samples)),
        "mean_absolute_error": float(
            evaluation_result["mean_absolute_error"]
        ),
        "mean_absolute_error_8bit": float(
            evaluation_result["mean_absolute_error_8bit"]
        ),
        "reference_regions": reference_regions,
        "target_regions": target_regions,
        "reference_contrast": reference_contrast,
        "target_contrast": target_contrast,
        "tone_curve": tone_curve,
        "tone_method": tone_method,
        "primaries": (
            primaries_result["primaries"] if primaries_result else None
        ),
        "primaries_diagnostics": (
            primaries_result["diagnostics"] if primaries_result else None
        ),
        "white_balance": white_balance,
        "reference_skin": reference_skin,
        "target_skin": target_skin,
        "skin_line_hue": float(SKIN_LINE_HUE),
        "skin_target_mode": skin_target_mode,
        "skin_target_used": skin_target_used,
        "skin_target_hue": float(skin_target_hue),
        "skin_hue_strength": float(skin_hue_strength),
        "skin_saturation": float(skin_saturation),
        "white_balance_diagnostics": white_balance_result["diagnostics"],
        "primaries_nodes": primaries_nodes,
        "white_balance_nodes": white_balance_nodes,
        "lut_input": lut_input,
        "tone_stage_nodes": tone_stage_nodes,
        "color_lut_raw": color_lut_raw,
        "lut_output": final_luts["color"],
        "lut_output_combined": final_luts["combined"],
        "transfer_strength": float(transfer_strength),
        "skin_protection": float(skin_protection),
        "skin_ratio": float(pair_data["skin_ratio"]),
        "target_rgb": target_rgb,
        "target_primaries_rgb": target_primaries_rgb,
        "target_tone_matched_rgb": target_tone_matched_rgb,
        "target_3d_lut_rgb": target_3d_lut_rgb,
    }

    save_analysis_summary(
        pipeline_result=result,
        output_path=analysis_json_path,
    )

    return result
