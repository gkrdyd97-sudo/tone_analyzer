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
    apply_primaries_to_image,
    estimate_primaries,
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
):
    """
    현재 강도 / 피부 보호 설정으로 내보낼 LUT 3종을 만든다.
    (앱에서 슬라이더를 움직일 때도 이 함수로 즉시 다시 계산한다)

    Returns
    -------
    dict
        color    : 색 단계만 (톤 단계를 거친 이미지에 거는 LUT)
        tone     : 톤 단계(프라이머리)만
        combined : 톤 단계 + 색 단계 (원본 소스에 하나만 거는 LUT)
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

    combined_lut = compose_with_tone_stage(
        tone_stage_nodes=tone_stage_nodes,
        color_lut_output=color_lut,
        lut_size=lut_size,
    )

    return {
        "color": color_lut,
        "tone": np.asarray(tone_stage_nodes, dtype=np.float32),
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
):
    """
    전체 톤 매칭 파이프라인.

    순서 (DaVinci 노드 구조와 같은 흐름)
    ----------------------------------
    1. 톤 단계
       - "primaries": Lift / Gamma / Gain + Contrast / Pivot 추정 (기본)
       - "histogram": Lab L 히스토그램 매칭 1D 커브 (이전 방식)
    2. 색 단계: 톤 단계 결과 -> Reference 색감 (3D LUT)
    3. 피부 보호 / 강도: 색 단계 LUT 에 적용

    내보내는 .cube
    -------------
    - target_to_reference_{N}.cube          : 톤 + 색 (원본 소스에 하나만 걸기)
    - target_to_reference_{N}_tone.cube     : 톤 단계만 (노드 1)
    - target_to_reference_{N}_color.cube    : 색 단계만 (노드 2)
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


    # 3. 톤 단계

    primaries_result = None

    if tone_method == "primaries":

        primaries_result = estimate_primaries(
            target_rgb=target_rgb,
            reference_rgb=reference_rgb,
            balance_strength=balance_strength,
        )

        primaries = primaries_result["primaries"]

        def tone_stage(pixels):
            return apply_primaries(pixels, primaries)

        target_tone_matched_rgb = apply_primaries_to_image(
            target_rgb,
            primaries,
        )

    elif tone_method == "histogram":

        def tone_stage(pixels):
            return apply_tone_curve_to_pixels(pixels, tone_curve)

        height, width = target_rgb.shape[:2]

        target_tone_matched_rgb = (
            tone_stage(target_rgb.reshape(-1, 3) / 255.0)
            .reshape(height, width, 3)
            * 255.0
        ).round().astype(np.uint8)

    else:
        raise ValueError(
            "tone_method는 'primaries' 또는 'histogram' 이어야 합니다."
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

    tone_stage_nodes = tone_stage(lut_input).astype(np.float32)

    final_luts = build_final_luts(
        lut_input=lut_input,
        tone_stage_nodes=tone_stage_nodes,
        color_lut_raw=color_lut_raw,
        lut_size=lut_size,
        transfer_strength=transfer_strength,
        skin_protection=skin_protection,
    )


    # 7. 결과 이미지 (톤 단계 결과에 색 LUT 적용)

    target_3d_lut_rgb = apply_3d_lut(
        rgb_image=target_tone_matched_rgb,
        lut_output=final_luts["color"],
        lut_size=lut_size,
    )


    # 8. 저장

    tone_matched_image_path = output_directory / "target_tone_matched.png"
    final_image_path = output_directory / "target_3d_lut_matched.png"
    cube_path = output_directory / f"target_to_reference_{lut_size}.cube"
    tone_cube_path = output_directory / f"target_to_reference_{lut_size}_tone.cube"
    color_cube_path = output_directory / f"target_to_reference_{lut_size}_color.cube"
    analysis_json_path = output_directory / "analysis_summary.json"

    save_rgb_image(
        rgb_image=target_tone_matched_rgb,
        output_path=tone_matched_image_path,
    )

    save_rgb_image(
        rgb_image=target_3d_lut_rgb,
        output_path=final_image_path,
    )

    save_cube_file(
        lut_output=final_luts["combined"],
        lut_size=lut_size,
        output_path=cube_path,
        title="Target to Reference (tone + color)",
    )

    save_cube_file(
        lut_output=final_luts["tone"],
        lut_size=lut_size,
        output_path=tone_cube_path,
        title=f"Target to Reference (tone: {tone_method})",
    )

    save_cube_file(
        lut_output=final_luts["color"],
        lut_size=lut_size,
        output_path=color_cube_path,
        title="Target to Reference (color only)",
    )


    # 9. 전체 결과

    result = {
        "reference_path": str(reference_path.resolve()),
        "target_path": str(target_path.resolve()),
        "tone_matched_image_path": str(tone_matched_image_path.resolve()),
        "final_image_path": str(final_image_path.resolve()),
        "cube_path": str(cube_path.resolve()),
        "tone_cube_path": str(tone_cube_path.resolve()),
        "color_cube_path": str(color_cube_path.resolve()),
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
        "lut_input": lut_input,
        "tone_stage_nodes": tone_stage_nodes,
        "color_lut_raw": color_lut_raw,
        "lut_output": final_luts["color"],
        "lut_output_combined": final_luts["combined"],
        "transfer_strength": float(transfer_strength),
        "skin_protection": float(skin_protection),
        "skin_ratio": float(pair_data["skin_ratio"]),
        "target_rgb": target_rgb,
        "target_tone_matched_rgb": target_tone_matched_rgb,
        "target_3d_lut_rgb": target_3d_lut_rgb,
    }

    save_analysis_summary(
        pipeline_result=result,
        output_path=analysis_json_path,
    )

    return result
