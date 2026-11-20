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

from .lut import apply_3d_lut_to_pixels

from .letterbox import remove_letterbox

from .saturation import apply_saturation, estimate_saturation

from .skin_tone import (
    MAX_SKIN_LIGHTNESS_SCALE,
    SKIN_LINE_HUE,
    build_skin_correction_lut,
    detect_faces,
    measure_skin_samples,
    resolve_target_hue,
    sample_face_pixels,
    skin_cluster_from_pixels,
    skin_leakage,
    srgb_to_lab,
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
    skin_samples=None,
    skin_enabled=True,
    saturation_nodes=None,
    skin_target_lightness=None,
    skin_lightness_strength=0.0,
):
    """
    현재 강도 / 피부 보호 설정으로 내보낼 LUT 를 만든다.

    skin_enabled : False 면 ④ 피부 보호와 ⑤ 교정을 건너뛴다.
        (사람이 없는 장면에서 피부와 비슷한 색의 사물을 피부로 처리하지 않도록)

    skin_samples : ①②③ 까지 적용한 이미지의 얼굴 피부 픽셀 (N, 3) 0~1.
        - ④ 피부 보호는 이 색 범위만 보호하고
        - ⑤ 교정은 이 샘플을 ④ 에 통과시킨 '그레이딩 후 피부' 색 범위만 교정한다.
    (앱에서 슬라이더를 움직일 때도 이 함수로 즉시 다시 계산한다)

    Returns
    -------
    dict
        primaries     : ① 프라이머리 (Lift→Gain→Gamma→Contrast)  (노드 1)
        white_balance : ② 화이트밸런스                          (노드 2)
        saturation    : ③ 채도                                  (노드 3)
        tone          : ①②③ 를 합친 것
        color         : ④ 색작업 (+ 피부 보호, 강도)            (노드 4)
        skin          : ⑤ 피부톤 (퀄리파이어: 색상·채도·밝기)    (노드 5)
        combined      : ①~⑤ 전부 (원본 소스에 하나만 거는 LUT)
        skin_hue_before / skin_hue_after             : ⑤ 전후 피부 색상각 (샘플 기준)
        skin_lightness_before / skin_lightness_after : ⑤ 전후 피부 밝기 (Lab L)

    skin_target_lightness : Reference 얼굴 피부 밝기(Lab L). 있으면 ⑤ 에서 피부 밝기도 맞춘다.
    skin_lightness_strength : 0~1
    """

    has_samples = (
        skin_enabled
        and skin_samples is not None
        and len(skin_samples) >= 50
    )

    if not skin_enabled:
        skin_protection = 0.0
        skin_hue_strength = 0.0
        skin_saturation = 1.0
        skin_lightness_strength = 0.0

    pre_grade_cluster = (
        skin_cluster_from_pixels(skin_samples) if has_samples else None
    )

    protected = protect_skin_in_lut(
        lut_input=lut_input,
        lut_output=color_lut_raw,
        skin_protection=skin_protection,
        skin_cluster=pre_grade_cluster,
    )

    color_lut = blend_lut_with_identity(
        lut_input=lut_input,
        lut_output=protected,
        strength=transfer_strength,
    )

    # ④ 를 통과한 뒤 피부가 어디로 갔는지 → ⑤ 의 퀄리파이어
    graded_samples = (
        apply_3d_lut_to_pixels(skin_samples, color_lut, lut_size)
        if has_samples else None
    )

    graded_cluster = (
        skin_cluster_from_pixels(graded_samples) if has_samples else None
    )

    def samples_lightness(samples):
        if samples is None:
            return None
        return float(np.median(srgb_to_lab(samples)[:, 0]))

    graded_lightness = samples_lightness(graded_samples)

    # 피부 밝기: 비율로, 강도만큼, 최대 ±12% 까지만 (큰 차이는 ① 프라이머리의 몫)
    lightness_scale = 0.0
    if skin_target_lightness is not None and graded_lightness:
        ratio = float(skin_target_lightness) / graded_lightness
        lightness_scale = float(np.clip(
            ratio ** float(np.clip(skin_lightness_strength, 0.0, 1.0)) - 1.0,
            -MAX_SKIN_LIGHTNESS_SCALE,
            MAX_SKIN_LIGHTNESS_SCALE,
        ))

    skin_lut = build_skin_correction_lut(
        lut_input=lut_input,
        target_hue=skin_target_hue,
        hue_strength=skin_hue_strength,
        saturation=skin_saturation,
        skin_chroma=skin_chroma,
        skin_cluster=graded_cluster,
        lightness_scale=lightness_scale,
    )

    def samples_hue(samples):
        if samples is None:
            return None
        lab = srgb_to_lab(samples)
        return float(np.degrees(np.arctan2(
            np.median(lab[:, 2]), np.median(lab[:, 1])
        )))

    corrected_samples = (
        apply_3d_lut_to_pixels(graded_samples, skin_lut, lut_size)
        if has_samples else None
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
        "saturation": (
            np.asarray(saturation_nodes, dtype=np.float32)
            if saturation_nodes is not None else identity
        ),
        "tone": np.asarray(tone_stage_nodes, dtype=np.float32),
        "color": color_lut,
        "skin": skin_lut,
        "combined": combined_lut,
        "skin_hue_before": samples_hue(graded_samples),
        "skin_hue_after": samples_hue(corrected_samples),
        "skin_lightness_before": graded_lightness,
        "skin_lightness_after": samples_lightness(corrected_samples),
        "skin_lightness_scale": lightness_scale,
        "skin_cluster_graded": graded_cluster,
    }


def resolve_skin_processing(
    face_samples,
    color_samples,
    skin_without_face=False,
):
    """
    ④⑤ 피부 처리를 할지, 어떤 피부 샘플을 쓸지 정한다.

    - 얼굴 샘플이 있으면: 처리 O, 얼굴 샘플 사용
    - 얼굴이 없고 skin_without_face=False (기본): 처리 X
      (사람이 없는 장면의 나무·모래·노을을 피부로 처리하지 않도록)
    - 얼굴이 없고 skin_without_face=True: 색 기준으로 추정한 피부로 처리

    Returns
    -------
    (skin_enabled, samples)
    """

    if face_samples is not None and len(face_samples) >= 50:
        return True, face_samples

    if skin_without_face and color_samples is not None and len(color_samples) >= 50:
        return True, color_samples

    return False, None


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
    balance_strength=0.7,
    white_balance_mode="neutral",
    skin_target_mode="reference",
    skin_hue_strength=0.5,
    skin_saturation=1.0,
    skin_without_face=False,
    auto_crop=True,
    tone_strength=1.0,
    tone_safe_limits=True,
    tone_follow=0.0,
    match_mode="look",
    saturation_strength=1.0,
    skin_lightness_strength=0.5,
):
    """
    전체 톤 매칭 파이프라인.

    match_mode
    ----------
    "look"       (기본) : 다른 장면의 룩을 옮긴다. ①② 는 안전 범위 안에서만.
    "same_scene"        : 같은 장면의 다른 컷 맞추기(리버스 숏 등).
                          두 컷은 이미 같은 노출·그레이딩이라 밝기 분포 차이는 '내용 차이'다.
                          → ①② 끔(tone_strength=0), ③ 은 Reference 중성색에 맞춤(강도 1).

    작업 순서 (시니어 컬러리스트의 노드 순서)
    -----------------------------------------
    ⓪ 레터박스 제거
    ① 프라이머리: Lift(블랙) → Gain(화이트) → 보간 → Gamma(중앙값) → Contrast/Pivot   (노드 1)
    ② 화이트밸런스                                                                   (노드 2)
    ③ 채도 (두 이미지에 얼굴이 있으면 피부 채도 기준)                                   (노드 3)
    ④ 색작업 (톤 영역별 색, 피부 보호)                                                 (노드 4)
    ⑤ 피부톤: 퀄리파이어로 피부만 잡아 Reference 피부의 색상·밝기에 맞춤               (노드 5)

    auto_crop : 레퍼런스·타깃의 레터박스/필러박스를 잘라내고 분석한다. (결과 이미지도 잘린 크기)
    saturation_strength : ③ 채도 강도
    skin_lightness_strength : ⑤ 피부 밝기 맞춤 강도 (Reference 얼굴이 있을 때)
    tone_strength / tone_safe_limits : ①② 강도와 안전 범위 (primaries.estimate_primaries)

    순서 (컬러리스트 작업 순서 / DaVinci 노드 구조)
    ---------------------------------------------
    ① Lift / Gamma / Gain   ┐ tone_method="primaries" (기본)
    ② Contrast / Pivot      ┘ tone_method="histogram" 이면 히스토그램 매칭 커브로 대체
    ③ 화이트밸런스 (Temp / Tint, 강도 = balance_strength 0~1)
       white_balance_mode = "neutral" (중성색을 회색으로) | "reference" (Reference 중성색에 맞춤)
    ④ 컬러 그레이딩 (Reference 색감, 3D LUT) + 피부 보호 / 강도
    ⑤ 스킨톤 교정 (피부 색상을 스킨톤 라인 또는 Reference 피부 쪽으로, 채도 조절)
       skin_target_mode = "line" | "reference"

    얼굴이 없으면 ④ 피부 보호와 ⑤ 교정은 기본적으로 건너뛴다.
    skin_without_face=True 면 색 기준으로 추정한 피부로 처리한다.
    (뒷모습, 아주 작은 인물처럼 사람은 있는데 얼굴이 안 잡히는 경우)

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

    if match_mode == "same_scene":
        tone_strength = 0.0
        white_balance_mode = "reference"
        balance_strength = 1.0
    elif match_mode != "look":
        raise ValueError("match_mode 는 'look' 또는 'same_scene' 이어야 합니다.")

    # 0. 레터박스 제거 (검은 띠가 '블랙'으로 통계에 섞이지 않도록)
    reference_crop = target_crop = None

    if auto_crop:
        reference_rgb, reference_crop = remove_letterbox(reference_rgb)
        target_rgb, target_crop = remove_letterbox(target_rgb)


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
            safe_limits=tone_safe_limits,
            strength=tone_strength,
            tone_follow=tone_follow,
        )

        primaries = primaries_result["primaries"]

        def primaries_stage(pixels):
            return apply_primaries(pixels, primaries)

    elif tone_method == "histogram":

        curve_strength = float(np.clip(tone_strength, 0.0, 1.0))

        def primaries_stage(pixels):
            curved = np.asarray(apply_tone_curve_to_pixels(pixels, tone_curve))
            return np.asarray(pixels) * (1.0 - curve_strength) + curved * curve_strength

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

    # 얼굴은 위치만 쓰므로 원본에서 한 번만 찾는다.
    target_faces = detect_faces(target_rgb)

    # 피부는 중성색 후보에서 빼기 위해 ①② 이후 피부 색 범위를 구해 둔다.
    primaries_skin_cluster = skin_cluster_from_pixels(
        sample_face_pixels(target_primaries_rgb, target_faces)
    )

    white_balance_result = estimate_white_balance(
        target_rgb=target_primaries_rgb,
        reference_rgb=reference_rgb,
        strength=balance_strength,
        mode=white_balance_mode,
        skin_cluster=primaries_skin_cluster,
    )

    white_balance = white_balance_result["white_balance"]

    def white_balance_stage(pixels):
        return apply_white_balance(pixels, white_balance)

    target_balanced_rgb = apply_stage_to_image(
        white_balance_stage,
        target_primaries_rgb,
    )


    # ③ 채도: 두 이미지 모두 얼굴이 있으면 피부 채도를 기준으로
    reference_skin, reference_skin_samples = measure_skin_samples(reference_rgb)

    if reference_skin["source"] != "face":
        reference_skin_samples = None

    saturation_result = estimate_saturation(
        target_rgb=target_balanced_rgb,
        reference_rgb=reference_rgb,
        target_skin_samples=sample_face_pixels(target_balanced_rgb, target_faces),
        reference_skin_samples=reference_skin_samples,
        safe_limits=tone_safe_limits,
        strength=saturation_strength,
    )

    saturation = saturation_result["saturation"]

    def saturation_stage(pixels):
        return apply_saturation(pixels, saturation)

    def tone_stage(pixels):
        return saturation_stage(white_balance_stage(primaries_stage(pixels)))

    target_tone_matched_rgb = apply_stage_to_image(
        saturation_stage,
        target_balanced_rgb,
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
    saturation_nodes = np.asarray(
        saturation_stage(lut_input), dtype=np.float32
    )
    tone_stage_nodes = np.asarray(
        tone_stage(lut_input), dtype=np.float32
    )

    # ④⑤ 피부: ①②③ 이후 이미지의 얼굴에서 피부 샘플링
    #    얼굴이 없으면 색 기반 후보만 따로 보관해 두고 기본적으로는 쓰지 않는다.
    target_skin, measured_samples = measure_skin_samples(
        target_tone_matched_rgb,
        faces=target_faces,
    )

    if target_skin["source"] == "face":
        target_skin_samples = measured_samples
        target_skin_color_samples = None
    else:
        target_skin_samples = None
        target_skin_color_samples = measured_samples

    skin_enabled, active_skin_samples = resolve_skin_processing(
        face_samples=target_skin_samples,
        color_samples=target_skin_color_samples,
        skin_without_face=skin_without_face,
    )


    target_skin["leakage"] = (
        skin_leakage(
            target_tone_matched_rgb,
            skin_cluster_from_pixels(active_skin_samples),
            target_faces if target_skin["source"] == "face" else [],
        )
        if skin_enabled else None
    )

    skin_target_hue, skin_target_used = resolve_target_hue(
        skin_target_mode,
        reference_skin,
    )

    # ⑤ 피부 밝기: Reference 얼굴 피부 밝기에 맞춘다.
    #    (실제 얼굴에서 잰 Target 피부가 있을 때만. 표준 피부 샘플의 밝기는 의미가 없다)
    reference_skin_lightness_target = (
        reference_skin.get("lightness")
        if skin_target_used == "reference" else None
    )
    skin_lightness_target = (
        reference_skin_lightness_target
        if (skin_enabled and target_skin["source"] == "face") else None
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
        skin_samples=active_skin_samples,
        skin_enabled=skin_enabled,
        saturation_nodes=saturation_nodes,
        skin_target_lightness=skin_lightness_target,
        skin_lightness_strength=skin_lightness_strength,
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
        "saturation": output_directory / f"target_to_reference_{lut_size}_3_sat.cube",
        "color": output_directory / f"target_to_reference_{lut_size}_4_color.cube",
        "skin": output_directory / f"target_to_reference_{lut_size}_5_skin.cube",
    }

    cube_titles = {
        "combined": "Target to Reference (1-5 all)",
        "primaries": f"Target to Reference (1 primaries: {tone_method})",
        "white_balance": "Target to Reference (2 white balance)",
        "saturation": "Target to Reference (3 saturation)",
        "color": "Target to Reference (4 color)",
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
        "saturation": float(saturation),
        "saturation_diagnostics": saturation_result["diagnostics"],
        "saturation_nodes": saturation_nodes,
        "saturation_strength": float(saturation_strength),
        "skin_lightness_strength": float(skin_lightness_strength),
        "reference_skin_lightness_target": reference_skin_lightness_target,
        "skin_lightness_target": skin_lightness_target,
        "target_balanced_rgb": target_balanced_rgb,
        "reference_skin": reference_skin,
        "reference_crop": reference_crop,
        "target_crop": target_crop,
        "match_mode": match_mode,
        "tone_strength": float(tone_strength),
        "tone_safe_limits": bool(tone_safe_limits),
        "target_skin": target_skin,
        "target_skin_samples": target_skin_samples,
        "target_skin_color_samples": target_skin_color_samples,
        "skin_enabled": skin_enabled,
        "skin_without_face": skin_without_face,
        "target_faces": target_faces,
        "white_balance_mode": white_balance_mode,
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
