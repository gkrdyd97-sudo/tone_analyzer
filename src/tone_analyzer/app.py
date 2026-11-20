# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

import sys
import uuid
from pathlib import Path


import cv2
import numpy as np
import streamlit as st


# 앱 폴더 경로 (src/tone_analyzer) ─ 업로드/결과 저장 위치 기준

PROJECT_ROOT = Path(
    __file__
).resolve().parent


# src 폴더 경로 (src/tone_analyzer 의 부모)
# 패키지를 설치하지 않고 pip / Streamlit Cloud 로 실행해도
# `import tone_analyzer` 가 동작하도록 sys.path 에 추가한다.

SRC_PATH = PROJECT_ROOT.parent


# src를 Python 모듈 검색 경로에 추가

if str(SRC_PATH) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_PATH),
    )


from tone_analyzer.lut import (
    apply_3d_lut,
    make_cube_text,
)

from tone_analyzer.pipeline import (
    build_final_luts,
    resolve_skin_processing,
    run_tone_matching_pipeline,
)

from tone_analyzer.primaries import (
    apply_primaries,
    format_primaries_table,
)

from tone_analyzer.white_balance import (
    format_white_balance_table,
)

from tone_analyzer.skin_tone import (
    SKIN_LINE_HUE,
    is_reliable_skin,
    skin_leakage,
    skin_mask_from_cluster,
    skin_cluster_from_pixels,
    render_vectorscope,
)


PREVIEW_MAX_WIDTH = 1600


def make_preview(rgb_image, max_width=PREVIEW_MAX_WIDTH):
    """화면 표시용으로 이미지를 줄인다. (실시간 강도 조절을 빠르게)"""

    height, width = rgb_image.shape[:2]

    if width <= max_width:
        return rgb_image

    scale = max_width / width

    return cv2.resize(
        rgb_image,
        (max_width, int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def encode_png(rgb_image):
    success, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
    )

    if not success:
        raise RuntimeError("PNG 인코딩에 실패했습니다.")

    return buffer.tobytes()


# 페이지 설정

st.set_page_config(
    page_title="Tone Analyzer",
    page_icon="🎨",
    layout="wide",
)


st.title("Tone Analyzer")

st.write(
    "Reference와 Target 이미지를 업로드하면 "
    "톤 매칭 이미지와 3D LUT를 생성합니다."
)


# 사이드바 설정 ─ 학습 설정 (변경 시 다시 실행 필요)

st.sidebar.header(
    "학습 설정"
)

st.sidebar.caption(
    "이 항목을 바꾸면 '톤 매칭 시작'을 다시 눌러야 적용됩니다."
)


match_mode = st.sidebar.radio(
    "매칭 방식",
    options=["look", "same_scene"],
    format_func=lambda value: {
        "look": "다른 장면의 룩 옮기기",
        "same_scene": "같은 장면의 컷 맞추기",
    }[value],
    help=(
        "룩 옮기기: 다른 장소·장면에 레퍼런스의 룩을 입힙니다 (카메라 앱과 같은 방식).\n"
        "컷 맞추기: 같은 장면을 반대편에서 찍은 컷(리버스 숏)처럼 이미 같은 노출·그레이딩인 두 컷을 맞춥니다. "
        "밝기 분포 차이는 화면 내용 차이이므로 ①② 를 끄고, ③ 은 Reference 중성색에 맞춥니다."
    ),
)

same_scene = match_mode == "same_scene"


auto_crop = st.sidebar.checkbox(
    "레터박스 자동 제거",
    value=True,
    help="위아래·좌우의 검은 띠를 찾아 잘라낸 뒤 분석합니다. 검은 띠가 '블랙'으로 계산되어 섀도우가 틀어지는 것을 막습니다.",
)


tone_method = st.sidebar.selectbox(
    "톤 단계 방식",
    options=["primaries", "histogram"],
    index=0,
    format_func=lambda value: {
        "primaries": "① LGG → ② Contrast/Pivot",
        "histogram": "히스토그램 매칭 커브 (이전 방식)",
    }[value],
)


tone_safe_limits = st.sidebar.checkbox(
    "①② 안전 범위",
    value=True,
    disabled=same_scene,
    help=(
        "Gamma 0.8~1.25, Gain 0.85~1.15, Lift ±0.04, Contrast 0.85~1.2 안에서만 맞춥니다. "
        "두 이미지의 '내용 차이'(검은 옷이 많은 레퍼런스 등)를 노출 차이로 착각해 계조를 크게 비트는 것을 막습니다. "
        "실제로 노출이 크게 다른 사진을 맞출 때만 끄세요."
    ),
)


tone_strength = st.sidebar.slider(
    "①② 강도",
    min_value=0.0,
    max_value=1.0,
    value=0.0 if same_scene else 1.0,
    step=0.05,
    disabled=same_scene,
    help="0 이면 밝기·대비를 건드리지 않습니다. (같은 장면 컷 맞추기에서는 0 고정)",
)


white_balance_mode = st.sidebar.radio(
    "③ 화이트밸런스 기준",
    options=["neutral", "reference"],
    index=1 if same_scene else 0,
    disabled=same_scene,
    format_func=lambda value: {
        "neutral": "중성색을 회색으로 (밸런스 보정)",
        "reference": "Reference 의 중성색에 맞춤",
    }[value],
    help=(
        "밸런스 보정: 회색이어야 할 부분을 실제 회색으로 맞춥니다. "
        "룩의 색감은 ④ 가 담당합니다 (컬러리스트 작업 순서). "
        "Reference 맞춤: Reference 의 회색에 섞인 색감까지 따라갑니다."
    ),
)


balance_strength = st.sidebar.slider(
    "③ 화이트밸런스 강도",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help=(
        "목표 쪽으로 옮기는 정도. 1 이면 중성색을 완전히 회색으로 맞춥니다. "
        "조명의 의도된 색(텅스텐의 따뜻함 등)을 조금 남기려면 0.6~0.8."
    ),
)


lut_size = st.sidebar.selectbox(
    "LUT 크기",
    options=[17, 33, 65],
    index=1,
)


max_sample_count = st.sidebar.number_input(
    "최대 학습 픽셀 수",
    min_value=1_000,
    max_value=500_000,
    value=100_000,
    step=10_000,
)


color_method = st.sidebar.selectbox(
    "색 전이 방식",
    options=["reinhard", "idt"],
    index=0,
    help=(
        "reinhard: 평균/분산 매칭 (자연스럽고 안정적)\n"
        "idt: 색 분포 전체 매칭 (강하지만 과해질 수 있음)"
    ),
)


tonal_regions = st.sidebar.checkbox(
    "톤 영역별 색 전이 (섀도우/미드/하이라이트)",
    value=True,
    disabled=(color_method != "reinhard"),
    help=(
        "섀도우, 미드톤, 하이라이트의 색을 각각 따로 맞춥니다. "
        "(reinhard 방식에서만 사용)"
    ),
)


preserve_luminance = st.sidebar.checkbox(
    "밝기는 1D 톤 커브 유지 (색만 전이)",
    value=True,
)


# 사이드바 설정 ─ 실시간 설정

st.sidebar.header(
    "실시간 조절"
)


transfer_strength = st.sidebar.slider(
    "LUT 강도",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="재학습 없이 즉시 반영됩니다.",
)


skin_protection = st.sidebar.slider(
    "④ 피부톤 보호",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.05,
    help=(
        "피부색으로 추정되는 색의 변화를 이 비율만큼 되돌립니다. "
        "재학습 없이 즉시 반영됩니다."
    ),
)


st.sidebar.subheader(
    "⑤ 스킨톤 교정"
)


skin_target_mode = st.sidebar.radio(
    "교정 목표",
    options=["reference", "line"],
    format_func=lambda value: {
        "reference": "Reference 속 피부 색 (영화의 피부 톤)",
        "line": "스킨톤 라인 (벡터스코프 I 라인)",
    }[value],
    help=(
        "Reference 피부를 믿을 수 없으면(얼굴 없음, 너무 어둡거나 무채색) 스킨톤 라인을 사용합니다."
    ),
)


skin_hue_strength = st.sidebar.slider(
    "색상 교정 강도",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="피부 색상과 목표 색상의 차이를 이 비율만큼 줄입니다. 0이면 교정하지 않습니다.",
)


skin_saturation = st.sidebar.slider(
    "피부 채도",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05,
    help="피부의 채도 배율. 1.0 = 그대로.",
)


skin_without_face = st.sidebar.checkbox(
    "얼굴이 없어도 색 기준으로 피부 처리",
    value=False,
    help=(
        "기본: 얼굴을 찾지 못하면 ④ 피부 보호와 ⑤ 교정을 건너뜁니다 "
        "(사람이 없는 장면의 나무·모래·노을을 피부로 처리하지 않도록). "
        "뒷모습이나 아주 작은 인물처럼 사람은 있는데 얼굴이 안 잡힐 때 켜세요."
    ),
)


# 이미지 업로드

reference_file = st.file_uploader(
    "Reference 이미지",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff",
    ],
)


target_file = st.file_uploader(
    "Target 이미지",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff",
    ],
)


# 업로드 이미지 미리보기

preview_col_1, preview_col_2 = st.columns(2)

if reference_file is not None:
    with preview_col_1:
        st.subheader(
            "Reference 미리보기"
        )

        st.image(
            reference_file,
            width="stretch",
        )


if target_file is not None:
    with preview_col_2:
        st.subheader(
            "Target 미리보기"
        )

        st.image(
            target_file,
            width="stretch",
        )


# 실행 버튼

run_button = st.button(
    "톤 매칭 시작",
    type="primary",
)


if run_button:

    if reference_file is None:
        st.error(
            "Reference 이미지를 업로드해주세요."
        )
        st.stop()


    if target_file is None:
        st.error(
            "Target 이미지를 업로드해주세요."
        )
        st.stop()


    # 작업별 고유 폴더 이름

    job_id = uuid.uuid4().hex


    upload_directory = (
        PROJECT_ROOT
        / "web_uploads"
        / job_id
    )


    output_directory = (
        PROJECT_ROOT
        / "web_results"
        / job_id
    )


    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    # 업로드 파일 저장

    reference_path = (
        upload_directory
        / f"reference_{reference_file.name}"
    )


    target_path = (
        upload_directory
        / f"target_{target_file.name}"
    )


    reference_path.write_bytes(
        reference_file.getvalue()
    )


    target_path.write_bytes(
        target_file.getvalue()
    )


    try:
        with st.spinner(
            "이미지 분석과 LUT 생성 중입니다..."
        ):
            pipeline_result = (
                run_tone_matching_pipeline(
                    reference_path=reference_path,
                    target_path=target_path,
                    output_directory=output_directory,
                    lut_size=lut_size,
                    max_sample_count=max_sample_count,
                    color_method=color_method,
                    transfer_strength=transfer_strength,
                    preserve_luminance=preserve_luminance,
                    tonal_regions=tonal_regions,
                    skin_protection=skin_protection,
                    skin_target_mode=skin_target_mode,
                    skin_hue_strength=skin_hue_strength,
                    skin_saturation=skin_saturation,
                    skin_without_face=skin_without_face,
                    tone_method=tone_method,
                    balance_strength=balance_strength,
                    white_balance_mode=white_balance_mode,
                    auto_crop=auto_crop,
                    tone_strength=tone_strength,
                    tone_safe_limits=tone_safe_limits,
                    match_mode=match_mode,
                )
            )

        tone_matched_rgb = pipeline_result[
            "target_tone_matched_rgb"
        ]

        preview_rgb = make_preview(
            tone_matched_rgb
        )

        # 슬라이더를 움직여도 다시 학습하지 않도록 세션에 저장
        st.session_state["result"] = {
            "pipeline": pipeline_result,
            "tone_matched_rgb": tone_matched_rgb,
            "preview_rgb": preview_rgb,
            "original_preview_rgb": make_preview(
                pipeline_result["target_rgb"]
            ),
            "primaries_preview_rgb": make_preview(
                pipeline_result["target_primaries_rgb"]
            ),
            "skin_mask_preview": skin_mask_from_cluster(
                preview_rgb,
                skin_cluster_from_pixels(
                    pipeline_result.get("target_skin_samples")
                ),
            ),
            "lut_size": lut_size,
        }

        st.session_state.pop("full_resolution", None)

        st.success(
            "톤 매칭이 완료되었습니다. "
            "사이드바의 'LUT 강도'는 바로 반영됩니다."
        )

    except Exception as error:
        st.error(
            "처리 중 오류가 발생했습니다."
        )

        st.exception(
            error
        )

        st.stop()


# ─────────────────────────────────────────
# 결과 표시 (세션에 저장된 결과 + 현재 강도)
# ─────────────────────────────────────────

session_result = st.session_state.get("result")

if session_result is not None:

    pipeline_result = session_result["pipeline"]
    result_lut_size = session_result["lut_size"]

    reference_skin = pipeline_result.get("reference_skin") or {}

    if skin_target_mode == "reference" and is_reliable_skin(reference_skin):
        skin_target_hue = reference_skin["hue"]
        skin_target_label = "Reference 피부"
    else:
        skin_target_hue = SKIN_LINE_HUE
        skin_target_label = "스킨톤 라인"
        if skin_target_mode == "reference":
            st.info(
                "Reference 의 피부를 믿을 만하게 측정하지 못해 "
                "(얼굴이 없거나 너무 어둡거나 무채색) 스킨톤 라인을 목표로 사용합니다."
            )

    skin_enabled, active_skin_samples = resolve_skin_processing(
        face_samples=pipeline_result.get("target_skin_samples"),
        color_samples=pipeline_result.get("target_skin_color_samples"),
        skin_without_face=skin_without_face,
    )

    current_luts = build_final_luts(
        lut_input=pipeline_result["lut_input"],
        tone_stage_nodes=pipeline_result["tone_stage_nodes"],
        color_lut_raw=pipeline_result["color_lut_raw"],
        lut_size=result_lut_size,
        transfer_strength=transfer_strength,
        skin_protection=skin_protection,
        primaries_nodes=pipeline_result["primaries_nodes"],
        white_balance_nodes=pipeline_result["white_balance_nodes"],
        skin_target_hue=skin_target_hue,
        skin_hue_strength=skin_hue_strength,
        skin_saturation=skin_saturation,
        skin_samples=active_skin_samples,
        skin_enabled=skin_enabled,
    )

    current_lut = current_luts["color"]

    preview_graded = apply_3d_lut(
        rgb_image=session_result["preview_rgb"],
        lut_output=current_lut,
        lut_size=result_lut_size,
    )

    preview_after = apply_3d_lut(
        rgb_image=preview_graded,
        lut_output=current_luts["skin"],
        lut_size=result_lut_size,
    )


    # 결과 지표

    st.subheader(
        "분석 결과"
    )


    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "LUT 크기",
        f"{result_lut_size}³",
    )

    metric_cols[1].metric(
        "학습 픽셀 수",
        f"{pipeline_result['sample_count']:,}",
    )

    metric_cols[2].metric(
        "LUT 근사 오차 (8bit)",
        f"{pipeline_result['mean_absolute_error_8bit']:.2f}",
        help="LUT가 색 전이 결과를 얼마나 잘 따라가는지 (낮을수록 좋음)",
    )

    def describe_crop(label, crop):
        if not crop or crop["confidence"] == "none":
            return f"{label}: 검은 띠 없음"
        sides = [
            f"{name} {crop[key]}px"
            for key, name in (("top", "위"), ("bottom", "아래"), ("left", "왼쪽"), ("right", "오른쪽"))
            if crop[key]
        ]
        aspect = f", {crop['aspect_name']}" if crop.get("aspect_match") else ""
        return f"{label}: {' · '.join(sides)} 제거 → {crop['width']}×{crop['height']}{aspect}"

    if pipeline_result.get("reference_crop") or pipeline_result.get("target_crop"):
        st.caption(
            "레터박스 — "
            + describe_crop("Reference", pipeline_result.get("reference_crop"))
            + " / "
            + describe_crop("Target", pipeline_result.get("target_crop"))
        )
        for label, crop in (
            ("Reference", pipeline_result.get("reference_crop")),
            ("Target", pipeline_result.get("target_crop")),
        ):
            for message in (crop or {}).get("warnings", []):
                st.warning(f"{label} 레터박스: {message}")

    if pipeline_result.get("match_mode") == "same_scene":
        st.caption("같은 장면의 컷 맞추기: ①② 끔, ③ Reference 중성색에 맞춤")

    target_skin_info = pipeline_result.get("target_skin") or {}

    metric_cols[3].metric(
        "피부 인식",
        (
            f"얼굴 {target_skin_info.get('face_count', 0)}개"
            if target_skin_info.get("source") == "face"
            else "얼굴 없음"
        ),
    )


    # 결과 이미지

    st.subheader(
        f"결과 이미지 (LUT 강도 {transfer_strength:.2f} / 피부 보호 {skin_protection:.2f} "
        f"/ 스킨 교정 {skin_hue_strength:.2f})"
    )


    tone_label = (
        "①② LGG + Contrast"
        if pipeline_result["tone_method"] == "primaries"
        else "①② 히스토그램 커브"
    )

    stage_images = [
        ("원본 Target", session_result["original_preview_rgb"]),
        (tone_label, session_result["primaries_preview_rgb"]),
        ("③ + 화이트밸런스", session_result["preview_rgb"]),
        ("④ + 컬러 그레이딩", preview_graded),
        ("⑤ + 스킨톤 교정 (최종)", preview_after),
    ]

    for row_start in (0, 3):
        image_cols = st.columns(3)

        for column, (label, image) in zip(
            image_cols,
            stage_images[row_start:row_start + 3],
        ):
            with column:
                st.write(label)
                st.image(image, width="stretch")


    # ⑤ 스킨톤 확인

    st.subheader(
        "⑤ 스킨톤"
    )

    target_skin = pipeline_result.get("target_skin") or {}

    skin_before = {"hue": current_luts.get("skin_hue_before")}
    skin_after = {"hue": current_luts.get("skin_hue_after")}

    using_face = target_skin.get("source") == "face"

    if using_face:
        st.caption(
            f"얼굴 {target_skin.get('face_count', 0)}개에서 피부를 샘플링해 "
            "④ 피부 보호와 ⑤ 교정 범위를 정했습니다."
        )
    elif skin_enabled:
        st.warning(
            "얼굴을 찾지 못해 색만으로 피부를 추정했습니다 "
            "('얼굴이 없어도 색 기준으로 피부 처리' 켜짐). "
            "피부와 비슷한 색의 배경도 피부로 처리될 수 있습니다."
        )
    else:
        st.info(
            "얼굴을 찾지 못해 ④ 피부 보호와 ⑤ 스킨톤 교정을 건너뛰었습니다. "
            "사람이 있는데 얼굴이 안 잡히는 장면(뒷모습, 작은 인물)이면 "
            "사이드바의 '얼굴이 없어도 색 기준으로 피부 처리'를 켜세요."
        )

    if skin_enabled:
        leakage = (
            target_skin.get("leakage")
            if using_face
            else skin_leakage(
                session_result["preview_rgb"],
                skin_cluster_from_pixels(active_skin_samples),
                [],
            )
        )
    else:
        leakage = None

    if leakage is not None and leakage > 0.05:
        st.warning(
            f"배경에 피부와 같은 색이 많습니다 (화면의 약 {leakage * 100:.0f}%). "
            "LUT 는 위치를 구분하지 못하므로 피부 보호·교정이 그 배경에도 적용됩니다. "
            "이런 장면은 ⑤ 강도를 낮추거나, DaVinci 에서 윈도우와 함께 퀄리파이어로 작업하세요."
        )

    def format_hue(value):
        return "-" if value is None else f"{value:.1f}°"

    skin_cols = st.columns(4)
    skin_cols[0].metric(
        f"목표 ({skin_target_label})",
        format_hue(skin_target_hue),
    )
    skin_cols[1].metric(
        "피부 색상 (교정 전)",
        format_hue(skin_before["hue"]),
    )
    skin_cols[2].metric(
        "피부 색상 (교정 후)",
        format_hue(skin_after["hue"]),
        delta=(
            None
            if skin_before["hue"] is None or skin_after["hue"] is None
            else f"{skin_after['hue'] - skin_before['hue']:+.1f}°"
        ),
        delta_color="off",
    )
    skin_cols[3].metric(
        "Reference 피부 색상",
        (
            format_hue(reference_skin.get("hue"))
            if is_reliable_skin(reference_skin)
            else "측정 불가"
        ),
    )

    st.caption(
        "Lab 색상각 기준 (0° = 마젠타/분홍 쪽, 90° = 노랑 쪽). "
        f"스킨톤 라인은 약 {SKIN_LINE_HUE:.0f}° 입니다. "
        "피부 색상은 얼굴 샘플 기준입니다."
    )

    scope_cols = st.columns(2)

    with scope_cols[0]:
        st.write("벡터스코프: ④ 후 (교정 전)")
        st.image(
            render_vectorscope(preview_graded, target_hue=skin_target_hue),
            width="stretch",
        )

    with scope_cols[1]:
        st.write("벡터스코프: ⑤ 후 (교정 후)")
        st.image(
            render_vectorscope(preview_after, target_hue=skin_target_hue),
            width="stretch",
        )

    st.caption(
        "노란 선: 스킨톤 라인 / 주황 점선: 현재 교정 목표 (스킨톤 라인과 다를 때)"
    )


    with st.expander("피부 인식 확인 (얼굴 위치 / 피부 마스크)"):
        mask_cols = st.columns(2)

        with mask_cols[0]:
            st.write("인식한 얼굴")
            boxed = session_result["preview_rgb"].copy()
            scale = boxed.shape[1] / pipeline_result["target_rgb"].shape[1]

            for x, y, w, h, _ in pipeline_result.get("target_faces") or []:
                cv2.rectangle(
                    boxed,
                    (int(x * scale), int(y * scale)),
                    (int((x + w) * scale), int((y + h) * scale)),
                    (0, 255, 0),
                    max(2, boxed.shape[1] // 400),
                )

            st.image(boxed, width="stretch")

        with mask_cols[1]:
            st.write("피부 마스크 (④ 보호 / ⑤ 교정 범위)")

            if not skin_enabled:
                mask = np.zeros(session_result["preview_rgb"].shape[:2], dtype=np.uint8)
            elif using_face:
                mask = session_result["skin_mask_preview"]
            else:
                mask = skin_mask_from_cluster(
                    session_result["preview_rgb"],
                    skin_cluster_from_pixels(active_skin_samples),
                )

            st.image(mask, width="stretch")

        st.caption(
            "흰색일수록 피부로 처리됩니다. 얼굴 밖에 흰 영역이 많으면 "
            "그 부분도 피부와 같은 색이라 함께 보정됩니다."
        )


    # 단계별 추정값

    st.subheader(
        "단계별 추정값"
    )

    st.caption(
        "표준 LGG 공식 기준 근사값입니다. DaVinci 휠에는 시작점으로 넣고 "
        "눈으로 미세 조정하세요. Gamma 는 지수 값입니다 "
        "(1.0 = 변화 없음, 1보다 작으면 중간톤이 어두워짐)."
    )

    def format_table(table):
        return {
            key: [
                f"{value:.3f}" if isinstance(value, float) else value
                for value in values
            ]
            for key, values in table.items()
        }

    value_cols = st.columns(2)

    if pipeline_result.get("primaries"):

        primaries = pipeline_result["primaries"]
        diagnostics = pipeline_result["primaries_diagnostics"]

        with value_cols[0]:
            st.write("①② Lift / Gamma / Gain → Contrast / Pivot")
            st.dataframe(
                format_table(format_primaries_table(primaries)),
                hide_index=True,
                width="stretch",
            )

            if diagnostics.get("limited") and pipeline_result.get("tone_strength", 1.0) > 0:
                raw = diagnostics.get("raw_primaries") or {}
                st.warning(
                    "①② 안전 범위로 제한한 값: "
                    + ", ".join(
                        f"{name} ({raw.get(name, float('nan')):.2f} → 제한)"
                        for name in diagnostics["limited"]
                    )
                    + ". 두 이미지의 밝기 분포 차이가 커서, 화면 내용 차이(검은 옷·그림자 비율 등)를 "
                    "노출 차이로 잘못 읽었을 가능성이 높습니다. 같은 장면의 컷이면 '같은 장면의 컷 맞추기'를, "
                    "실제로 노출이 크게 다르면 '①② 안전 범위'를 끄세요."
                )
            elif diagnostics.get("parameters_at_bounds") and not diagnostics.get("safe_limits"):
                st.warning(
                    "허용 범위 끝에 닿은 값: "
                    + ", ".join(diagnostics["parameters_at_bounds"])
                    + ". 두 이미지의 노출 차이가 매우 커서 값이 극단적입니다. "
                    "숫자는 참고만 하고 LUT 결과를 기준으로 판단하세요."
                )

    with value_cols[1]:
        st.write("③ 화이트밸런스")
        st.dataframe(
            format_table(
                format_white_balance_table(pipeline_result["white_balance"])
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Temp / Tint 는 이 앱의 단위입니다 (DaVinci 의 K 값과 다름). "
            "DaVinci 에는 RGB Gain 비율로 옮기는 것이 정확합니다."
        )

    if pipeline_result.get("primaries"):

        curve_input = np.linspace(0.0, 1.0, 64)
        curve_output = apply_primaries(
            np.repeat(curve_input[:, None], 3, axis=1),
            primaries,
        )[:, 1]

        chart_cols = st.columns(2)

        with chart_cols[0]:
            st.write("①② 전달 곡선 (입력 → 출력)")
            st.line_chart(
                {
                    "적용 후": curve_output,
                    "변화 없음": curve_input,
                },
                color=["#3e63dd", "#9b9b9b"],
            )

        with chart_cols[1]:
            st.write("루마 분위수 비교")
            st.line_chart(
                {
                    "Reference": diagnostics["reference_luma_quantiles"],
                    "Target (전)": diagnostics["target_luma_quantiles_before"],
                    "Target (후)": diagnostics["target_luma_quantiles_after"],
                },
            )
            st.caption(
                "가로축: 1, 5, 10, 25, 50, 75, 90, 95, 99 퍼센타일"
            )


    # 다운로드 파일

    st.subheader(
        "파일 다운로드"
    )

    st.caption(
        "원본 소스에는 '전체' LUT 하나만 걸면 앱 결과와 같습니다. "
        "DaVinci 에서 노드를 나눌 때는 노드 1 → 2 → 3 → 4 순서로 거세요 "
        "(차례로 건 결과도 앱 결과와 같습니다)."
    )


    download_cols = st.columns(3)

    current_skin_key = (
        skin_target_mode,
        skin_hue_strength,
        skin_saturation,
    )

    strength_tag = (
        f"s{int(round(transfer_strength * 100)):03d}"
        f"_skin{int(round(skin_protection * 100)):03d}"
        f"_fix{int(round(skin_hue_strength * 100)):03d}"
    )

    cube_cols = st.columns(5)

    for column, (key, label, suffix) in zip(
        cube_cols,
        [
            ("combined", "LUT: 전체 ①~⑤ (원본 소스용)", ""),
            ("primaries", "노드 1: ①② LGG + Contrast", "_1_primaries"),
            ("white_balance", "노드 2: ③ 화이트밸런스", "_2_wb"),
            ("color", "노드 3: ④ 컬러 그레이딩", "_3_color"),
            ("skin", "노드 4: ⑤ 스킨톤 교정", "_4_skin"),
        ],
    ):
        with column:
            st.download_button(
                label=label,
                data=make_cube_text(
                    lut_output=current_luts[key],
                    lut_size=result_lut_size,
                    title=f"Tone Analyzer {key} ({strength_tag})",
                ),
                file_name=(
                    f"target_to_reference_{result_lut_size}"
                    f"{suffix}_{strength_tag}.cube"
                ),
                mime="text/plain",
                key=f"download_cube_{key}",
            )

    with download_cols[0]:
        st.caption(f"현재 설정: 강도 {transfer_strength:.2f} / 피부 보호 {skin_protection:.2f}")

    with download_cols[1]:
        full_resolution = st.session_state.get("full_resolution")

        if (
            full_resolution is not None
            and full_resolution["strength"] == transfer_strength
            and full_resolution["skin"] == skin_protection
            and full_resolution.get("skin_correction") == current_skin_key
        ):
            st.download_button(
                label="보정 이미지 다운로드 (원본 해상도)",
                data=full_resolution["png"],
                file_name="tone_matched_result.png",
                mime="image/png",
            )

        elif st.button("원본 해상도 이미지 만들기"):
            with st.spinner("원본 해상도로 LUT를 적용하는 중..."):
                full_rgb = apply_3d_lut(
                    rgb_image=apply_3d_lut(
                        rgb_image=session_result["tone_matched_rgb"],
                        lut_output=current_lut,
                        lut_size=result_lut_size,
                    ),
                    lut_output=current_luts["skin"],
                    lut_size=result_lut_size,
                )

                st.session_state["full_resolution"] = {
                    "strength": transfer_strength,
                    "skin": skin_protection,
                    "skin_correction": current_skin_key,
                    "png": encode_png(full_rgb),
                }

            st.rerun()

    with download_cols[2]:
        analysis_json_path = Path(
            pipeline_result["analysis_json_path"]
        )

        st.download_button(
            label="분석 JSON 다운로드",
            data=analysis_json_path.read_bytes(),
            file_name="analysis_summary.json",
            mime="application/json",
        )


    # 대비 분석

    st.subheader(
        "대비 분석"
    )


    reference_contrast = (
        pipeline_result[
            "reference_contrast"
        ]
    )


    target_contrast = (
        pipeline_result[
            "target_contrast"
        ]
    )


    contrast_table = {
        "항목": [
            "RMS",
            "Percentile",
            "Shadow 평균",
            "Highlight 평균",
            "Shadow-Highlight 간격",
        ],
        "Reference": [
            reference_contrast["rms"],
            reference_contrast["percentile"],
            reference_contrast["shadow_mean"],
            reference_contrast["highlight_mean"],
            reference_contrast["gap"],
        ],
        "Target": [
            target_contrast["rms"],
            target_contrast["percentile"],
            target_contrast["shadow_mean"],
            target_contrast["highlight_mean"],
            target_contrast["gap"],
        ],
    }


    st.table(
        contrast_table
    )
