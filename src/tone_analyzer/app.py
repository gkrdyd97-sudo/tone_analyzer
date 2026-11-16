import sys
import uuid
from pathlib import Path


import cv2
import numpy as np
import streamlit as st


# 프로젝트 루트 경로

PROJECT_ROOT = Path(
    __file__
).resolve().parent


# src 폴더 경로

SRC_PATH = PROJECT_ROOT / "src"


# src를 Python 모듈 검색 경로에 추가

if str(SRC_PATH) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_PATH),
    )


from tone_analyzer.color_matching import (
    compute_skin_mask,
)

from tone_analyzer.lut import (
    apply_3d_lut,
    make_cube_text,
)

from tone_analyzer.pipeline import (
    build_final_luts,
    run_tone_matching_pipeline,
)

from tone_analyzer.primaries import (
    apply_primaries,
    format_primaries_table,
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


tone_method = st.sidebar.selectbox(
    "톤 단계 방식",
    options=["primaries", "histogram"],
    index=0,
    format_func=lambda value: {
        "primaries": "프라이머리 (Lift/Gamma/Gain + Contrast/Pivot)",
        "histogram": "히스토그램 매칭 커브 (이전 방식)",
    }[value],
)


balance_strength = st.sidebar.slider(
    "프라이머리 밸런스 강도",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    disabled=(tone_method != "primaries"),
    help=(
        "중성색(회색 계열)의 색 균형을 Reference 쪽으로 옮기는 정도. "
        "0이면 밝기와 대비만 맞춥니다."
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
    "피부톤 보호",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.05,
    help=(
        "피부색으로 추정되는 색의 변화를 이 비율만큼 되돌립니다. "
        "재학습 없이 즉시 반영됩니다."
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
                    tone_method=tone_method,
                    balance_strength=balance_strength,
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
            "skin_mask_preview": compute_skin_mask(preview_rgb),
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

    current_luts = build_final_luts(
        lut_input=pipeline_result["lut_input"],
        tone_stage_nodes=pipeline_result["tone_stage_nodes"],
        color_lut_raw=pipeline_result["color_lut_raw"],
        lut_size=result_lut_size,
        transfer_strength=transfer_strength,
        skin_protection=skin_protection,
    )

    current_lut = current_luts["color"]

    preview_after = apply_3d_lut(
        rgb_image=session_result["preview_rgb"],
        lut_output=current_lut,
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

    metric_cols[3].metric(
        "피부 추정 비율",
        f"{pipeline_result['skin_ratio'] * 100:.1f}%",
    )


    # 결과 이미지

    st.subheader(
        f"결과 이미지 (LUT 강도 {transfer_strength:.2f} / 피부 보호 {skin_protection:.2f})"
    )


    image_col_0, image_col_1, image_col_2 = (
        st.columns(3)
    )


    tone_label = (
        "② 프라이머리 결과"
        if pipeline_result["tone_method"] == "primaries"
        else "② 히스토그램 커브 결과"
    )


    with image_col_0:
        st.write(
            "① 원본 Target"
        )

        st.image(
            session_result["original_preview_rgb"],
            width="stretch",
        )


    with image_col_1:
        st.write(
            tone_label
        )

        st.image(
            session_result["preview_rgb"],
            width="stretch",
        )


    with image_col_2:
        st.write(
            "③ + 색 단계 (최종)"
        )

        st.image(
            preview_after,
            width="stretch",
        )


    with st.expander("피부 보호 마스크 보기"):
        st.caption(
            "흰색일수록 피부로 판단되어 색 변화가 줄어드는 영역입니다."
        )

        st.image(
            session_result["skin_mask_preview"],
            width="stretch",
        )


    # 프라이머리 값

    if pipeline_result.get("primaries"):

        primaries = pipeline_result["primaries"]
        diagnostics = pipeline_result["primaries_diagnostics"]

        st.subheader(
            "프라이머리 추정값"
        )

        st.caption(
            "적용 순서: Contrast/Pivot → Lift → Gain → Gamma. "
            "표준 LGG 공식 기준 근사값이므로 DaVinci 휠에는 "
            "시작점으로 넣고 눈으로 미세 조정하세요. "
            "Gamma 는 지수 값입니다 (1.0 = 변화 없음, 1보다 작으면 중간톤이 어두워짐)."
        )

        primary_cols = st.columns([1, 3])

        with primary_cols[0]:
            st.metric("Contrast", f"{primaries['contrast']:.3f}")
            st.metric("Pivot", f"{primaries['pivot']:.3f}")

        with primary_cols[1]:
            table = format_primaries_table(primaries)
            st.dataframe(
                {
                    key: (
                        [f"{value:.3f}" for value in values]
                        if key != "항목" else values
                    )
                    for key, values in table.items()
                },
                hide_index=True,
                width="stretch",
            )

        curve_input = np.linspace(0.0, 1.0, 64)
        curve_output = apply_primaries(
            np.repeat(curve_input[:, None], 3, axis=1),
            primaries,
        )

        chart_cols = st.columns(2)

        with chart_cols[0]:
            st.write("채널별 전달 곡선 (입력 → 출력)")
            st.line_chart(
                {
                    "R": curve_output[:, 0],
                    "G": curve_output[:, 1],
                    "B": curve_output[:, 2],
                    "변화 없음": curve_input,
                },
                color=["#e5484d", "#30a46c", "#3e63dd", "#9b9b9b"],
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
        "원본 소스에는 '톤 + 색' LUT 하나만 걸면 앱 결과와 같습니다. "
        "DaVinci 에서 노드를 나누고 싶다면 노드 1에 '톤', 노드 2에 '색' LUT를 거세요."
    )


    download_cols = st.columns(3)

    strength_tag = (
        f"s{int(round(transfer_strength * 100)):03d}"
        f"_skin{int(round(skin_protection * 100)):03d}"
    )

    cube_cols = st.columns(3)

    for column, (key, label, suffix) in zip(
        cube_cols,
        [
            ("combined", "LUT: 톤 + 색 (원본 소스용)", ""),
            ("tone", "LUT: 톤 단계만 (노드 1)", "_tone"),
            ("color", "LUT: 색 단계만 (노드 2)", "_color"),
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
                    rgb_image=session_result["tone_matched_rgb"],
                    lut_output=current_lut,
                    lut_size=result_lut_size,
                )

                st.session_state["full_resolution"] = {
                    "strength": transfer_strength,
                    "skin": skin_protection,
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
