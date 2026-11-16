# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

import json
from pathlib import Path

import numpy as np


def convert_to_jsonable(value):
    """
    NumPy 자료형을 JSON 저장 가능한
    Python 기본 자료형으로 변환한다.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()


    if isinstance(value, np.integer):
        return int(value)


    if isinstance(value, np.floating):
        return float(value)


    if isinstance(value, np.bool_):
        return bool(value)


    if isinstance(value, Path):
        return str(value)


    if isinstance(value, dict):
        return {
            str(key): convert_to_jsonable(item)
            for key, item in value.items()
        }


    if isinstance(value, list):
        return [
            convert_to_jsonable(item)
            for item in value
        ]


    if isinstance(value, tuple):
        return [
            convert_to_jsonable(item)
            for item in value
        ]


    return value


def build_analysis_summary(
    pipeline_result,
):
    """
    전체 파이프라인 결과에서
    JSON으로 저장할 요약 정보만 추출한다.
    """

    summary = {
        "reference_path": pipeline_result[
            "reference_path"
        ],
        "target_path": pipeline_result[
            "target_path"
        ],
        "tone_matched_image_path": pipeline_result[
            "tone_matched_image_path"
        ],
        "final_image_path": pipeline_result[
            "final_image_path"
        ],
        "cube_path": pipeline_result[
            "cube_path"
        ],
        "lut_size": pipeline_result[
            "lut_size"
        ],
        "sample_count": pipeline_result[
            "sample_count"
        ],
        "mean_absolute_error": pipeline_result[
            "mean_absolute_error"
        ],
        "mean_absolute_error_8bit": pipeline_result[
            "mean_absolute_error_8bit"
        ],
        "reference_regions": pipeline_result[
            "reference_regions"
        ],
        "target_regions": pipeline_result[
            "target_regions"
        ],
        "reference_contrast": pipeline_result[
            "reference_contrast"
        ],
        "target_contrast": pipeline_result[
            "target_contrast"
        ],
        "tone_method": pipeline_result.get("tone_method"),
        "primaries": pipeline_result.get("primaries"),
        "primaries_diagnostics": pipeline_result.get(
            "primaries_diagnostics"
        ),
        "cube_paths": pipeline_result.get("cube_paths"),
        "white_balance": pipeline_result.get("white_balance"),
        "white_balance_diagnostics": pipeline_result.get(
            "white_balance_diagnostics"
        ),
        "transfer_strength": pipeline_result.get("transfer_strength"),
        "skin_protection": pipeline_result.get("skin_protection"),
    }


    return convert_to_jsonable(
        summary
    )


def save_analysis_summary(
    pipeline_result,
    output_path,
):
    """
    파이프라인 분석 결과를 JSON 파일로 저장한다.
    """

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary = build_analysis_summary(
        pipeline_result
    )


    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


    return output_path