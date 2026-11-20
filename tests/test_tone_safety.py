import cv2
import numpy as np
import pytest

from tone_analyzer.pipeline import run_tone_matching_pipeline
from tone_analyzer.primaries import SAFE_LOWER, SAFE_UPPER, blend_primaries, estimate_primaries


def _scene(height, width, seed, dark_fraction=0.0):
    """질감 있는 중간 밝기 장면. dark_fraction 만큼 오른쪽을 검은 옷처럼 덮는다."""
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 210, size=(height // 8 + 1, width // 8 + 1, 3))
    image = np.kron(base, np.ones((8, 8, 1)))[:height, :width].astype(np.uint8)
    if dark_fraction:
        image[:, int(width * (1 - dark_fraction)):] = rng.integers(0, 8, size=(height, int(round(width * dark_fraction)), 3))
    return image


@pytest.fixture(scope="module")
def content_difference_pair():
    # 같은 노출이지만 레퍼런스는 화면 40% 를 검은 재킷이 덮은 리버스 숏 같은 상황
    target = _scene(240, 400, seed=0)
    reference = _scene(240, 400, seed=1, dark_fraction=0.4)
    return target, reference


def _vector(primaries):
    return np.array([primaries["lift"], primaries["gamma"], primaries["gain"], primaries["contrast"]])


def test_wide_fit_overreacts_to_content_difference(content_difference_pair):
    target, reference = content_difference_pair
    result = estimate_primaries(target, reference, safe_limits=False)
    values = _vector(result["primaries"])
    # 내용 차이를 노출 차이로 착각해 안전 범위를 벗어나는 값이 나온다
    assert np.any((values < SAFE_LOWER - 1e-3) | (values > SAFE_UPPER + 1e-3))


def test_safe_limits_keep_values_in_range_and_report(content_difference_pair):
    target, reference = content_difference_pair
    result = estimate_primaries(target, reference, safe_limits=True)
    values = _vector(result["primaries"])

    assert np.all(values >= SAFE_LOWER - 1e-6)
    assert np.all(values <= SAFE_UPPER + 1e-6)
    assert result["diagnostics"]["limited"]                      # 제한한 항목을 알려줌
    raw = _vector(result["diagnostics"]["raw_primaries"])
    assert np.any((raw < SAFE_LOWER - 1e-3) | (raw > SAFE_UPPER + 1e-3))


def test_strength_zero_is_identity():
    primaries = {"lift": 0.03, "gamma": 0.8, "gain": 1.1, "contrast": 1.15, "pivot": 0.4}
    neutral = blend_primaries(primaries, 0.0)
    assert neutral["lift"] == 0.0
    assert neutral["gamma"] == neutral["gain"] == neutral["contrast"] == 1.0

    half = blend_primaries(primaries, 0.5)
    assert half["gamma"] == pytest.approx(0.8 ** 0.5)


def test_same_image_needs_no_change():
    image = _scene(240, 400, seed=2)
    result = estimate_primaries(image, image)
    assert result["primaries"]["gamma"] == pytest.approx(1.0, abs=0.02)
    assert result["diagnostics"]["limited"] == []


# ── 파이프라인 ─────────────────────────────

@pytest.fixture(scope="module")
def letterboxed_files(tmp_path_factory, content_difference_pair):
    folder = tmp_path_factory.mktemp("cut")
    target, reference = content_difference_pair

    def bars(image, top, bottom):
        height, width = image.shape[:2]
        full = np.zeros((height + top + bottom, width, 3), np.uint8)
        full[top:top + height] = image
        return full

    cv2.imwrite(str(folder / "reference.png"), bars(reference, 33, 21)[..., ::-1])
    cv2.imwrite(str(folder / "target.png"), bars(target, 33, 21)[..., ::-1])
    return folder


def _run(folder, **kwargs):
    return run_tone_matching_pipeline(
        reference_path=folder / "reference.png",
        target_path=folder / "target.png",
        output_directory=folder / f"out_{len(kwargs)}_{hash(tuple(sorted(kwargs.items())))}",
        lut_size=9,
        max_sample_count=3000,
        **kwargs,
    )


def test_pipeline_crops_letterbox_before_analysis(letterboxed_files):
    result = _run(letterboxed_files)
    assert result["reference_crop"]["top"] >= 33
    assert result["target_crop"]["bottom"] >= 21
    assert result["target_rgb"].shape[0] <= 240


def test_pipeline_without_crop_keeps_size(letterboxed_files):
    result = _run(letterboxed_files, auto_crop=False)
    assert result["reference_crop"] is None
    assert result["target_rgb"].shape[0] == 240 + 33 + 21


def test_same_scene_mode_turns_off_tone_and_matches_reference_neutrals(letterboxed_files):
    result = _run(letterboxed_files, match_mode="same_scene")
    assert result["primaries"]["gamma"] == pytest.approx(1.0)
    assert result["primaries"]["lift"] == pytest.approx(0.0)
    assert result["white_balance_mode"] == "reference"
    assert result["tone_strength"] == 0.0


def test_skin_target_defaults_to_reference_mode(letterboxed_files):
    result = _run(letterboxed_files)
    assert result["skin_target_mode"] == "reference"
    # 레퍼런스에 얼굴이 없으면 스킨톤 라인으로 대체
    assert result["skin_target_used"] == "line"


def test_unknown_match_mode_raises(letterboxed_files):
    with pytest.raises(ValueError):
        _run(letterboxed_files, match_mode="nope")
