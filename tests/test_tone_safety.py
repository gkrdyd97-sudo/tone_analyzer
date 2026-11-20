import cv2
import numpy as np
import pytest

from tone_analyzer.pipeline import run_tone_matching_pipeline
from tone_analyzer.primaries import SAFE_RANGES, estimate_primaries, limit_with_distrust


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


def _black_white_mid(image):
    from tone_analyzer.primaries import luma
    values = luma(image.reshape(-1, 3).astype(float) / 255.0)
    return np.percentile(values, [0.5, 50.0, 99.5])


def _graded(image, primaries):
    from tone_analyzer.primaries import apply_primaries_to_image
    return apply_primaries_to_image(image, primaries)


def test_lift_gain_match_black_and_white_points():
    # 같은 장면을 블랙을 띄우고 화이트를 낮춰 '바랜' 레퍼런스
    target = _scene(240, 400, seed=0)
    faded = np.clip(target.astype(float) * 0.8 + 20, 0, 255).astype(np.uint8)

    result = estimate_primaries(target, faded)
    out = _graded(target, result["primaries"])

    reference_black, _, reference_white = _black_white_mid(faded)
    out_black, _, out_white = _black_white_mid(out)
    assert out_black == pytest.approx(reference_black, abs=0.01)
    assert out_white == pytest.approx(reference_white, abs=0.01)
    assert result["primaries"]["lift"] > 0          # 블랙을 띄움
    assert len(result["diagnostics"]["lift_gain_history"]) >= 2   # 번갈아 보간


def test_gamma_matches_median_with_black_white_fixed():
    target = _scene(240, 400, seed=0)
    # 미드톤만 밝은 레퍼런스 (블랙·화이트는 같음)
    brighter_mids = (255 * (target / 255.0) ** 0.85).astype(np.uint8)

    result = estimate_primaries(target, brighter_mids)
    out = _graded(target, result["primaries"])

    _, reference_mid, _ = _black_white_mid(brighter_mids)
    _, out_mid, _ = _black_white_mid(out)
    assert out_mid == pytest.approx(reference_mid, abs=0.01)
    assert result["primaries"]["gamma"] > 1.0
    assert result["diagnostics"]["limited"] == []


def test_content_difference_is_distrusted_not_pinned(content_difference_pair):
    target, reference = content_difference_pair
    wide = estimate_primaries(target, reference, safe_limits=False)
    safe = estimate_primaries(target, reference, safe_limits=True)

    raw_gamma = wide["primaries"]["gamma"]
    gamma = safe["primaries"]["gamma"]

    assert not (SAFE_RANGES["gamma"][0] <= raw_gamma <= SAFE_RANGES["gamma"][1])
    assert "gamma" in safe["diagnostics"]["limited"]
    # 범위 끝에 붙이지 않고 그보다 원래 값(1) 쪽으로
    assert SAFE_RANGES["gamma"][0] < gamma < SAFE_RANGES["gamma"][1]


def test_limit_with_distrust():
    assert limit_with_distrust(1.1, 0.8, 1.25) == (pytest.approx(1.1), False)
    value, limited = limit_with_distrust(1.51, 0.8, 1.25)
    assert limited and 1.0 < value < 1.25
    # 더 많이 벗어날수록 더 적게 적용
    assert limit_with_distrust(3.0, 0.8, 1.25)[0] < value


def test_strength_zero_is_identity():
    target = _scene(240, 400, seed=0)
    reference = _scene(240, 400, seed=1, dark_fraction=0.4)
    result = estimate_primaries(target, reference, strength=0.0)
    out = _graded(target, result["primaries"])
    assert np.abs(out.astype(int) - target.astype(int)).max() <= 1


def test_same_image_needs_no_change():
    image = _scene(240, 400, seed=2)
    result = estimate_primaries(image, image)
    primaries = result["primaries"]
    assert primaries["lift"] == pytest.approx(0.0, abs=0.005)
    assert primaries["gain"] == pytest.approx(1.0, abs=0.01)
    assert primaries["gamma"] == pytest.approx(1.0, abs=0.02)
    assert primaries["contrast"] == pytest.approx(1.0, abs=0.03)


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


def test_five_nodes_chain_equals_combined(letterboxed_files):
    from tone_analyzer.lut import apply_3d_lut_to_pixels, load_cube_file

    result = _run(letterboxed_files)
    paths = result["cube_paths"]
    assert set(paths) == {"combined", "primaries", "white_balance", "saturation", "color", "skin"}
    assert paths["saturation"].endswith("_3_sat.cube") and paths["skin"].endswith("_5_skin.cube")

    pixels = np.random.default_rng(0).random((2000, 3)).astype(np.float32)
    chained = pixels
    for key in ("primaries", "white_balance", "saturation", "color", "skin"):
        lut, size = load_cube_file(paths[key])
        chained = apply_3d_lut_to_pixels(chained, lut, size)

    combined, size = load_cube_file(paths["combined"])
    direct = apply_3d_lut_to_pixels(pixels, combined, size)

    # 9³ 격자라 보간 오차는 있지만 평균 1% 안
    assert np.abs(chained - direct).mean() < 0.01


def test_gain_compression_does_not_flatten_midtones():
    """
    Reference 화이트가 낮아 Gain 이 전체를 누를 때, 안전 범위 때문에 미드톤 대비가
    원본보다 납작해지면 안 된다. (저장소 예제: 예전 0.327 → 0.249, 지금 0.312, Reference 0.331)
    """
    import cv2
    from pathlib import Path
    from tone_analyzer.image_io import load_rgb_image

    root = Path(__file__).resolve().parents[1] / "images"
    target = load_rgb_image(root / "target" / "target_1.png")
    reference = load_rgb_image(root / "reference" / "reference_1.png")
    target = cv2.resize(target, (target.shape[1] // 4, target.shape[0] // 4), interpolation=cv2.INTER_AREA)
    reference = cv2.resize(reference, (reference.shape[1] // 4, reference.shape[0] // 4), interpolation=cv2.INTER_AREA)

    anchors = estimate_primaries(target, reference)["diagnostics"]["anchors"]
    before = anchors["target_before"]["spread"]
    after = anchors["target_after"]["spread"]
    goal = anchors["reference"]["spread"]
    assert goal > before            # 이 쌍은 Reference 가 대비가 더 크다
    assert after > before * 0.9     # 원본보다 납작해지지 않음
