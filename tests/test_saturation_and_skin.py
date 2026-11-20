import numpy as np
import pytest

from tone_analyzer.lut import build_lut_from_model
from tone_analyzer.saturation import SAFE_RANGE, apply_saturation, estimate_saturation
from tone_analyzer.skin_tone import (
    SKIN_LINE_HUE,
    build_skin_correction_lut,
    skin_cluster_from_pixels,
    lab_to_srgb,
    srgb_to_lab,
)
from tone_analyzer.primaries import luma


def standard_skin_samples():
    """테스트용 피부 색 묶음 (Lab L 30~85, 색상 30~62°, 채도 12~32)"""
    grid_l, grid_h, grid_c = np.meshgrid(
        np.linspace(30, 85, 12), np.radians(np.linspace(30, 62, 9)), np.linspace(12, 32, 6), indexing="ij"
    )
    lab = np.stack([grid_l.ravel(), (grid_c * np.cos(grid_h)).ravel(), (grid_c * np.sin(grid_h)).ravel()], axis=1)
    rgb = lab_to_srgb(lab)
    return rgb[np.linalg.norm(srgb_to_lab(rgb) - lab, axis=1) < 1.0]


def _colorful(seed=0, size=(200, 300)):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 225, size=(*size, 3)).astype(np.uint8)


def _desaturate(image, amount):
    pixels = image.reshape(-1, 3).astype(float) / 255.0
    return (apply_saturation(pixels, amount).reshape(image.shape) * 255).round().astype(np.uint8)


# ── ③ 채도 ────────────────────────────────

def test_apply_saturation_keeps_luma():
    pixels = np.random.default_rng(0).random((500, 3)) * 0.8 + 0.1
    out = apply_saturation(pixels, 0.6)
    assert np.allclose(luma(out), luma(pixels), atol=1e-6)
    assert np.allclose(apply_saturation(pixels, 0.0), luma(pixels)[:, None].repeat(3, 1), atol=1e-6)


def test_scene_saturation_matches_within_safe_range():
    target = _colorful()
    reference = _desaturate(target, 0.88)
    result = estimate_saturation(target, reference)
    assert result["diagnostics"]["source"] == "scene"
    assert result["saturation"] == pytest.approx(0.88, abs=0.04)
    assert not result["diagnostics"]["limited"]


def test_scene_saturation_outside_range_is_distrusted():
    target = _colorful()
    reference = _desaturate(target, 0.4)
    result = estimate_saturation(target, reference)
    assert result["diagnostics"]["limited"]
    assert SAFE_RANGE[0] < result["saturation"] < 1.0


def test_skin_based_saturation_ignores_scene_content():
    # 장면은 완전히 다르지만(채도 높은 장면 vs 무채색 장면) 피부 채도는 같다
    colorful = _colorful()
    gray = np.repeat(np.random.default_rng(1).integers(40, 220, size=(200, 300, 1)), 3, axis=2).astype(np.uint8)
    skin = standard_skin_samples()

    result = estimate_saturation(colorful, gray, target_skin_samples=skin, reference_skin_samples=skin)
    assert result["diagnostics"]["source"] == "skin"
    assert result["saturation"] == pytest.approx(1.0, abs=1e-6)


# ── ⑤ 피부 밝기 ────────────────────────────

class _Identity:
    def predict(self, x):
        return x


def _skin_lut(lightness_scale, hue_strength=0.0):
    lut_input, _ = build_lut_from_model(_Identity(), 17)
    cluster = skin_cluster_from_pixels(standard_skin_samples())
    lut = build_skin_correction_lut(
        lut_input, target_hue=SKIN_LINE_HUE, hue_strength=hue_strength, saturation=1.0,
        skin_cluster=cluster, lightness_scale=lightness_scale,
    )
    return lut_input, lut


def test_skin_lightness_darkens_skin_only():
    lut_input, lut = _skin_lut(-0.1)
    lab_in = srgb_to_lab(lut_input.astype(float))
    lab_out = srgb_to_lab(lut.astype(float))

    skin_l, skin_a, skin_b = 60.0, 14.0, 18.0          # 대표 피부 색
    skin = np.argmin(np.linalg.norm(lab_in - [skin_l, skin_a, skin_b], axis=1))
    blue = np.argmin(np.linalg.norm(lab_in - [50.0, 0.0, -40.0], axis=1))

    assert lab_out[skin, 0] == pytest.approx(lab_in[skin, 0] * 0.9, abs=1.0)
    assert lab_out[blue, 0] == pytest.approx(lab_in[blue, 0], abs=0.2)


def test_skin_lut_is_identity_when_nothing_to_do():
    lut_input, lut = _skin_lut(0.0)
    assert np.allclose(lut, lut_input, atol=1e-6)
