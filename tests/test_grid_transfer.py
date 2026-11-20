"""D8: 톤 영역별 Reinhard 전이를 LUT 격자에 직접 적용하는 방식 검증."""

import numpy as np

from tone_analyzer.color_matching import (
    apply_reinhard_tonal,
    fit_reinhard_tonal,
    transfer_color_reinhard_tonal,
)


def _samples(seed, scale, offset):
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((20000, 3)) * scale + offset, 0.0, 1.0).astype(np.float32)


def test_same_distribution_is_identity():
    samples = _samples(0, 0.8, 0.1)
    model = fit_reinhard_tonal(samples, samples)
    out = apply_reinhard_tonal(samples, model)
    assert np.abs(out - samples).max() < 2e-3


def test_matches_sample_transfer():
    target = _samples(1, np.array([0.6, 0.8, 0.9]), 0.05)
    reference = _samples(2, np.array([0.9, 0.7, 0.4]), 0.05)
    model = fit_reinhard_tonal(target, reference)
    grid_way = apply_reinhard_tonal(target, model)
    sample_way = transfer_color_reinhard_tonal(target, reference, preserve_luminance=True)
    # sample_way 는 cv2 Lab(8bit 근사) 이라 아주 작은 차이만 허용
    assert np.mean(np.abs(grid_way - sample_way)) < 0.01


def test_moves_color_toward_reference():
    target = _samples(3, np.array([0.5, 0.6, 0.9]), 0.05)   # 푸른 장면
    reference = _samples(4, np.array([0.9, 0.6, 0.4]), 0.05)  # 따뜻한 장면
    model = fit_reinhard_tonal(target, reference)
    out = apply_reinhard_tonal(target, model)
    before = (target[:, 0] - target[:, 2]).mean()
    after = (out[:, 0] - out[:, 2]).mean()
    assert after > before + 0.1
