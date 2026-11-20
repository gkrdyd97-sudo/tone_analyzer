import numpy as np
import pytest

from tone_analyzer.letterbox import crop_letterbox, detect_letterbox, remove_letterbox


def _content(height, width, seed=0, low=40, high=220):
    """질감이 있는 실제 장면 같은 내용."""
    rng = np.random.default_rng(seed)
    base = rng.integers(low, high, size=(height // 8 + 1, width // 8 + 1, 3))
    image = np.kron(base, np.ones((8, 8, 1)))[:height, :width]
    noise = rng.integers(-6, 7, size=image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def _letterbox(content, top, bottom, left=0, right=0, level=0, noise=0, seed=1):
    height, width = content.shape[:2]
    rng = np.random.default_rng(seed)
    full = np.full((height + top + bottom, width + left + right, 3), level, dtype=np.int16)
    if noise:
        full += rng.integers(-noise, noise + 1, size=full.shape)
    full[top:top + height, left:left + width] = content
    return np.clip(full, 0, 255).astype(np.uint8)


def _assert_crop_close(result, top, bottom, left=0, right=0, tolerance=14):
    for name, expected in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        if expected == 0:
            assert result[name] == 0, (name, result)
        else:
            # 마진만큼 더 잘리는 건 허용, 덜 잘리면 안 됨
            assert expected <= result[name] <= expected + tolerance, (name, result)


def test_real_capture_asymmetric_16_9():
    # 실제 Mac 스크린샷: 3456x2234 안에 16:9 영상, 위 178 / 아래 112
    image = _letterbox(_content(1944, 3456), top=178, bottom=112)
    result = detect_letterbox(image)

    _assert_crop_close(result, 178, 112)
    assert result["confidence"] == "high"
    assert result["aspect_name"] == "16:9"


@pytest.mark.parametrize("aspect", [2.39, 1.85])
def test_cinemascope_bars(aspect):
    width = 1920
    content_height = int(round(width / aspect))
    bar = (1080 - content_height) // 2
    image = _letterbox(_content(content_height, width), top=bar, bottom=1080 - content_height - bar)

    result = detect_letterbox(image)
    _assert_crop_close(result, bar, 1080 - content_height - bar)
    assert result["confidence"] == "high"


def test_video_black_level_16_with_noise():
    image = _letterbox(_content(800, 1920), top=140, bottom=140, level=16, noise=2)
    result = detect_letterbox(image)
    _assert_crop_close(result, 140, 140)


def test_subtitles_inside_bar_are_ignored():
    image = _letterbox(_content(800, 1920), top=140, bottom=140)
    # 아래 띠 안에 흰 자막 (가로 폭의 약 15%)
    image[1000:1030, 800:1088] = 235
    result = detect_letterbox(image)
    _assert_crop_close(result, 140, 140)


def test_pillarbox_and_windowbox():
    pillar = _letterbox(_content(1080, 1440), top=0, bottom=0, left=240, right=240)
    _assert_crop_close(detect_letterbox(pillar), 0, 0, 240, 240)

    window = _letterbox(_content(900, 1600), top=90, bottom=90, left=160, right=160)
    _assert_crop_close(detect_letterbox(window), 90, 90, 160, 160)


def test_vertical_video_with_bars():
    image = _letterbox(_content(1080, 1080), top=420, bottom=420)
    result = detect_letterbox(image)
    _assert_crop_close(result, 420, 420)


def test_no_bars():
    image = _content(1080, 1920)
    result = detect_letterbox(image)
    assert result["confidence"] == "none"
    assert result["top"] == result["bottom"] == result["left"] == result["right"] == 0


def test_dark_night_sky_is_not_cropped():
    image = _content(1080, 1920)
    # 위쪽 300줄: 아주 어둡지만 아래로 갈수록 서서히 밝아지는 하늘
    ramp = np.linspace(2, 30, 300)[:, None, None]
    rng = np.random.default_rng(3)
    sky = ramp + rng.normal(0, 1.5, size=(300, 1920, 3))
    image[:300] = np.clip(sky, 0, 255).astype(np.uint8)
    image[300:] = np.clip(image[300:].astype(int) * 0.3 + 25, 0, 255).astype(np.uint8)

    result = detect_letterbox(image)
    assert result["top"] == 0


def test_mostly_black_frame_is_not_cropped():
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[530:550, 900:1020] = 200   # 작은 빛 하나
    result = detect_letterbox(image)
    assert result["confidence"] == "none"
    assert result["top"] == 0


def test_crop_and_remove():
    image = _letterbox(_content(800, 1920), top=140, bottom=140)
    cropped, result = remove_letterbox(image)
    assert cropped.shape[0] == result["height"]
    assert cropped.shape[1] == result["width"] == 1920
    assert np.array_equal(cropped, crop_letterbox(image, result))
    # 잘린 결과에 검은 띠가 남지 않음
    assert cropped[:2].max() > 20 and cropped[-2:].max() > 20
