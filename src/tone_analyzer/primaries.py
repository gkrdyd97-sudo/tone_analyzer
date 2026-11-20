# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

"""
① 프라이머리: Lift → Gain → (Lift ↔ Gain 보간) → Gamma → Contrast / Pivot

컬러리스트가 파형 모니터를 보며 작업하는 순서 그대로 '밝기 계조'만 다룬다.
(색 균형은 ② 화이트밸런스, 채도는 ③ 채도 단계에서 따로 처리)

shot matching 에서 시니어 컬러리스트는 두 이미지의 파형 '전체 모양'이 아니라
'서로 대응하는 지점'(블랙 바닥, 화이트 피크, 미드톤)을 맞춘다.
화면 내용이 다르면(검은 옷이 많은 컷 등) 파형 전체 모양은 원래 다르기 때문이다.

단계 (값 0~1, Rec.709 루마로 측정, RGB 세 채널에 같은 값 = 마스터 휠)
---------------------------------------------------------------------
1. Lift  : y = x + lift * (1 - x)         블랙 포인트(하위 0.5%)를 Reference 블랙에
2. Gain  : y = y * gain                   화이트 포인트(상위 0.5%)를 Reference 화이트에
3. 보간  : Gain 이 블랙을, Lift 가 화이트를 조금씩 움직이므로 1·2 를 번갈아 반복
4. Gamma : 블랙·화이트 고정, n = (y - black) / (white - black),  n' = n ** (1 / gamma)
           전체 루마 중앙값(미드톤)을 Reference 중앙값에.  1 = 변화 없음, < 1 이면 어두워짐
5. Contrast / Pivot : 블랙·화이트·피벗을 고정한 S 커브 (DaVinci "Use S-curve for contrast" 와 같은 방식)
           피벗 = Reference 중앙값, 25~75% 폭(계조)을 Reference 에 맞춤

블랙·화이트는 어느 장면에나 있어서 화면 내용 차이에 강하다.
미드톤(Gamma)·계조(Contrast)는 내용 차이에 흔들리므로 안전 범위를 두고,
범위를 크게 벗어나면 '노출 차이가 아니라 내용 차이'로 보고 원래 값 쪽으로 줄인다.

주의: DaVinci Resolve 내부 수식은 공개되어 있지 않으므로 값은 표준 공식 기준 근사값이다.
"""

import numpy as np

from .image_io import validate_rgb_image


# Rec.709 루마 계수 (파형 모니터와 같은 기준)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# 앱 그래프용 분위수
LUMA_QUANTILES = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)

BLACK_PERCENTILE = 0.5      # 블랙 포인트: 먼지 같은 점은 무시
WHITE_PERCENTILE = 99.5     # 화이트 포인트: 작은 반사광은 무시

NEUTRAL_PRIMARIES = {
    "lift": 0.0,
    "gain": 1.0,
    "gamma": 1.0,
    "contrast": 1.0,
    "pivot": 0.435,
    "black": 0.0,
    "white": 1.0,
    "boost": 1.0,
}

# Lift / Gain 은 블랙·화이트 포인트라 내용 차이에 강하다 → 넓은 한계만
LIFT_RANGE = (-0.1, 0.25)
GAIN_RANGE = (0.6, 1.6)

# 안전 범위 (내용 차이에 흔들리는 단계)
SAFE_RANGES = {
    "gamma": (0.8, 1.25),
    "contrast": (0.85, 1.2),
}

# Reference 미드톤(루마 중앙값)이 이보다 어두우면 '로우키' 장면으로 본다.
# 로우키 영화의 화이트 포인트가 낮은 건 화면에 밝은 게 적어서지, 하이라이트를 눌러서가 아니다.
# → 화이트를 끌어내리지 않는다 (밝은 조명·창은 살리고 미드톤만 깊게).
LOW_KEY_MID = 0.3

# 밝기 따라가기의 Gain 보강 기준: 파형 맨 윗부분 (작은 반사광 제외)
TOP_PERCENTILE = 98.0
# Gain 보강 최대 배율
MAX_BOOST = 1.5

# Contrast 안전 범위의 기준점(= 원래 미드톤 대비를 유지하는 값)이 가질 수 있는 범위
BASELINE_CONTRAST_RANGE = (0.75, 1.4)

# 넓은 탐색 범위
WIDE_RANGES = {
    "gamma": (0.3, 3.0),
    "contrast": (0.4, 2.5),
}


def _s_curve(n, contrast, pivot_n):
    """
    [0, 1] 정규화 값에 S 커브 대비. 0, 1, 피벗은 고정, 피벗에서 기울기 = contrast.
    범위 밖 값은 그대로 둔다.
    """

    q = float(np.clip(pivot_n, 0.02, 0.98))
    c = float(contrast)

    inside = (n > 0.0) & (n < 1.0)
    below = inside & (n <= q)
    above = inside & (n > q)

    out = n.copy()
    out[below] = q * (n[below] / q) ** c
    out[above] = 1.0 - (1.0 - q) * ((1.0 - n[above]) / (1.0 - q)) ** c
    return out


def apply_primaries(pixels, primaries):
    """
    (N, 3) 0~1 float RGB 에 ① 프라이머리를 적용한다. (Lift → Gain → Gamma → Contrast)
    """

    x = np.asarray(pixels, dtype=np.float64)

    lift = float(primaries.get("lift", 0.0))
    gain = float(primaries.get("gain", 1.0))
    gamma = float(primaries.get("gamma", 1.0))
    contrast = float(primaries.get("contrast", 1.0))
    black = float(primaries.get("black", 0.0))
    white = float(primaries.get("white", 1.0))
    pivot = float(primaries.get("pivot", 0.435))
    boost = float(primaries.get("boost", 1.0))

    y = (x + lift * (1.0 - x)) * gain

    span = max(white - black, 1e-3)
    n = (y - black) / span

    inside = (n > 0.0) & (n < 1.0)
    n = np.where(inside, np.clip(n, 1e-6, 1.0) ** (1.0 / gamma), n)

    # 밝기 따라가기의 Gain 보강: Gamma 로 내려온 파형 윗부분을 다시 올린다 (블랙 기준)
    if abs(boost - 1.0) > 1e-9:
        n = n * boost

    if abs(contrast - 1.0) > 1e-6:
        n = _s_curve(n, contrast, (pivot - black) / span)

    return np.clip(black + n * span, 0.0, 1.0)


def apply_primaries_to_image(rgb_image, primaries):
    """
    (H, W, 3) uint8 RGB 이미지에 프라이머리를 적용한다.
    """

    validate_rgb_image(rgb_image)

    height, width = rgb_image.shape[:2]

    pixels = rgb_image.reshape(-1, 3).astype(np.float64) / 255.0
    output = apply_primaries(pixels, primaries)

    return (
        (output * 255.0).round().astype(np.uint8).reshape(height, width, 3)
    )


def luma(pixels):
    return np.asarray(pixels, dtype=np.float64) @ LUMA_WEIGHTS


def sample_pixels(rgb_image, sample_count, random_generator):
    pixels = rgb_image.reshape(-1, 3)
    count = min(sample_count, len(pixels))
    indices = random_generator.choice(len(pixels), size=count, replace=False)
    return pixels[indices].astype(np.float64) / 255.0


def limit_with_distrust(value, low, high):
    """
    안전 범위 밖 값을 '범위 끝에 붙이지 않고' 원래 값(1) 쪽으로 줄인다. (로그 공간)

    범위를 r 배 넘어서면 범위 끝의 1/r 만큼만 적용한다.
    많이 벗어날수록 '노출 차이가 아니라 화면 내용 차이'일 가능성이 크기 때문.
    예) 안전 상한 1.25, 계산값 1.51 → r = ln1.51/ln1.25 = 1.85 → 1.25 ** (1/1.85) = 1.13
    """

    log_value = np.log(value)

    if np.log(low) <= log_value <= np.log(high):
        return float(value), False

    bound = np.log(high) if log_value > 0 else np.log(low)
    ratio = log_value / bound

    return float(np.exp(bound / ratio)), True


def blend_primaries(primaries, strength):
    """
    ① 강도. 0 = 변화 없음, 1 = 추정값 그대로.
    곱하는 값(gamma, gain, contrast)은 지수로, 더하는 값(lift)은 선형으로 섞는다.
    블랙·화이트·피벗은 섞은 값으로 다시 계산해야 하므로 estimate_primaries 안에서 처리한다.
    """

    strength = float(np.clip(strength, 0.0, 1.0))

    blended = dict(primaries)
    blended["lift"] = float(primaries["lift"]) * strength
    for name in ("gamma", "gain", "contrast", "boost"):
        blended[name] = float(primaries[name]) ** strength
    return blended


def _lift_gain_interleaved(target_black, target_white, reference_black, reference_white, iterations=12):
    """
    Lift 로 블랙, Gain 으로 화이트를 맞추고 번갈아 보간한다.
    y = (x + lift * (1 - x)) * gain
    """

    lift, gain = 0.0, 1.0
    history = []

    for _ in range(iterations):
        # Lift: 현재 Gain 에서 target 블랙 → reference 블랙
        lift = (reference_black / gain - target_black) / max(1.0 - target_black, 1e-3)
        lift = float(np.clip(lift, *LIFT_RANGE))

        # Gain: 현재 Lift 에서 target 화이트 → reference 화이트
        lifted_white = target_white + lift * (1.0 - target_white)
        gain = float(np.clip(reference_white / max(lifted_white, 1e-3), *GAIN_RANGE))

        black_now = (target_black + lift * (1.0 - target_black)) * gain
        white_now = (target_white + lift * (1.0 - target_white)) * gain
        history.append({"lift": lift, "gain": gain, "black": black_now, "white": white_now})

        if abs(black_now - reference_black) < 0.001 and abs(white_now - reference_white) < 0.001:
            break

    return lift, gain, history


def _bisect(function, low, high, goal, iterations=40):
    """function 이 증가 함수일 때 function(x) = goal 인 x (구간 밖이면 끝 값)"""

    f_low, f_high = function(low), function(high)
    if goal <= f_low:
        return low
    if goal >= f_high:
        return high

    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if function(middle) < goal:
            low = middle
        else:
            high = middle

    return 0.5 * (low + high)


def estimate_primaries(
    target_rgb,
    reference_rgb,
    sample_count=60_000,
    random_seed=42,
    safe_limits=True,
    strength=1.0,
    tone_follow=0.0,
):
    """
    Target -> Reference 의 ① 프라이머리를 컬러리스트 순서대로 하나씩 구한다.

    tone_follow : 0~1 '밝기 따라가기'. 0 = 안전 범위로 줄인 Gamma (기본),
                  1 = 안전 범위 없이 Reference 미드톤까지 (로그 공간에서 섞음)

    safe_limits : True 면 Gamma / Contrast 가 안전 범위를 벗어날 때 원래 값 쪽으로 줄인다
                  (limit_with_distrust). 줄인 항목은 diagnostics["limited"].
    strength    : ① 강도 0~1

    Returns
    -------
    dict
        primaries   : {"lift", "gain", "gamma", "contrast", "pivot", "black", "white"}
        diagnostics : 블랙·화이트·미드톤 전/후, Lift↔Gain 보간 과정, 넓은 범위 추정값, 제한 여부
    """

    validate_rgb_image(target_rgb)
    validate_rgb_image(reference_rgb)

    random_generator = np.random.default_rng(random_seed)

    target_pixels = sample_pixels(target_rgb, sample_count, random_generator)
    reference_pixels = sample_pixels(reference_rgb, sample_count, random_generator)

    target_luma = luma(target_pixels)
    reference_luma = luma(reference_pixels)

    reference_black, reference_mid, reference_white = np.percentile(
        reference_luma, [BLACK_PERCENTILE, 50.0, WHITE_PERCENTILE]
    )
    reference_spread = float(np.subtract(*np.percentile(reference_luma, [75.0, 25.0])))

    target_black, target_white = np.percentile(target_luma, [BLACK_PERCENTILE, WHITE_PERCENTILE])

    # 로우키 Reference 면 화이트를 내리지 않는다 (LOW_KEY_MID 설명 참고)
    low_key = bool(reference_mid < LOW_KEY_MID)
    white_goal = float(max(reference_white, target_white)) if low_key else float(reference_white)

    # 1~3. Lift → Gain → 보간
    lift, gain, history = _lift_gain_interleaved(
        target_black, target_white, reference_black, white_goal
    )

    def primaries_with(gamma_value, contrast_value, lift_value=lift, gain_value=gain, boost_value=1.0):
        black = (target_black + lift_value * (1.0 - target_black)) * gain_value
        white = (target_white + lift_value * (1.0 - target_white)) * gain_value
        return {
            "lift": lift_value, "gain": gain_value,
            "gamma": gamma_value, "contrast": contrast_value, "boost": boost_value,
            "pivot": float(reference_mid),
            "black": float(black), "white": float(white),
        }

    def output_luma(primaries):
        return luma(apply_primaries(target_pixels, primaries))

    # 4. Gamma: 전체 중앙값 → Reference 중앙값 (블랙·화이트 고정)
    def median_for(log_gamma):
        return float(np.median(output_luma(primaries_with(np.exp(log_gamma), 1.0))))

    raw_gamma = float(np.exp(_bisect(
        median_for, *np.log(WIDE_RANGES["gamma"]), goal=reference_mid
    )))

    gamma, gamma_limited = (
        limit_with_distrust(raw_gamma, *SAFE_RANGES["gamma"]) if safe_limits else (raw_gamma, False)
    )

    # 밝기 따라가기 (사용자가 원할 때만). 컬러리스트 방식:
    #   Gamma 를 더 눌러 미드톤을 무겁게 → 파형 맨 윗부분(상위 2%)이 내려온 만큼 Gain 으로 보강
    #   → 미드톤은 어둡고 하이라이트는 살아 있어 묵직하면서 대비가 생긴다.
    follow = float(np.clip(tone_follow, 0.0, 1.0))
    boost = 1.0
    if follow > 0.0:
        safe_gamma = gamma
        gamma = float(np.exp(np.log(gamma) + follow * (np.log(raw_gamma) - np.log(gamma))))

        def top_for(gamma_value):
            return float(np.percentile(output_luma(primaries_with(gamma_value, 1.0)), TOP_PERCENTILE))

        black_now = primaries_with(gamma, 1.0)["black"]
        top_safe, top_now = top_for(safe_gamma), top_for(gamma)
        if top_now - black_now > 1e-3:
            boost = float(np.clip((top_safe - black_now) / (top_now - black_now), 1.0, MAX_BOOST))

    # 5. Contrast / Pivot: 25~75% 폭 → Reference 폭 (S 커브, 피벗 = Reference 중앙값)
    def spread_for(log_contrast):
        values = output_luma(primaries_with(gamma, np.exp(log_contrast), boost_value=boost))
        return float(np.subtract(*np.percentile(values, [75.0, 25.0])))

    raw_contrast = float(np.exp(_bisect(
        spread_for, *np.log(WIDE_RANGES["contrast"]), goal=reference_spread
    )))

    # 믿을 수 있는 구간: 1(변화 없음) ~ baseline('원래 미드톤 대비(25~75% 폭)를 유지하는' 값).
    # Gain 이 화이트를 눌러 전체가 납작해지면 baseline 이 1 보다 커진다. 거기까지는 되살려야
    # 원본보다 밋밋해지지 않는다. 안전 범위(0.85~1.2)는 이 구간 바깥으로 더 가는 부분에만 건다.
    target_spread = float(np.subtract(*np.percentile(target_luma, [75.0, 25.0])))
    baseline_contrast = float(np.clip(
        np.exp(_bisect(spread_for, *np.log(WIDE_RANGES["contrast"]), goal=target_spread)),
        *BASELINE_CONTRAST_RANGE,
    ))

    if safe_limits:
        trusted_low = min(1.0, baseline_contrast)
        trusted_high = max(1.0, baseline_contrast)
        anchor = trusted_high if raw_contrast > trusted_high else trusted_low
        if trusted_low <= raw_contrast <= trusted_high:
            contrast, contrast_limited = raw_contrast, False
        else:
            ratio, contrast_limited = limit_with_distrust(raw_contrast / anchor, *SAFE_RANGES["contrast"])
            contrast = anchor * ratio
    else:
        contrast, contrast_limited = raw_contrast, False

    # 밝기 따라가기 중에는 대비를 낮추지 않는다. 어두운 영화의 좁은 미드톤 폭은 '밋밋하게 하라'가 아니라
    # 화면 대부분이 어둡다는 뜻이라, 따라가면 Gamma 로 만든 무게감이 다시 납작해진다.
    if follow > 0.0 and contrast < 1.0:
        contrast = float(np.exp(np.log(contrast) * (1.0 - follow)))

    limited = [name for name, flag in (("gamma", gamma_limited), ("contrast", contrast_limited)) if flag]

    # ① 강도: 섞은 값으로 블랙·화이트를 다시 계산
    strength = float(np.clip(strength, 0.0, 1.0))
    blended = blend_primaries(primaries_with(gamma, contrast, boost_value=boost), strength)
    primaries = primaries_with(
        blended["gamma"], blended["contrast"], lift_value=blended["lift"], gain_value=blended["gain"],
        boost_value=blended["boost"],
    )

    if strength < 1.0:
        # 피벗은 섞은 결과의 실제 미드톤으로 (대비가 미드톤을 움직이지 않도록)
        primaries["pivot"] = float(np.median(output_luma({**primaries, "contrast": 1.0})))

    after = output_luma(primaries)

    diagnostics = {
        "luma_quantile_levels": LUMA_QUANTILES.tolist(),
        "reference_luma_quantiles": np.percentile(reference_luma, LUMA_QUANTILES).tolist(),
        "target_luma_quantiles_before": np.percentile(target_luma, LUMA_QUANTILES).tolist(),
        "target_luma_quantiles_after": np.percentile(after, LUMA_QUANTILES).tolist(),
        "anchors": {
            "reference": {"black": float(reference_black), "mid": float(reference_mid), "white": float(reference_white), "spread": reference_spread},
            "target_before": {
                "black": float(target_black), "mid": float(np.median(target_luma)), "white": float(target_white),
                "spread": float(np.subtract(*np.percentile(target_luma, [75.0, 25.0]))),
            },
            "target_after": {
                "black": float(np.percentile(after, BLACK_PERCENTILE)), "mid": float(np.median(after)),
                "white": float(np.percentile(after, WHITE_PERCENTILE)),
                "spread": float(np.subtract(*np.percentile(after, [75.0, 25.0]))),
            },
        },
        "lift_gain_history": history,
        "raw_primaries": {"gamma": raw_gamma, "contrast": raw_contrast},
        "baseline_contrast": baseline_contrast,
        "low_key": low_key,
        "white_goal": white_goal,
        "tone_follow": follow,
        "boost": boost,
        "safe_limits": bool(safe_limits),
        "limited": limited,
        "strength": strength,
        "parameters_at_bounds": [],
    }

    return {
        "primaries": primaries,
        "diagnostics": diagnostics,
    }


def format_primaries_table(primaries):
    """
    UI / 로그 표시용 표 데이터 (컬러리스트 작업 순서).
    """

    return {
        "단계": [
            "1 Lift (블랙)",
            "2 Gain (화이트)",
            "4 Gamma (미드톤)",
            "5 Contrast",
            "5 Pivot",
        ],
        "값": [
            primaries["lift"],
            primaries["gain"],
            primaries["gamma"],
            primaries["contrast"],
            primaries["pivot"],
        ],
        "변화 없음": [0.0, 1.0, 1.0, 1.0, "-"],
    }
