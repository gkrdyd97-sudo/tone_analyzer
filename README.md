# Camera Tone Matching & LUT Generator

두 카메라로 같은 장면을 촬영했을 때, 한 카메라의 계조와 톤을 다른 카메라에 맞추기 위한 학습용 이미지 분석 프로젝트입니다.

이 프로젝트는 다음 단계로 발전합니다.

```text
JPG 이미지 분석
→ Shadow / Midtone / Highlight 분석
→ Reference와 Target 차이 계산
→ 1D 톤 커브 생성
→ 색상 차이 분석
→ 3D LUT 생성
→ .cube 파일 저장
```

## 프로젝트 목적

이 프로젝트에서 사용하는 용어는 다음과 같습니다.

- **Reference**: 기준으로 삼을 카메라의 이미지
- **Target**: Reference의 계조와 톤에 맞출 카메라의 이미지
- **톤 매칭**: 밝기, 대비, Shadow, Midtone, Highlight를 기준 이미지와 비슷하게 맞추는 작업
- **3D LUT**: 입력 RGB 색상을 다른 RGB 색상으로 변환하는 표

최종적으로는 다음과 같은 작업을 목표로 합니다.

```text
카메라 A 이미지 = Reference
카메라 B 이미지 = Target

카메라 B의 밝기·색상 차이 분석
→ 카메라 B를 카메라 A에 맞추는 보정
→ 보정 규칙을 LUT로 저장
```

## RAG와의 관계

이 프로젝트의 핵심은 RAG가 아닙니다.

현재 작업은 다음에 해당합니다.

```text
OpenCV + NumPy
→ 이미지 픽셀 분석
→ 색상과 밝기 변환
→ LUT 생성
```

RAG는 문서나 분석 결과를 검색한 뒤 LLM이 답변하게 만드는 구조입니다. 나중에 다음과 같은 기능을 추가할 때 RAG를 사용할 수 있습니다.

```text
"기준 샷과 가장 비슷한 촬영본은 무엇인가?"
"이 샷의 Shadow가 기준보다 얼마나 밝은가?"
"대비가 강한 샷만 찾아줘."
```

하지만 카메라 매칭 자체에는 먼저 이미지 분석과 색상 변환이 필요합니다.

## 권장 프로젝트 구조

```text
tone_analyzer/
├── README.md
├── tone_analysis.ipynb
├── images/
│   ├── reference/
│   │   └── reference.jpg
│   └── target/
│       └── target.jpg
└── results/
```

나중에 코드가 커지면 다음처럼 확장할 수 있습니다.

```text
tone_analyzer/
├── notebooks/
│   ├── 01_load_images.ipynb
│   ├── 02_analyze_tone.ipynb
│   ├── 03_build_tone_curve.ipynb
│   └── 04_build_3d_lut.ipynb
├── src/
│   ├── image_io.py
│   ├── tone_analysis.py
│   ├── color_matching.py
│   └── lut_export.py
├── images/
├── results/
└── pyproject.toml
```

## 설치

프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
uv init
uv add opencv-python numpy pandas matplotlib scipy scikit-learn ipykernel
```

Jupyter 커널을 등록합니다.

```bash
uv run python -m ipykernel install --user \
  --name tone-analyzer \
  --display-name "Python (tone_analyzer)"
```

Notebook에서 `Python (tone_analyzer)` 커널을 선택합니다.

## 이미지 형식

학습 단계에서는 JPG나 PNG 모두 사용할 수 있습니다.

```text
.jpg
.jpeg
.png
.tif
.tiff
```

실제 카메라 매칭에서는 JPG보다 RAW 또는 Log 원본이 더 많은 정보를 보존할 수 있습니다. 다만 Log 원본은 카메라별 감마와 색역을 정확히 해석해야 하므로, 처음에는 JPG로 구조를 학습하는 것이 좋습니다.

## 전체 작업 흐름

### 1. 이미지 준비

가능하면 Reference와 Target은 다음 조건을 만족해야 합니다.

- 같은 장면
- 같은 시간
- 같은 조명
- 비슷한 구도
- 고정된 노출
- 고정된 화이트 밸런스
- 가능하면 ColorChecker 포함

서로 다른 장소나 시간에 촬영한 이미지는 색 차이가 카메라 때문인지 장면 때문인지 분리하기 어렵습니다. 아무 이미지 두 장은 코드 테스트에는 사용할 수 있지만, 실제 카메라 변환용 LUT 제작에는 적합하지 않습니다.

### 2. 이미지 읽기

```python
import cv2
from pathlib import Path

reference_path = Path("images/reference/reference.jpg")
target_path = Path("images/target/target.jpg")

reference_bgr = cv2.imread(
    str(reference_path),
    cv2.IMREAD_COLOR,
)

target_bgr = cv2.imread(
    str(target_path),
    cv2.IMREAD_COLOR,
)

if reference_bgr is None:
    raise FileNotFoundError("Reference 이미지를 읽지 못했습니다.")

if target_bgr is None:
    raise FileNotFoundError("Target 이미지를 읽지 못했습니다.")

reference_rgb = cv2.cvtColor(
    reference_bgr,
    cv2.COLOR_BGR2RGB,
)

target_rgb = cv2.cvtColor(
    target_bgr,
    cv2.COLOR_BGR2RGB,
)
```

OpenCV는 기본적으로 BGR 순서로 이미지를 읽고, Matplotlib는 RGB 순서를 사용합니다. 따라서 화면에 표시하거나 색상을 분석하기 전에 RGB로 변환합니다.

### 3. 공통 색공간으로 변환

밝기 분석에는 RGB보다 Lab 색공간이 편리합니다.

```python
reference_lab = cv2.cvtColor(
    reference_rgb,
    cv2.COLOR_RGB2LAB,
)

target_lab = cv2.cvtColor(
    target_rgb,
    cv2.COLOR_RGB2LAB,
)

reference_lightness = (
    reference_lab[:, :, 0].astype(np.float32)
    / 255
    * 100
)

target_lightness = (
    target_lab[:, :, 0].astype(np.float32)
    / 255
    * 100
)
```

Lab 채널은 다음 의미를 가집니다.

```text
L → 밝기
a → 초록색 ↔ 빨간색
b → 파란색 ↔ 노란색
```

분석에는 Lab을 사용하고, 최종 LUT는 보통 RGB 색공간에서 생성합니다.

### 4. Shadow / Midtone / Highlight 분류

이미지를 사각형으로 자르는 것이 아니라, 픽셀 밝기를 기준으로 분류합니다.

```python
import numpy as np

def make_tonal_masks(lightness):
    shadow_threshold, highlight_threshold = np.percentile(
        lightness,
        [20, 80],
    )

    shadow_mask = lightness <= shadow_threshold

    midtone_mask = (
        (lightness > shadow_threshold)
        & (lightness < highlight_threshold)
    )

    highlight_mask = lightness >= highlight_threshold

    return {
        "shadow": shadow_mask,
        "midtone": midtone_mask,
        "highlight": highlight_mask,
        "shadow_threshold": shadow_threshold,
        "highlight_threshold": highlight_threshold,
    }
```

백분위 기준은 각 이미지의 상대적인 분위기를 분석할 때 유용합니다.

```text
가장 어두운 20% → Shadow
중간 60%       → Midtone
가장 밝은 20%  → Highlight
```

여러 이미지의 절대 밝기를 비교할 때는 고정 기준을 사용할 수 있습니다.

```text
L* < 25       → Shadow
25 ≤ L* ≤ 75  → Midtone
L* > 75       → Highlight
```

### 5. 영역별 통계 계산

```python
def analyze_region(lightness, mask):
    selected = lightness[mask]

    return {
        "pixel_ratio": float(len(selected) / lightness.size),
        "mean_lightness": float(selected.mean()),
        "std_lightness": float(selected.std()),
        "min_lightness": float(selected.min()),
        "max_lightness": float(selected.max()),
    }
```

분석할 수 있는 대표적인 값은 다음과 같습니다.

```text
pixel_ratio      → 해당 영역이 전체에서 차지하는 비율
mean_lightness   → 평균 밝기
std_lightness    → 영역 내부의 밝기 변화
min/max          → 밝기 범위
```

### 6. 대비 분석

```python
def analyze_contrast(lightness, masks):
    p05, p95 = np.percentile(lightness, [5, 95])

    shadow_mean = lightness[masks["shadow"]].mean()
    highlight_mean = lightness[masks["highlight"]].mean()

    return {
        "rms_contrast": float(lightness.std()),
        "percentile_contrast": float(p95 - p05),
        "highlight_shadow_gap": float(
            highlight_mean - shadow_mean
        ),
    }
```

해석 방법은 다음과 같습니다.

```text
rms_contrast가 큼 → 밝기 변화가 큼
percentile_contrast가 큼 → 실질적인 명암 범위가 넓음
highlight_shadow_gap이 큼 → Shadow와 Highlight 차이가 큼
```

### 7. Reference와 Target 차이 계산

```python
def compare_region_stats(reference_stats, target_stats):
    comparison = {}

    for region in ["shadow", "midtone", "highlight"]:
        reference_value = reference_stats[region][
            "mean_lightness"
        ]

        target_value = target_stats[region][
            "mean_lightness"
        ]

        comparison[region] = {
            "reference": reference_value,
            "target": target_value,
            "target_minus_reference": (
                target_value - reference_value
            ),
        }

    return comparison
```

예를 들어 Target의 Shadow 차이가 `+7`이면 Target의 Shadow가 Reference보다 밝다는 뜻입니다.

```text
Target - Reference > 0 → Target이 더 밝음
Target - Reference < 0 → Target이 더 어두움
```

## 1D 톤 커브 생성

1D 톤 커브는 밝기만 변환합니다.

```text
Target 밝기 → Reference 밝기
```

백분위별 밝기를 대응시켜 커브를 만들 수 있습니다.

```python
percentiles = np.linspace(0, 100, 101)

target_points = np.percentile(
    target_lightness,
    percentiles,
)

reference_points = np.percentile(
    reference_lightness,
    percentiles,
)

target_points, unique_indices = np.unique(
    target_points,
    return_index=True,
)

reference_points = reference_points[
    unique_indices
]

mapped_lightness = np.interp(
    target_lightness,
    target_points,
    reference_points,
)
```

`np.interp()`는 지정된 기준점 사이의 값을 선형 보간합니다.

```text
Target 25 → Reference 18
Target 50 → Reference 48
Target 75 → Reference 80
```

새 밝기값을 Target의 Lab 이미지에 적용합니다.

```python
target_lab_tone_matched = target_lab.copy()

new_l_channel = np.clip(
    mapped_lightness / 100 * 255,
    0,
    255,
).astype(np.uint8)

target_lab_tone_matched[:, :, 0] = new_l_channel

target_tone_matched_rgb = cv2.cvtColor(
    target_lab_tone_matched,
    cv2.COLOR_LAB2RGB,
)
```

이 단계에서는 밝기와 대비를 먼저 맞추고, 색상 보정은 다음 단계에서 처리하는 것이 좋습니다.

## 3D LUT 생성 원리

3D LUT는 RGB 세 값을 동시에 변환합니다.

```text
Target RGB → Reference RGB
```

두 이미지가 같은 장면이고 정렬되어 있다면 같은 위치의 픽셀을 대응 데이터로 사용할 수 있습니다.

```python
target_pixels = (
    target_rgb
    .reshape(-1, 3)
    .astype(np.float32)
    / 255
)

reference_pixels = (
    reference_rgb
    .reshape(-1, 3)
    .astype(np.float32)
    / 255
)
```

각 행은 하나의 RGB 픽셀입니다.

```text
Target RGB          Reference RGB
[0.12, 0.20, 0.35]  → [0.18, 0.22, 0.31]
```

이미지 전체 픽셀을 그대로 사용하면 너무 많을 수 있으므로 샘플을 추출합니다.

```python
rng = np.random.default_rng(42)

sample_count = min(
    100000,
    len(target_pixels),
)

sample_indices = rng.choice(
    len(target_pixels),
    size=sample_count,
    replace=False,
)

target_samples = target_pixels[
    sample_indices
]

reference_samples = reference_pixels[
    sample_indices
]
```

## RGB 변환 모델 만들기

교육용 구현에서는 KNN 보간을 사용할 수 있습니다.

```python
from sklearn.neighbors import KNeighborsRegressor

color_model = KNeighborsRegressor(
    n_neighbors=32,
    weights="distance",
    n_jobs=-1,
)

color_model.fit(
    target_samples,
    reference_samples,
)
```

모델은 Target RGB가 들어오면 가장 가까운 색상 샘플을 찾아 Reference RGB를 예측합니다.

```python
test_colors = np.array([
    [0.1, 0.1, 0.1],
    [0.5, 0.5, 0.5],
    [0.8, 0.6, 0.4],
])

predicted_colors = color_model.predict(
    test_colors,
)

predicted_colors = np.clip(
    predicted_colors,
    0,
    1,
)
```

## 3D LUT 격자 만들기

처음에는 `17 × 17 × 17` LUT를 사용합니다.

```python
lut_size = 17
grid = np.linspace(0, 1, lut_size)
```

총 색상점 수는 다음과 같습니다.

```text
17 × 17 × 17 = 4,913개
```

```python
cube_points = []

for b in grid:
    for g in grid:
        for r in grid:
            cube_points.append([r, g, b])

cube_points = np.array(cube_points)
```

각 격자점의 출력값을 예측합니다.

```python
cube_outputs = color_model.predict(
    cube_points,
)

cube_outputs = np.clip(
    cube_outputs,
    0,
    1,
)
```

## `.cube` 파일 저장

```python
def save_cube_lut(
    path,
    lut_size,
    cube_outputs,
    title="Camera Match LUT",
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")

        for output_rgb in cube_outputs:
            r, g, b = output_rgb

            f.write(
                f"{r:.8f} "
                f"{g:.8f} "
                f"{b:.8f}\n"
            )
```

```python
lut_path = Path(
    "results/target_to_reference_17.cube"
)

save_cube_lut(
    lut_path,
    lut_size,
    cube_outputs,
    title="Target to Reference",
)

print("LUT 저장 완료:", lut_path)
```

생성된 `.cube` 파일은 DaVinci Resolve 등 LUT를 지원하는 프로그램에서 불러올 수 있습니다.

## Log 원본 처리

Log 원본도 사용할 수 있지만, 먼저 카메라의 감마와 색역을 확인해야 합니다.

```text
Sony      → S-Log3 / S-Gamut3.Cine
Canon     → C-Log2 / Cinema Gamut
Panasonic → V-Log / V-Gamut
ARRI      → LogC
```

두 카메라의 Log 데이터를 바로 비교하면 안 됩니다.

```text
카메라 A Log → 공통 작업 색공간
카메라 B Log → 공통 작업 색공간
                 ↓
              비교 및 LUT 생성
```

처음에는 다음 흐름을 권장합니다.

```text
Log 원본
→ 카메라 전용 Log-to-Rec.709 변환
→ PNG 또는 TIFF로 출력
→ 현재 Python 분석 프로그램에 입력
```

나중에는 ACES나 OpenColorIO를 사용해 Log-to-Log 변환과 공통 색공간 변환을 직접 처리할 수 있습니다.

## 고정 픽셀 청킹이 필요한가?

전체 색감과 계조 분석에는 고정 픽셀 크기가 필수는 아닙니다.

```text
원본 해상도 유지
→ 모든 픽셀의 밝기와 색상 분석
→ 비율과 평균값으로 비교
```

다만 다음 경우에는 이미지 정렬과 고정된 공간 기준이 필요합니다.

- 같은 위치의 픽셀을 직접 비교할 때
- ColorChecker 영역을 추출할 때
- 3D LUT 대응 데이터를 만들 때
- 이미지의 왼쪽, 중앙, 오른쪽 영역을 비교할 때

이미지 크기보다 중요한 것은 같은 위치의 픽셀이 같은 장면을 가리키는지 여부입니다.

## 정확한 LUT를 위한 주의사항

### 같은 장면과 정렬

서로 다른 장소의 이미지는 카메라 차이와 장면 차이를 분리하기 어렵습니다.

### ColorChecker 사용

카메라 매칭에는 ColorChecker의 각 색상 패치를 대응시키는 방법이 가장 안정적입니다.

### 클리핑 제거

완전히 검거나 흰 픽셀은 정보가 사라진 상태일 수 있습니다.

```text
완전한 Shadow 클리핑
완전한 Highlight 클리핑
```

이런 픽셀은 LUT 학습에서 제외하는 것이 좋습니다.

### 노출과 화이트 밸런스

노출과 화이트 밸런스를 먼저 맞추지 않으면 LUT가 카메라 색상 차이뿐 아니라 촬영 실수까지 학습할 수 있습니다.

### LUT는 사라진 정보를 복원하지 못함

Target의 Highlight가 완전히 날아갔다면 LUT가 원래 색을 복구할 수 없습니다.

## 권장 개발 순서

```text
1. JPG 두 장 읽기
2. 이미지 크기와 색상 확인
3. Lab 밝기 분석
4. Shadow/Midtone/Highlight 분류
5. Reference와 Target의 통계 비교
6. 1D 톤 커브 생성
7. 보정 전후 이미지 비교
8. RGB 대응 데이터 생성
9. 17³ 3D LUT 생성
10. .cube 파일 저장
11. 다른 이미지로 검증
12. ColorChecker와 Log 원본으로 개선
```

## 최종 목표

최종적으로 다음과 같은 프로그램을 만드는 것이 목표입니다.

```text
Reference 이미지 또는 프레임
Target 이미지 또는 프레임
        ↓
공통 색공간 변환
        ↓
노출·밝기·색상 분석
        ↓
차이 계산
        ↓
1D 톤 커브와 색상 변환 계산
        ↓
3D LUT 생성
        ↓
.cube 파일 출력
```

이 프로젝트는 먼저 분석 가능한 코드를 만드는 데서 시작하고, 이후 보정과 LUT 생성으로 단계적으로 확장합니다.
