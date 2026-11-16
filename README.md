# Tone Analyzer

Reference 이미지의 톤과 색감을 Target 이미지에 옮기고, 그 변환을 **DaVinci Resolve 등에서 바로 쓸 수 있는 3D LUT(`.cube`)** 로 내보내는 Streamlit 웹 앱입니다.

컬러리스트의 작업 순서를 그대로 따릅니다.

```text
① Lift / Gamma / Gain
        ↓
② Contrast / Pivot                    ─▶ 노드 1 LUT
        ↓
③ 화이트밸런스 (Temp / Tint)           ─▶ 노드 2 LUT
        ↓
④ 컬러 그레이딩 (톤 영역별 색 전이)
   + 피부톤 보호 · 강도 조절            ─▶ 노드 3 LUT
        ↓
.cube 내보내기 (전체 하나 / 노드별 3개)
```

---

## 빠른 시작

```bash
# 의존성 설치 (Python 3.12, uv 사용)
uv sync

# 웹 앱 실행
uv run streamlit run src/tone_analyzer/app.py
```

1. **Reference**(따라 하고 싶은 룩)와 **Target**(보정할 이미지)을 업로드합니다.
2. 사이드바의 **학습 설정**을 확인하고 `톤 매칭 시작`을 누릅니다.
3. **실시간 조절**(LUT 강도, 피부톤 보호)을 움직이며 결과를 확인합니다. 재학습 없이 바로 반영됩니다.
4. 원하는 설정에서 LUT를 다운로드합니다.

---

## 어떤 LUT를 받아야 하나요?

| DaVinci 에서의 사용 방식 | 받을 파일 |
|---|---|
| LUT 하나로 끝내기 | **전체 ①~④** (`..._{N}_sXXX_skinYYY.cube`) |
| 노드를 나눠 단계별로 조절 | 노드 1 `_1_primaries` → 노드 2 `_2_wb` → 노드 3 `_3_color` (차례로 건 결과 = 앱 결과) |
| ①~③은 휠로 직접 맞추기 | **노드 3** `_3_color` 만 — 앱의 추정값에 가깝게 맞춘 뒤 사용 |

> ⚠️ **색공간 주의**: LUT는 Rec.709 디스플레이 이미지(JPG/PNG/스크린샷) 기준으로 만들어집니다.
> S-Log, LogC 등 로그 소스에는 먼저 CST/변환 LUT로 Rec.709 로 바꾼 **다음 노드**에 거세요.

> 강도와 피부 보호 값은 **다운로드 시점의 값으로 LUT에 고정**됩니다. 파일 이름의 `s080_skin085` 는 강도 0.80, 피부 보호 0.85 를 뜻합니다.

---

## 처리 과정

### ①② Lift / Gamma / Gain → Contrast / Pivot (`primaries.py`)

**밝기 계조만** 다룹니다. 루마 분위수(1, 5, 10, 25, 50, 75, 90, 95, 99%)를 Reference 에 맞추는 값을 최적화(`scipy.optimize.least_squares`)로 찾습니다. RGB 세 채널에 같은 값을 쓰는 **마스터 휠**에 해당합니다.

```text
① Lift   : y = x + lift * (1 - x)          흰색 고정, 검정 이동
   Gain  : y = y * gain                    검정 고정, 흰색 이동
   Gamma : y = y ** (1 / gamma)            지수. 1 = 변화 없음, < 1 이면 중간톤이 어두워짐
② Contrast / Pivot : y = (y - pivot) * contrast + pivot     (pivot = Reference 루마 중앙값)
```

- **계조 간격 유지 벌점**: 이웃한 분위수 간격이 원래 간격의 40% 보다 좁아지지 않게 해서 블랙 크러시 / 화이트 클리핑을 막습니다.
- 값이 허용 범위 끝에 닿으면 앱에 경고가 표시됩니다. (두 이미지의 노출 차이가 매우 클 때)
- 앱에서 추정값 표, 전달 곡선, 전/후 루마 분위수 그래프를 볼 수 있습니다.

이전 방식인 **히스토그램 매칭 커브**(Lab L 채널, 101포인트)도 `톤 단계 방식`에서 선택할 수 있습니다. 분포를 그대로 복사하므로 더 강하게 맞지만, 장면 내용 차이까지 커브에 반영되어 부자연스러울 수 있습니다.

### ③ 화이트밸런스 (`white_balance.py`)

①② 를 적용한 이미지에서 **원래 회색이어야 할 픽셀**(루마 20~95%, 상대 채도 하위 20%)의 Lab a·b 중앙값을 Reference 쪽으로 옮기는 Temp / Tint 를 찾습니다.

```text
gain_R = exp(+temp),  gain_G = exp(-tint),  gain_B = exp(-temp)
gains  = gains / luma(gains)        # 밝기 유지
temp > 0 : 따뜻하게,  tint > 0 : 마젠타 쪽
```

- 회색 계열 픽셀만 보므로 Reference 의 빨간 옷 같은 피사체 색에 끌려가지 않습니다.
- `③ 화이트밸런스 강도`(기본 0.5): Reference 의 중성색에는 룩의 색감(예: 틸 그레이딩)이 섞여 있을 수 있어 절반만 옮기고, 나머지 룩은 ④ 가 담당합니다.
- Temp / Tint 는 이 앱의 단위입니다(DaVinci 의 K 값과 다름). DaVinci 에는 함께 표시되는 **RGB Gain 비율**로 옮기는 것이 정확합니다.

> **DaVinci 수식과의 차이**: Resolve 내부 수식은 공개되어 있지 않아 ①~③ 값은 표준 공식 기준의 **근사값**입니다. 휠에 옮겨 넣을 때는 시작점으로 쓰고 눈으로 미세 조정하세요. LUT 는 근사가 아니라 앱 결과 그대로입니다.

### ④ 컬러 그레이딩 — 분포 기반 색 전이 (`color_matching.py`)

Reference 와 Target 은 **서로 다른 사진**이므로 같은 좌표의 픽셀끼리는 아무 관계가 없습니다. 그래서 다음과 같이 학습 데이터를 만듭니다.

```text
Target 픽셀 샘플링 ─┐
                    ├─▶ Target 색 분포를 Reference 분포로 전이
Reference 픽셀 샘플링 ┘
        ↓
(Target 원본 픽셀, 같은 픽셀의 전이 결과) = 학습 쌍
        ↓
3차 다항식 + Ridge 회귀로 부드러운 RGB→RGB 매핑 학습
        ↓
3D LUT 격자점(17³ / 33³ / 65³) 계산
```

색 전이 방식:

| 방식 | 설명 |
|---|---|
| **reinhard** (기본) | Lab 평균/표준편차 매칭. `톤 영역별 색 전이`를 켜면 섀도우/미드/하이라이트 각각 따로 맞춘 뒤 부드럽게 섞음 (스플릿 토닝 재현) |
| **idt** | Iterative Distribution Transfer (Pitié 2005). 분포 전체를 맞추므로 강하지만, Reference 의 특정 색이 엉뚱한 곳으로 번질 수 있음 |

`밝기는 1D 톤 커브 유지`(기본 켜짐)를 켜면 밝기는 톤 단계 결과를 유지하고 색(Lab a, b)만 옮깁니다.

### 피부톤 보호 · 강도 (`lut.py`)

두 기능 모두 **학습이 아니라 LUT 격자점에 직접** 적용되므로 `.cube` 에 그대로 반영되고, 앱에서 실시간으로 조절됩니다.

- **피부톤 보호**: 각 격자점의 입력 색이 피부색일 가능성(Lab 색상각 약 0~65°, 적당한 채도, 중간 밝기)을 계산하고, 그만큼 출력을 원래 색으로 되돌립니다. 피부색은 RGB 공간에서 얇은 영역이라 보간 중 보호가 새지 않도록 가중치를 이웃 격자로 한 칸 넓힙니다(3×3×3 dilation).
- **LUT 강도**: `identity × (1 − s) + LUT × s`

| 피부 보호 | 0 | 0.6 | 0.85 | 1.0 |
|---|---|---|---|---|
| 피부 영역 평균 ΔE (강도 0.8, 예시 이미지) | 14.9 | 5.9 | 2.2 | 0.1 |

### LUT 합성과 내보내기

- **전체** LUT 는 각 격자점에 ①②③ 을 적용한 뒤 ④ LUT 를 삼선형 보간으로 통과시켜 하나로 합칩니다. 원본에 이 LUT 하나를 건 결과와 앱 결과의 차이는 평균 ΔE 약 0.7 (육안 구분 불가) 입니다.
- 노드 1 → 2 → 3 LUT 를 차례로 건 결과는 앱 결과와 같습니다 (ΔE 0.00).
- `.cube` 는 표준 순서(**R 이 가장 빠르게 변함**)로 저장되며 `DOMAIN 0~1` 입니다.

---

## 프로젝트 구조

```text
tone_analyzer/
├── README.md
├── pyproject.toml / uv.lock
├── images/                      # 예제 입력 (git 제외)
├── results/                     # 실험 결과 (git 제외)
├── notebooks/                   # 실험용 노트북 (git 제외)
└── src/tone_analyzer/
    ├── app.py                   # Streamlit 웹 앱
    ├── pipeline.py              # 전체 파이프라인 + 최종 LUT 3종 생성(build_final_luts)
    ├── primaries.py             # ①② Lift/Gamma/Gain → Contrast/Pivot 추정·적용
    ├── white_balance.py         # ③ 화이트밸런스(Temp/Tint) 추정·적용
    ├── tone_curve.py            # 히스토그램 매칭 1D 톤 커브
    ├── color_matching.py        # 색 전이(Reinhard / 영역별 / IDT), 피부 가중치, 회귀 모델
    ├── lut.py                   # 3D LUT 생성·적용·합성, 피부 보호, 강도, .cube 저장
    ├── analysis.py              # 밝기·톤 영역·대비 분석
    ├── image_io.py              # 이미지 읽기/저장/검증
    ├── result_io.py             # 분석 결과 JSON 저장
    ├── config.py
    ├── web_uploads/             # 앱 업로드 파일 (작업별 폴더, git 제외)
    └── web_results/             # 앱 결과 파일 (작업별 폴더, git 제외)
```

## 코드에서 사용하기

```python
from tone_analyzer.pipeline import run_tone_matching_pipeline

result = run_tone_matching_pipeline(
    reference_path="images/reference/reference_1.png",
    target_path="images/target/target_1.png",
    output_directory="results/run_01",
    lut_size=33,
    tone_method="primaries",     # "primaries" | "histogram"
    balance_strength=0.5,         # ③ 화이트밸런스 강도
    color_method="reinhard",     # "reinhard" | "idt"
    tonal_regions=True,
    preserve_luminance=True,
    transfer_strength=0.8,
    skin_protection=0.85,
)

print(result["primaries"])       # 추정된 프라이머리 값
print(result["white_balance"])   # Temp / Tint / RGB Gain
print(result["cube_paths"])      # 전체 + 노드별 LUT 경로
```

출력 폴더에는 다음 파일이 생성됩니다.

| 파일 | 내용 |
|---|---|
| `target_tone_matched.png` | ①②③ 까지 적용한 이미지 |
| `target_3d_lut_matched.png` | 최종 이미지 |
| `target_to_reference_{N}.cube` | 전체 ①~④ |
| `target_to_reference_{N}_1_primaries.cube` | 노드 1: ①② |
| `target_to_reference_{N}_2_wb.cube` | 노드 2: ③ |
| `target_to_reference_{N}_3_color.cube` | 노드 3: ④ |
| `analysis_summary.json` | 톤 영역·대비 분석, 프라이머리 값과 진단 정보 |

## 앱 설정 요약

| 구분 | 항목 | 기본값 | 설명 |
|---|---|---|---|
| 학습 설정 | 톤 단계 방식 | 프라이머리 | 또는 히스토그램 매칭 커브 |
| | ③ 화이트밸런스 강도 | 0.5 | 중성색을 Reference 쪽으로 옮기는 정도 |
| | LUT 크기 | 33 | 17 / 33 / 65 |
| | 최대 학습 픽셀 수 | 100,000 | |
| | 색 전이 방식 | reinhard | 또는 idt |
| | 톤 영역별 색 전이 | 켜짐 | reinhard 에서만 |
| | 밝기 유지 (색만 전이) | 켜짐 | |
| 실시간 조절 | LUT 강도 | 0.8 | 재학습 없이 반영 |
| | 피부톤 보호 | 0.85 | 재학습 없이 반영 |

학습 설정을 바꾸면 `톤 매칭 시작`을 다시 눌러야 합니다.

---

## 한계와 주의사항

- **색공간**: 8비트 Rec.709 디스플레이 이미지 기준입니다. 로그/RAW 소스용 입력 변환은 아직 없습니다.
- **프라이머리 값은 근사치**입니다 (위 "DaVinci 수식과의 차이" 참고). 노출 차이가 매우 크면 Gamma 등이 허용 범위 끝(0.4~2.5)에 닿을 수 있습니다.
- **내용 차이는 따라가지 않습니다**: 장면 구성이 다르면(예: Reference 에 검은 옷이 많음) 섀도우 분포가 완전히 같아지지 않습니다. 의도된 동작입니다.
- **피부 판정은 색 기반**입니다. 주황빛 배경이 함께 보호되거나, 진한 립 컬러처럼 채도가 높은 부분은 보호되지 않을 수 있습니다.
- **클리핑된 정보는 복원할 수 없습니다**: 날아간 하이라이트, 뭉개진 섀도우의 디테일은 LUT 로 되살릴 수 없습니다.
- **같은 장면의 카메라 매칭**(같은 조명·구도, ColorChecker 촬영)이라면 분포 전이보다 픽셀/패치 대응 기반 방식이 더 정확합니다. 현재 구현은 서로 다른 장면 간 룩 매칭에 맞춰져 있습니다.
- **업로드 파일은 `web_uploads/` 에 계속 쌓입니다.** 배포 전에는 저장하지 않도록 하거나 주기적으로 정리해야 합니다.

## 다음 개선 후보

- 로그 소스 지원 (S-Log3 / LogC / DaVinci Intermediate 입력 변환, 로그 공간 대비·피벗)
- 회색 램프·컬러 차트 기반 LUT 자동 검증 (밴딩·뒤집힘·클리핑)
- Reference 여러 장의 통계 결합
- 팔레트(k-means) 매칭, 시맨틱 분할 기반 영역별 매칭
- 레퍼런스 유사도 지표 (Lab 히스토그램 거리, 영역별 ΔE2000)
- 테스트 코드 (`tests/`)

## 개발 메모

- Notebook 은 실험용, `src/tone_analyzer/` 의 모듈은 실제 기능용으로 분리합니다.
- Jupyter 커널 등록:

  ```bash
  uv run python -m ipykernel install --user \
    --name tone-analyzer \
    --display-name "Python (tone_analyzer)"
  ```

- 커밋 전 확인: `git status`, `git diff`, `git diff --check`

---

## 라이선스

[MIT License](LICENSE) © 2026 정학용 ([gkrdyd97-sudo](https://github.com/gkrdyd97-sudo))

자유롭게 사용·수정·배포할 수 있으며, **저작권 표시와 라이선스 전문을 함께 유지**해야 합니다.

예제 이미지는 저작권 문제로 저장소에 포함하지 않습니다. 직접 가진 이미지로 테스트해 주세요.
