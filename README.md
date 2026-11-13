# Camera Tone Matching & LUT Generator

두 카메라로 촬영한 이미지를 비교하여 Target 카메라의 계조와 톤을 Reference 카메라에 맞추고, 최종적으로 LUT 파일을 제공하는 웹 애플리케이션을 만드는 프로젝트입니다.

## 최종 사용 흐름

```text
사용자가 홈페이지 접속
        ↓
Reference 이미지 업로드
Target 이미지 업로드
        ↓
밝기·색상·대비 분석
        ↓
Target을 Reference에 맞춤
        ↓
보정 결과 미리보기
        ↓
1D 톤 커브 또는 3D LUT(.cube) 다운로드
```

## 프로젝트 목표

최종 홈페이지에서 사용자가 다음 작업을 할 수 있도록 합니다.

1. Reference 이미지 업로드
2. Target 이미지 업로드
3. 두 이미지의 밝기·대비·색상 차이 분석
4. Shadow / Midtone / Highlight 차이 확인
5. Target을 Reference에 맞춘 결과 확인
6. `.cube` LUT 다운로드

처음에는 JPG 이미지로 학습하고, 이후 Log 원본과 실제 카메라 매칭으로 확장합니다.

## Reference와 Target

- **Reference**: 기준으로 삼을 카메라 또는 이미지
- **Target**: Reference의 계조와 톤에 맞출 카메라 또는 이미지

```text
Reference = 카메라 A
Target    = 카메라 B

Target 이미지
→ Reference와 비슷한 계조와 색상
```

정확한 카메라 매칭을 위해서는 두 이미지가 같은 장면, 같은 시간, 같은 조명에서 촬영된 것이 좋습니다. 가능하면 ColorChecker도 함께 촬영합니다.

## 이 프로젝트는 RAG가 아니다

핵심 기능은 RAG가 아니라 이미지 처리와 색상 변환입니다.

```text
OpenCV + NumPy
→ 이미지 픽셀 분석
→ 밝기와 색상 비교
→ 톤 커브 생성
→ 3D LUT 생성
```

RAG는 문서나 분석 결과를 검색한 뒤 LLM이 자연어로 답변하게 만드는 기술입니다. 나중에 다음 기능을 추가하고 싶다면 RAG를 연결할 수 있습니다.

```text
"기준 샷과 가장 비슷한 이미지를 찾아줘."
"이 Target의 Shadow가 기준보다 얼마나 밝아?"
"대비가 강한 샷만 보여줘."
```

하지만 LUT 생성 자체에는 RAG가 필요하지 않습니다.

## 왜 처음에는 Notebook을 사용하는가?

개발은 `ipynb`와 `.py`를 함께 사용합니다.

### Notebook의 역할

Notebook은 학습과 실험을 위한 공간입니다.

```text
이미지 읽기
→ 결과 확인
→ 그래프 확인
→ 코드 수정
→ 다시 실행
```

Notebook을 사용하는 이유:

- 코드를 셀 단위로 실행할 수 있음
- 이미지와 그래프를 바로 확인할 수 있음
- 중간 계산 결과를 확인하기 쉬움
- Markdown으로 개념과 설명을 함께 기록할 수 있음
- 초보자가 NumPy와 이미지 배열을 이해하기 좋음

### Python 파일의 역할

실제 애플리케이션 기능은 `.py` 파일에 작성합니다.

```text
Notebook에서 실험
→ 작동하는 코드를 함수로 정리
→ src 패키지에 저장
→ 웹 앱에서 import
```

Notebook에 모든 코드를 넣으면 배포와 유지보수가 어려워집니다. 따라서 Notebook은 실험용, Python 모듈은 실제 기능용으로 분리합니다.

## 권장 기술 구조

```text
Notebook
    ↓ 실험이 끝난 코드
Python 모듈
    ↓ import
Streamlit 웹 앱
    ↓ 배포
사용자 홈페이지
```

초기 웹 앱은 Streamlit으로 만듭니다. Streamlit은 Python 코드만으로 파일 업로드, 이미지 표시, 버튼, 다운로드 기능을 만들 수 있어 초보자에게 적합합니다.

## 프로젝트 구조

```text
tone_analyzer/
├── README.md
├── pyproject.toml
├── uv.lock
├── app.py
├── src/
│   └── tone_analyzer/
│       ├── __init__.py
│       ├── image_io.py
│       ├── analysis.py
│       ├── tone_curve.py
│       ├── color_matching.py
│       └── lut.py
├── notebooks/
│   ├── 01_image_loading.ipynb
│   ├── 02_tone_analysis.ipynb
│   ├── 03_reference_target_comparison.ipynb
│   ├── 04_tone_curve.ipynb
│   └── 05_3d_lut.ipynb
├── tests/
│   ├── test_image_io.py
│   └── test_analysis.py
├── images/
├── results/
└── .gitignore
```

## 파일별 역할

### `app.py`

홈페이지의 진입점입니다. 파일 업로드, 분석 버튼, 미리보기, 다운로드 기능을 담당합니다.

### `image_io.py`

이미지를 읽고 RGB, Lab 등 필요한 색공간으로 변환합니다.

### `analysis.py`

Shadow, Midtone, Highlight, 평균 밝기, 채도, 대비, 히스토그램 등을 계산합니다.

### `tone_curve.py`

Target의 밝기를 Reference의 밝기에 맞추는 1D 톤 커브를 만듭니다.

### `color_matching.py`

두 이미지의 RGB 또는 Lab 색상 차이를 계산하고 변환 관계를 만듭니다.

### `lut.py`

3D 색상 변환 모델을 만들고 `.cube` 파일로 저장합니다.

## 설치

```bash
uv add opencv-python numpy pandas matplotlib scipy scikit-learn streamlit ipykernel
```

Jupyter 커널 등록:

```bash
uv run python -m ipykernel install --user \
  --name tone-analyzer \
  --display-name "Python (tone_analyzer)"
```

Notebook에서 `Python (tone_analyzer)` 커널을 선택합니다.

## 전체 개발 로드맵

### Phase 1. Python과 NumPy 기초

학습 내용:

- 변수와 자료형
- 리스트와 딕셔너리
- NumPy 배열
- 이미지의 높이·너비·채널
- 인덱싱과 슬라이싱
- Boolean mask
- 평균과 표준편차
- JSON 저장과 NumPy 자료형 변환

산출물:

```text
Python/NumPy 기초 Notebook
```

### Phase 2. 이미지 읽기와 표시

작업:

1. 파일 경로 설정
2. JPG/PNG 읽기
3. BGR을 RGB로 변환
4. 이미지 크기 확인
5. Reference와 Target을 나란히 표시

커밋:

```text
Add image loading and preview
```

### Phase 3. 밝기와 색공간 분석

작업:

1. RGB를 Lab으로 변환
2. L 채널에서 밝기 추출
3. 밝기 히스토그램 표시
4. 평균·최솟값·최댓값·백분위 계산
5. HSV에서 채도 계산

커밋:

```text
Add lightness and color analysis
```

### Phase 4. Shadow / Midtone / Highlight 분석

밝기 기준으로 세 영역을 나눕니다.

```text
Shadow    → 어두운 영역
Midtone   → 중간 밝기 영역
Highlight → 밝은 영역
```

분석값:

- 각 영역의 픽셀 비율
- 평균 밝기
- 밝기 표준편차
- 평균 RGB
- 평균 채도
- Lab의 a/b 색상 방향

커밋:

```text
Add tonal region analysis
```

### Phase 5. Reference와 Target 비교

```text
Target Shadow - Reference Shadow
Target Midtone - Reference Midtone
Target Highlight - Reference Highlight
Target 대비 - Reference 대비
Target 색상 - Reference 색상
```

결과 예시:

```text
Target은 Reference보다 전체적으로 밝음
Target Shadow가 Reference보다 파란색
Target의 대비가 Reference보다 약함
```

커밋:

```text
Add reference target comparison
```

### Phase 6. 1D 톤 커브 생성

```text
Target 밝기 → Reference 밝기
```

작업:

1. 두 이미지의 밝기 백분위 계산
2. 같은 백분위끼리 대응
3. `np.interp()`로 중간값 보간
4. Target의 Lab L 채널에 적용
5. 보정 이미지 저장

커밋:

```text
Generate one dimensional tone curve
```

### Phase 7. 색상 매칭

Lab과 HSV를 활용하여 다음을 분석합니다.

```text
Shadow가 너무 파란가?
Highlight가 너무 노란가?
Midtone의 채도가 너무 높은가?
```

커밋:

```text
Add color matching analysis
```

### Phase 8. 3D LUT 생성

```text
Target R,G,B → Reference R,G,B
```

작업:

1. 두 이미지를 같은 크기와 구도로 맞춤
2. 같은 위치의 Target/Reference 픽셀 수집
3. RGB를 0~1로 정규화
4. 색상 변환 관계 학습
5. 17³ 또는 33³ 격자 생성
6. 각 격자점의 출력 RGB 예측

커밋:

```text
Build 3D color transform
```

### Phase 9. `.cube` LUT 저장

```text
results/target_to_reference_17.cube
```

커밋:

```text
Export 3D LUT as cube file
```

### Phase 10. Streamlit 홈페이지

웹 화면:

```text
Reference 업로드
Target 업로드
분석 시작 버튼
분석 결과 표
원본과 보정 이미지 비교
LUT 다운로드 버튼
```

로컬 실행:

```bash
uv run streamlit run app.py
```

커밋:

```text
Add Streamlit upload and preview app
```

### Phase 11. 테스트

테스트 대상:

- 이미지 경로 오류
- 지원하지 않는 파일
- 빈 이미지
- 서로 다른 크기의 이미지
- 완전히 검은 이미지
- 완전히 흰 이미지
- LUT 파일 헤더

커밋:

```text
Add analysis and LUT tests
```

### Phase 12. 배포

배포 전 점검:

- API 키와 비밀값이 저장소에 없는가
- 개인 촬영 이미지가 저장소에 없는가
- 의존성이 `pyproject.toml`에 기록되었는가
- 업로드 파일 크기 제한이 있는가
- 업로드 파일을 서버에 영구 저장하지 않는가
- LUT 다운로드가 정상 작동하는가
- 오류 메시지가 이해하기 쉬운가

초기 배포 후보:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- 개인 서버

## 이미지와 Log 원본

학습 단계에서는 JPG와 PNG를 사용합니다. 실제 카메라 매칭에서는 Log 원본을 사용할 수 있지만, 카메라별 감마와 색역을 먼저 해석해야 합니다.

```text
카메라 A Log → 공통 작업 색공간
카메라 B Log → 공통 작업 색공간
        ↓
밝기·색상 비교
        ↓
보정 및 LUT 생성
```

처음에는 Log 원본에 카메라 전용 Log-to-Rec.709 또는 Log-to-ACES 변환을 적용한 뒤 Python 분석을 시작합니다.

## 중요한 제한사항

서로 다른 장면의 이미지를 사용하면 색상 차이가 카메라 때문인지 장면 때문인지 구분하기 어렵습니다.

또한 LUT는 이미 클리핑되어 사라진 정보를 복원할 수 없습니다.

```text
완전히 날아간 Highlight → 원래 색상 복원 불가
완전히 뭉개진 Shadow → 원래 디테일 복원 불가
```

가장 정확한 카메라 매칭을 위해서는 같은 조명 아래에서 ColorChecker를 촬영하는 것을 권장합니다.

## Git 커밋 전략

각 단계가 작동할 때마다 하나의 커밋을 남깁니다.

```text
Initialize project documentation
Add image loading and preview
Add lightness and color analysis
Add tonal region analysis
Add reference target comparison
Generate one dimensional tone curve
Add tone curve application
Build 3D color transform
Export 3D LUT as cube file
Add Streamlit upload app
Add tests and deployment configuration
```

커밋 전 확인:

```bash
git status
git diff
git diff --check
```

커밋:

```bash
git add 파일명
git commit -m "변경 내용"
```

## 현재 시작점

우리는 다음 순서로 직접 코드를 작성합니다.

```text
1. Python과 NumPy 기초
2. 이미지 읽기와 표시
3. Lab 밝기 분석
4. Shadow/Midtone/Highlight 분석
5. Reference/Target 비교
6. 1D 톤 커브
7. 색상 매칭
8. 3D LUT
9. Streamlit 홈페이지
10. 배포
```

각 단계에서 먼저 Notebook에 전체 코드를 작성하고, 코드가 작동하는 것을 확인한 다음 설명과 함께 Python 모듈로 옮깁니다.
