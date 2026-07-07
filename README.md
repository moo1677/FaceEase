# FaceEase

FaceEase는 웹캠으로 얼굴 표정을 분석해 자연스러움 점수와 피드백을 제공하는 브라우저 앱입니다. MediaPipe Face Landmarker로 얼굴 blendshape feature를 추출하고, 학습된 로컬 모델을 `public/models/faceease_models.json`으로 내보내 Vite 앱에서 바로 추론합니다.

## 주요 기능

- 웹캠 기반 실시간 얼굴 landmark 및 blendshape 분석
- 무표정 기준값 수집 후 과제별 5초 표정 분석
- 지원 과제: 자연스러운 미소, 경청하는 표정, 차분한 표정
- 최종 자연스러움 점수, 과제 수행 점수, 표정 안정도, 피드백 표시
- `data_set` 영상과 `Labeling.xlsx` 라벨을 이용한 모델 재학습

## 요구 사항

- Node.js
- npm
- Python 3.10 이상 권장
- 웹캠 권한을 허용할 수 있는 브라우저

## 설치

```bash
npm install

python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

이미 `.venv`와 `node_modules`가 준비되어 있다면 위 설치 단계는 생략할 수 있습니다.

## 실행

```bash
npm run dev
```

Vite 개발 서버가 뜨면 브라우저에서 안내된 로컬 주소로 접속합니다. 카메라 API는 보통 `localhost` 또는 HTTPS 환경에서 정상 동작합니다.

앱 사용 순서:

1. `카메라 시작`을 누릅니다.
2. `무표정`을 눌러 얼굴 기준값을 수집합니다.
3. `미소`, `경청`, `차분` 중 분석할 과제를 선택합니다.
4. `5초 분석 시작`을 눌러 결과를 확인합니다.

## 빌드

```bash
npm run build
npm run preview
```

빌드 결과물은 `dist/`에 생성됩니다.

## 모델 재학습

기본 학습 데이터는 다음 위치를 사용합니다.

- 영상 데이터: `data_set/*.mp4`
- 라벨 파일: `Labeling.xlsx`
- 학습 산출물: `output/`
- 브라우저용 모델: `public/models/faceease_models.json`

전체 모델을 다시 만들고 브라우저용 JSON까지 갱신하려면:

```bash
npm run ml:refresh
```

단계별로 실행하려면:

```bash
npm run ml:train
npm run ml:export
```

`ml:train`은 영상 feature 추출, 라벨 조인, 회귀/분류 모델 학습을 수행합니다. `ml:export`는 학습된 `.pkl` 모델을 브라우저에서 사용할 수 있는 JSON 형식으로 변환합니다.

주요 산출물:

- `output/faceease_score_model.pkl`
- `output/faceease_negative_pattern_model.pkl`
- `output/faceease_training_report.json`
- `output/faceease_labeled_dataset.csv`
- `public/models/faceease_models.json`

## 데이터 규칙

영상 파일명은 사용자 ID와 표정명을 `_`로 구분합니다.

```text
사용자ID_자연스러운미소.mp4
사용자ID_경청하는표정.mp4
사용자ID_차분한표정.mp4
사용자ID_어색한표정.mp4
사용자ID_과하게웃는표정.mp4
```

스크립트는 위 표정명을 다음 task type으로 매핑합니다.

| 표정명 | taskType |
| --- | --- |
| 자연스러운미소 | `natural_smile` |
| 경청하는표정 | `listening` |
| 차분한표정 | `calm` |
| 어색한표정 | `awkward` |
| 과하게웃는표정 | `over_smile` |

`Labeling.xlsx`는 `score` 컬럼이 있는 시트를 우선 사용합니다. 없으면 `Raw_Clean` 시트의 `영상번호`, `표정`, `avg` 컬럼을 읽어 라벨을 구성합니다.

## 프로젝트 구조

```text
.
├── index.html
├── src/
│   ├── main.js
│   └── styles.css
├── public/models/
│   └── faceease_models.json
├── scripts/
│   ├── faceease_ml_pipeline.py
│   ├── export_faceease_models.py
│   └── faceease_service.py
├── data_set/
├── output/
├── Labeling.xlsx
├── package.json
└── requirements.txt
```

## 문제 해결

- `분석 모델 로딩 실패`가 표시되면 `npm run ml:export`를 실행해 `public/models/faceease_models.json`을 갱신합니다.
- 카메라가 시작되지 않으면 브라우저 권한과 접속 주소가 `localhost`인지 확인합니다.
- 학습 중 라벨 조인 결과가 비어 있으면 `data_set`의 영상 파일명과 `Labeling.xlsx`의 영상명/표정명이 일치하는지 확인합니다.
- MediaPipe 모델 다운로드가 실패하면 네트워크 연결을 확인한 뒤 `npm run ml:train`을 다시 실행합니다.
