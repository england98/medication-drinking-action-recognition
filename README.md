# medication-drinking-action-recognition

복약 / 음수(물 마시기) / 기타 행동을 짧은 영상에서 분류하는 **경량 비전 모델 1차 Pilot 프로젝트**입니다.

현재 목표는 전체 데이터에서 최고 성능을 만드는 것이 아니라, 축소된 대표 데이터로 전체 파이프라인을 한 번 완주하여 설계와 구현의 유효성을 검증하는 것입니다.

---

## 1. Pilot 목표

```text
환경·경로·Git 기준선
→ Full Candidate Inventory
→ Fixed Pilot Manifest
→ ROI Preflight
→ Stage A Visual Encoder
→ ETRI Embedding Cache
→ Stage B Clip Classifier
→ 2×2 Ablation
→ participant-disjoint 5-fold CV
→ deployment/check model
→ ETRI + self_recorded inference pipeline check
```

최종 출력 class:

```text
복약
음수 = 물 마시기
기타
```

---

## 2. Baseline 모델

```text
Input Video
↓
Frame Sampling
↓
MediaPipe ROI / Fallback
↓
MobileNetV3-Small
↓
Frame Embedding Sequence
↓
Mean Pooling + Linear
        VS
GRU + Linear
↓
복약 / 음수 / 기타
```

1차 Pilot의 핵심 비교:

| Encoder | Stage B |
|---|---|
| ImageNet-only | Mean Pooling + Linear |
| AI-Hub fine-tuned | Mean Pooling + Linear |
| ImageNet-only | GRU + Linear |
| AI-Hub fine-tuned | GRU + Linear |

ETRI 학습 중 visual encoder는 frozen 상태로 사용합니다.

---

## 3. 데이터 역할

### AI-Hub

사용 목적:

```text
Stage A visual encoder 학습
```

기준:

- Local Training + Validation JSON/File 18,420건이 inventory master
- `viewpoint_3` only
- actor-disjoint split
- 동일 video의 JPG 3장은 같은 split

Stage A mapping:

```text
Take_pills        → 복약
Drink_bever       → 음수 auxiliary positive
Drink_alcohol     → 음수 auxiliary positive
나머지            → 기타
```

### ETRI-Activity3D-LivingLab

사용 목적:

```text
Stage B clip-level classifier 학습·평가
```

기준:

- Batch B only
- participant-disjoint 5-fold
- A003 = 복약
- A004 = 물 마시기
- 나머지 = 기타
- A045~A048 multi-person action은 Pilot에서 제외

---

## 4. 개발환경

- Host: Windows
- WSL 2
- Ubuntu 26.04.1 LTS
- Python 3.12.14
- `uv`
- PyTorch 2.13.0+cu126
- CUDA runtime 12.6
- GPU: NVIDIA GeForce GTX 1650 Ti / 4GB
- VS Code + WSL

Python Interpreter:

```text
<WSL_PROJECT_ROOT>/.venv/bin/python
```

환경 상세:

```text
docs/01_개발환경_구축_기록.md
```

---

## 5. Project Root

```text
/home/user/projects/medication-drinking-action-recognition
```

대용량 Raw / Working 데이터는 Project Root 밖의 외부 SSD에 둡니다.

Raw Data:

```text
/mnt/d/AI 도약과정/데이터/data_raw/
```

Working Data:

```text
/mnt/d/AI 도약과정/데이터/data_workspace/medication_drinking_action_data
```

Raw 데이터는 수정하지 않습니다.

---

## 6. 프로젝트 구조

```text
.
├── configs/
├── docs/
├── manifests/
├── scripts/
├── src/
├── tests/
├── runtime/
├── .gitignore
├── .python-version
├── requirements-lock.txt
├── AGENTS.md
├── README.md
└── STATUS.md
```

필요한 디렉토리는 구현 단계에 맞춰 생성합니다.

Phase 3 Fixed Pilot Manifest 생성 및 validation:

```bash
.venv/bin/python -m scripts.build_pilot_manifests
```

선정 정책은 `configs/pilot_manifest.yaml`에서 관리하며, 전체 candidate row를 유지한
manifest와 selected-only CSV 및 SHA-256 요약은 Working Data의 `manifests/pilot/`에 생성됩니다.

Phase 4 ROI Preflight 실행(공식 MediaPipe 모델이 Working cache에 없으면 함께 다운로드):

```bash
.venv/bin/python -m scripts.run_roi_preflight --download-models
```

MediaPipe Tasks 초기화에는 WSL의 `libGLESv2.so.2`가 필요합니다. 누락된 환경에서는 사용자가
먼저 시스템 패키지 `libgles2`를 설치해야 합니다.

대표 sample 선정과 ROI 기본값은 `configs/roi_preflight.yaml`에서 관리하며, visual output과
report는 Working Data의 `roi_preflight/`에 생성됩니다. 통계와 visual review 후 사용자가
PASS 여부를 판단하기 전까지 전체 Pilot preprocessing을 시작하지 않습니다.

Phase 5 Stage A 구현과 사용자 실행 명령은 다음 문서에 정리되어 있습니다.

```text
docs/04_Phase5_Stage_A_실행_가이드.md
```

Phase 6 ETRI embedding preflight의 실제 1-clip Encoder A/B smoke test:

```bash
.venv/bin/python -m scripts.run_etri_embedding_smoke --output /tmp/phase6-etri-embedding-smoke.pt
```

이 명령은 전체 cache를 만들지 않고 Fixed ETRI Pilot clip 1개만 T=64로 처리합니다.

Phase 6 multi-clip preflight와 전체 239-clip cache 생성에 사용한 재현 명령:

```bash
.venv/bin/python -m scripts.run_etri_embedding_cache --limit 3 --resume
.venv/bin/python -m scripts.run_etri_embedding_cache --resume
```

정상 cache만 resume하며, 손상되거나 provenance가 다른 기존 cache는 명시적으로 실패합니다.
Phase 6 전체 239-clip cache와 validation gate는 PASS했으며, 현재 상태는
`Phase 6 COMPLETE / Phase 7 READY TO START`입니다. 상세 결과와 warning은 `STATUS.md`를 따릅니다.

---

## 7. 주요 문서

### 상위 설계 기준

```text
docs/00_Pilot_Design_Baseline.md
```

모델·데이터·평가·Pipeline·Pilot 실행 순서의 최상위 기준입니다.

### 개발환경 기록

```text
docs/01_개발환경_구축_기록.md
```

현재 Python / WSL / GPU / CUDA / 경로 / MLflow 환경 기준을 기록합니다.

### 모델 구현 Reference

```text
docs/03_Model_Implementation_References.md
```

Phase 5~10의 모델·학습·평가·inference 구현에서 사용할 PyTorch / torchvision 공식 API와
승인된 외부 GitHub Reference의 참고·채택 범위를 정의합니다.

### Coding Agent 지침

```text
AGENTS.md
```

Coding Agent의 작업 권한, 실행 범위, 금지 사항, 작업 보고 방식을 정의합니다.

### 현재 진행 상태

```text
STATUS.md
```

현재 Phase, 완료 작업, 진행 중 작업, 다음 작업, blocker를 기록하는 **작업 상태의 Single Source of Truth**입니다.

### 문서 역할 요약

| 문서 | 역할 |
|---|---|
| `AGENTS.md` | Coding Agent 작업 규칙 |
| `README.md` | 프로젝트의 안정적인 개요 |
| `STATUS.md` | 현재 작업 진행 상태 |
| `docs/00_Pilot_Design_Baseline.md` | Pilot 상위 설계 기준 |
| `docs/03_Model_Implementation_References.md` | Phase 5~10 모델 구현 공식 API·외부 Reference 기준 |

일상적인 작업 진행에 따라서는 `STATUS.md`를 갱신하고, 프로젝트 규칙·구조·사용법 자체가 변경될 때만 `AGENTS.md` 또는 `README.md`를 수정합니다.

---

## 8. 작업 방식

기본 역할 분담:

```text
Coding Agent
→ 코드 작성·수정
→ config / test 작성
→ 짧은 검증
→ 사용자 실행 명령 제공

사용자
→ 전체 데이터 scan
→ 대규모 preprocessing
→ GPU 학습
→ embedding 생성
→ CV / ablation / inference 등 주요 실행
```

상세 규칙은 `AGENTS.md`를 따릅니다.

---

## 9. 평가 기준

ETRI의 정량 모델 선택 기준:

```text
participant-disjoint 5-fold CV
```

Primary metric:

```text
5-fold mean Macro-F1
```

함께 기록:

- Macro-F1 mean ± std
- class별 Recall mean ± std
- fold별 Confusion Matrix
- OOF aggregate Confusion Matrix

별도의 untouched final test 성능으로 과장하지 않습니다.

---

## 10. self_recorded 데이터

경로:

```text
<SSD_WORK_ROOT>/self_recorded/pipeline_check/
```

용도:

- inference pipeline 동작 검증
- label / confidence 생성 확인
- 기능적 오류 확인
- 정성 확인

사용하지 않는 용도:

- 학습
- validation
- test metric
- model selection
- threshold tuning
- preprocessing tuning

---

## 11. Project Status

현재 Phase, 완료 작업, 진행 중 작업, 다음 작업, blocker 등 **상세 작업 상태는 `STATUS.md`를 Single Source of Truth로 관리합니다.**

`README.md`에는 상세 진행 상황을 중복 기록하지 않습니다.

```text
STATUS.md
```
