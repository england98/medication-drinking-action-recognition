# medication-drinking-action-recognition

복약 / 음수(물 마시기) / 기타 행동을 짧은 영상에서 분류하는 **경량 비전 모델 1차 Pilot 프로젝트**입니다.

Baseline에서 정의한 구현·평가·검증 절차를 완주하여 **Phase 10과 1차 Pilot을 완료**했습니다.
이는 축소된 Pilot 범위의 완료이며 production 완성이나 광범위한 일반화 성능 달성을 뜻하지 않습니다.

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

최종 선택 구조는 **Experiment D**입니다.

```text
AI-Hub fine-tuned MobileNetV3-Small Encoder B (frozen, D=1024)
+ GRU (hidden size 128, final hidden, T=64)
+ Linear 3-class classifier
```

Phase 7의 fold 0~4 모델은 participant-disjoint OOF 평가에 사용했습니다. Phase 9의 단일
`deployment_check.pt`는 selected-valid 239개 전체로 Stage B를 다시 학습한 Phase 10
functional/qualitative pipeline check용 모델이며, 독립 성능 checkpoint가 아닙니다.

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
- Selected Pilot 400 videos: distinct actors 192명 (train 152 / validation 40 / overlap 0)
- Candidate-pool split population: 202명 (train-split 162 / validation-split 40)
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
- Fixed Pilot selected-valid subset 239 clips / 30 participants
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
Phase 6 전체 239-clip cache와 validation gate는 PASS했으며 Phase 6은 COMPLETE입니다.

Phase 7 Stage B 2×2 ablation 검증 진입점:

```bash
.venv/bin/python scripts/run_phase7_ablation.py --validate-only
```

Pre-run Independent Audit 이후 Mean/GRU 최소 smoke에 사용한 명령입니다.

```bash
.venv/bin/python scripts/run_phase7_ablation.py --experiment A --fold 0 --smoke --epochs 1 --no-mlflow
.venv/bin/python scripts/run_phase7_ablation.py --experiment C --fold 0 --smoke --epochs 1 --no-mlflow
```

Mean/GRU/MLflow smoke와 정식 A/B/C/D × 5-fold CV를 완료했습니다. Production run은 20/20
FINISHED, participant leakage는 0이며 A/B/C/D OOF는 각각 239개 exact-once입니다. Post-run
Independent Final Audit은 `PASS_WITH_WARNINGS`이고 metric 독립 재계산은 exact match했습니다.

5-fold Macro-F1 mean ± std:

```text
A: 0.4507 ± 0.0794
B: 0.4491 ± 0.0874
C: 0.5190 ± 0.0420
D: 0.5322 ± 0.0546
```

현재 상태는 `Phase 10 COMPLETE / 1st Pilot COMPLETE`입니다. Phase 8은 고정된 5-fold mean
Macro-F1 기준으로 Experiment D(AI-Hub fine-tuned Encoder + GRU)를 선택했습니다. Machine-readable
handoff는 `configs/phase8_selected_model.yaml`, 상세 근거는
`docs/05_Phase8_Structure_Selection_Result.md`에 있습니다.

정식 Full CV에 사용한 실행 명령:

```bash
.venv/bin/python scripts/run_phase7_ablation.py --all-experiments
```

Phase 8 선택을 재현하려면 다음 명령을 사용합니다. 동일 evidence에서 기존 artifact와 값이 다르면
overwrite하지 않고 실패합니다.

```bash
.venv/bin/python scripts/run_phase8_selection.py
```

Phase 9 — Pilot Deployment/Check Model은 완료되었습니다. Frozen Encoder B와 새로 초기화한 GRU를
전체 selected-valid Pilot 239개/30명에 fold filter 없이 fixed 15 epochs로 학습했습니다. 생성된
`deployment_check.pt`의 독립 reload verification과 Phase 9 Independent Final Audit은 PASS했습니다.
이 checkpoint는 Phase 10 integration/pipeline check용이며 best CV 또는 공식 성능 checkpoint가 아닙니다.

다음 명령은 완료된 Phase 9 경로의 재현/재검증용입니다.

```bash
.venv/bin/python scripts/run_phase9_deployment.py --dry-run
.venv/bin/python scripts/run_phase9_deployment.py --train
.venv/bin/python scripts/run_phase9_deployment.py \
  --verify-checkpoint "<work_root>/checkpoints/phase9_deployment/phase9_deployment_full_pilot/deployment_check.pt"
```

Phase 9 training metric은 training diagnostic일 뿐 공식 정량 성능이 아닙니다. 공식 정량평가는
**ETRI Batch B Fixed Pilot participant-disjoint 5-fold raw-video OOF End-to-End evaluation**입니다.

Phase 10 결과:

- Raw MP4 239/239 inference 성공, participant leakage/duplicate/missing/failure 모두 0
- Aggregate OOF Macro-F1: `0.538337`
- Fold Macro-F1 mean ± population std: `0.532176 ± 0.054575`
- Class Recall: 복약 `0.355932`, 음수 `0.333333`, 기타 `0.900000`
- Raw-vs-cached sample/prediction/metric/confusion matrix 일치; prediction agreement `239/239`
- Probability는 max absolute difference `1.2517e-6`의 미세 차이가 있어 stored verdict
  `DIFFERENCE_OBSERVED`를 유지
- Raw-video OOF는 CPU, self-recorded 3-class pipeline check는 CUDA에서 실행
- Self-recorded 3개 모두 label/confidence 생성과 전체 경로가 정상 동작했으나, 이는 정성적 기능
  검증일 뿐 accuracy가 아님

핵심 limitation은 untouched independent final test 부재, 낮은 복약/음수 Recall, 239개 Pilot subset의
작은 규모, ETRI에서 frozen encoder 사용, 대부분 partial인 ROI, external/cross-batch generalization
미평가, self-recorded의 qualitative-only 사용입니다. 상세 결과와 해석 경계는
`docs/06_Pilot_Final_Evaluation.md`를 따릅니다.

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

### 1차 Pilot 최종 평가

```text
docs/06_Pilot_Final_Evaluation.md
```

1차 Pilot 최종 정량평가, error analysis, limitation 및 interpretation boundary를 기록합니다.

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
| `docs/06_Pilot_Final_Evaluation.md` | 1차 Pilot 최종 평가·오류 분석·해석 경계 |

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
