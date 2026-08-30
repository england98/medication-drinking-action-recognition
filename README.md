# medication-drinking-action-recognition

짧은 RGB 영상에서 **복약 / 음수(물 마시기) / 기타** 행동을 분류하는 경량 비전 모델 1차 Pilot 프로젝트입니다.

현재 상태는 **1st Pilot COMPLETE**입니다. 이는 제한된 Pilot 데이터에서 설계한 학습·평가·inference pipeline을 완주했다는 뜻이며, production 완성이나 임상적 성능, 광범위한 일반화 성능을 의미하지 않습니다. 상세 Phase 이력과 현재 작업 상태는 [`STATUS.md`](STATUS.md)를 따릅니다.

## 1. Overview

최종 class taxonomy는 다음과 같습니다.

| Index | Class | 의미 |
|---:|---|---|
| 0 | 복약 | 약 먹기 |
| 1 | 음수 | 물 마시기 |
| 2 | 기타 | 위 두 행동이 아닌 행동 |

이 프로젝트에서 **End-to-End**는 다음의 전체 inference path를 뜻합니다.

```text
raw video
→ frame sampling
→ ROI / fallback
→ visual encoder
→ Stage B classifier
→ 복약 / 음수 / 기타
```

CNN과 temporal model을 함께 최적화하는 joint end-to-end video training을 의미하지 않습니다.

## 2. Architecture

최종 선택 구조는 **Experiment D**입니다.

```text
Input Video
→ Fixed-uniform Frame Sampling (T=64)
→ MediaPipe ROI / Full-frame Fallback
→ AI-Hub fine-tuned MobileNetV3-Small Encoder B (frozen)
→ Frame Embedding Sequence [T, 1024]
→ GRU (hidden size 128, final hidden)
→ Linear 3-class Classifier
→ 복약 / 음수 / 기타
```

핵심 사양:

- visual encoder: ImageNet pretrained 후 AI-Hub에서 fine-tuned한 MobileNetV3-Small Encoder B
- encoder 상태: ETRI Stage B 학습과 inference에서 frozen
- embedding dimension: `D=1024`
- sequence length: `T=64`
- Stage B: 1-layer, unidirectional GRU (`hidden_size=128`) + Linear
- tensor contract: `[B, 3, H, W] → [B, 1024]`, `[B, 64, 1024] → [B, 3]`

### Model selection

동일한 manifest, participant folds, ROI, sampling, normalization, loss, seed 정책과 frozen encoder 조건에서 수행한 2×2 ablation 결과입니다. 표준편차는 population standard deviation입니다.

| Exp | Encoder | Stage B | 5-fold Macro-F1 mean ± std |
|---|---|---|---:|
| A | ImageNet-only | Mean Pooling + Linear | 0.450650 ± 0.079355 |
| B | AI-Hub fine-tuned | Mean Pooling + Linear | 0.449102 ± 0.087356 |
| C | ImageNet-only | GRU + Linear | 0.518992 ± 0.042044 |
| **D** | **AI-Hub fine-tuned** | **GRU + Linear** | **0.532176 ± 0.054575** |

사전에 고정한 primary metric인 5-fold mean Macro-F1이 가장 높은 Experiment D를 선택했습니다. 상세 근거와 machine-readable configuration은 [`docs/05_Phase8_Structure_Selection_Result.md`](docs/05_Phase8_Structure_Selection_Result.md)와 [`configs/phase8_selected_model.yaml`](configs/phase8_selected_model.yaml)에 있습니다.

## 3. Data

### AI-Hub

Stage A visual encoder 학습에 사용했습니다.

- `viewpoint_3` only
- actor-disjoint split
- selected Pilot: 400 videos
- `Take_pills → 복약`
- `Drink_bever`, `Drink_alcohol → 음수 auxiliary positive`
- 나머지 행동 `→ 기타`

AI-Hub 음수 데이터는 손-용기-입의 시각 특징을 학습하기 위한 auxiliary class입니다. 최종 음수 target의 의미는 ETRI A004의 **물 마시기**입니다.

### ETRI-Activity3D-LivingLab

Stage B clip-level classifier 학습과 정량 평가에 사용했습니다.

- Batch B only
- Fixed Pilot selected-valid subset: 239 clips / 30 participants
- participant-disjoint 5-fold
- `A003 → 복약`, `A004 → 음수`, 나머지 `→ 기타`
- multi-person action `A045~A048` 제외

### Self-recorded

직접 촬영한 3개 영상은 전체 inference path와 label/confidence 생성을 확인하는 functional/qualitative check에만 사용했습니다. 세 영상 모두 pipeline 실행에는 성공했지만 모두 `기타`로 예측되어 복약·음수 intended sample은 일치하지 않았습니다.

Self-recorded 데이터는 학습, validation, model selection, tuning 또는 성능 지표 산출에 사용하지 않았으며 accuracy나 일반화 성능의 근거가 아닙니다.

## 4. Pilot Results

공식 정량 평가는 Experiment D의 fold-specific checkpoint를 사용한 **ETRI Batch B Pilot participant-disjoint 5-fold raw-video OOF End-to-End evaluation**입니다.

| Metric | Result |
|---|---:|
| Raw MP4 inference | 239 / 239 성공 |
| Participant leakage | 0 |
| Duplicate / missing / failure | 0 / 0 / 0 |
| Aggregate raw-video OOF Macro-F1 | 0.538337 |
| Fold Macro-F1 mean ± population std | 0.532176 ± 0.054575 |
| 복약 Recall (aggregate OOF) | 0.355932 |
| 음수 Recall (aggregate OOF) | 0.333333 |
| 기타 Recall (aggregate OOF) | 0.900000 |

Aggregate OOF Macro-F1은 239개 prediction을 합쳐 계산한 값이고, fold mean ± std는 fold별 Macro-F1 다섯 개의 통계이므로 서로 대체하지 않습니다. Raw-video와 cached evaluation은 239개 prediction과 metric을 동일하게 재현했습니다.

이 결과는 architecture selection에도 사용한 participant-disjoint 5-fold Pilot evidence입니다. 별도의 untouched independent final test 결과가 아니며, production·clinical 성능이나 외부 데이터 일반화 성능으로 해석하지 않습니다. 상세 metric, confusion matrix, 오류 분석과 해석 경계는 [`docs/06_Pilot_Final_Evaluation.md`](docs/06_Pilot_Final_Evaluation.md)를 참조하십시오.

## 5. Quick Start

### Environment

이 프로젝트는 WSL 2, Python 3.12, PyTorch/torchvision, CUDA, MediaPipe 환경에서 개발·검증했으며 기준 GPU는 NVIDIA GeForce GTX 1650 Ti 4GB입니다. Python 환경은 `uv`와 프로젝트의 `.venv`를 사용합니다.

- 정확한 package snapshot: [`requirements-lock.txt`](requirements-lock.txt)
- 환경·경로 상세: [`docs/01_개발환경_구축_기록.md`](docs/01_개발환경_구축_기록.md)
- 공유 경로 예시: [`configs/paths.example.yaml`](configs/paths.example.yaml)
- 현재 PC의 실제 경로: `configs/paths.local.yaml` (Git 제외)

대용량 Raw/Working 데이터와 checkpoint는 repository 밖에 두며, Raw 데이터는 수정하지 않습니다.

### Single Video Inference

기본 단일 영상 inference는 selected-valid Pilot 239개 전체로 다시 학습한 `deployment_check.pt`를 사용합니다. 입력 경로와 새 JSON 출력 경로를 지정합니다.

```bash
.venv/bin/python scripts/run_inference.py "<video.mp4>" \
  --output-json "<work_root>/evaluations/inference/<name>.json"
```

`configs/paths.local.yaml`이 아닌 다른 path config를 쓰려면 `--paths-config`를 추가할 수 있습니다. 출력 JSON은 기존 파일이나 Raw 데이터 경로를 덮어쓰지 않습니다.

`deployment_check.pt`는 선택된 D 구조의 Stage B를 전체 Pilot 데이터로 다시 학습한 functional/qualitative pipeline-check용 단일 checkpoint입니다. best CV checkpoint, independent performance checkpoint 또는 공식 정량 평가 checkpoint가 아닙니다.

### Raw-video OOF Re-evaluation

공식 평가 절차를 재검증할 때는 Experiment D의 fold별 held-out checkpoint를 사용합니다. 이 명령은 239개 raw MP4 전체를 처리하는 장시간 작업이므로 사용자가 직접 실행하며, 기존 공식 artifact와 다른 새 output directory를 지정해야 합니다.

```bash
.venv/bin/python scripts/run_phase10_raw_video_oof.py \
  --output-root "<work_root>/evaluations/phase10_raw_video_oof_recheck"
```

`--output-root`는 configured `<work_root>` 내부여야 하고, 이미 존재하는 directory는 overwrite하지 않습니다.

## 6. Project Structure

```text
.
├── configs/               # 공유 pipeline·experiment configuration
├── docs/                  # 설계, 환경, 실행 및 최종 결과 문서
├── manifests/             # 소형 manifest metadata와 schema
├── scripts/               # 학습·평가·inference CLI
├── src/                   # 모델과 data pipeline 구현
├── tests/                 # unit·regression test
├── runtime/               # 로컬 runtime 산출물 (Git 제외)
├── requirements-lock.txt  # 환경 package snapshot
├── AGENTS.md              # Coding Agent 작업 규칙
├── README.md              # 프로젝트의 안정적인 개요
└── STATUS.md              # 상세 진행 상태의 Single Source of Truth
```

## 7. Evaluation Protocol

- ETRI Fixed Pilot 239개를 participant-disjoint 5-fold로 평가합니다.
- 동일 participant의 모든 clip은 같은 fold에 속합니다.
- primary model-selection metric은 5-fold mean Macro-F1입니다.
- 공식 raw-video OOF에서는 각 sample을 그 participant가 학습에 포함되지 않은 fold-specific Experiment D checkpoint로 정확히 한 번 평가합니다.
- 별도의 untouched independent final test는 없습니다.
- 전체 Pilot로 재학습한 `deployment_check.pt`는 기능·정성 확인용이며 공식 정량 평가에는 사용하지 않습니다.

## 8. Limitations

- model selection에 사용되지 않은 untouched independent final test가 없습니다.
- 복약 Recall `0.355932`, 음수 Recall `0.333333`으로 target action sensitivity가 낮습니다.
- ETRI Pilot subset이 239 clips / 30 participants로 작습니다.
- ETRI에서는 visual encoder를 frozen 상태로 사용했으며 domain-specific fine-tuning을 평가하지 않았습니다.
- ETRI의 ROI 처리는 대부분 `partial` status에 의존했습니다.
- external dataset 및 ETRI cross-batch generalization을 평가하지 않았습니다.
- self-recorded 3개 결과는 qualitative-only이며 성능 근거가 아닙니다.

## 9. Documentation

| 문서 | 역할 |
|---|---|
| [`docs/00_Pilot_Design_Baseline.md`](docs/00_Pilot_Design_Baseline.md) | Pilot의 상위 설계·데이터·평가 기준 |
| [`docs/01_개발환경_구축_기록.md`](docs/01_개발환경_구축_기록.md) | WSL, Python, GPU, 경로와 환경 기록 |
| [`docs/03_Model_Implementation_References.md`](docs/03_Model_Implementation_References.md) | 모델 구현 공식 API와 외부 Reference 범위 |
| [`docs/05_Phase8_Structure_Selection_Result.md`](docs/05_Phase8_Structure_Selection_Result.md) | 2×2 ablation과 Experiment D 선택 근거 |
| [`docs/06_Pilot_Final_Evaluation.md`](docs/06_Pilot_Final_Evaluation.md) | 최종 평가, 오류 분석, limitation과 해석 경계 |
| [`AGENTS.md`](AGENTS.md) | Coding Agent의 작업 권한과 절차 |
| [`STATUS.md`](STATUS.md) | 상세 개발 이력, 현재 상태와 다음 작업 |

`README.md`는 완료된 프로젝트의 안정적인 overview를, `STATUS.md`는 상세 진행 상태의 Single Source of Truth를 담당합니다.
