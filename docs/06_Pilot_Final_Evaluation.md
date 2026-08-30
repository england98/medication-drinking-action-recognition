# Pilot Final Evaluation

**복약 / 음수 / 기타 3-class 행동 인식 모델 — 1차 Pilot 최종 평가 기록**

- 작성일: 2026-08-30
- 평가 대상: ETRI Batch B Fixed Pilot Manifest
- 문서 상태: **FINAL — Phase 10 COMPLETE / 1st Pilot COMPLETE**
- 상위 설계 기준: `docs/00_Pilot_Design_Baseline.md`
- 정량 평가 명칭: **ETRI Batch B Pilot participant-disjoint 5-fold raw-video OOF End-to-End evaluation**

이 문서는 새 실험 결과가 아니라 repository와 `<work_root>`에 이미 존재하는 machine-readable artifact,
checkpoint provenance, config, 구현 및 MLflow 기록을 교차검증한 공식 Pilot 평가 기록이다. 이 문서에서
`End-to-End`는 CNN과 temporal model의 joint training이 아니라 raw video부터 3-class 출력까지의 전체
inference path를 뜻한다.

# 1. Executive Summary

1차 Pilot은 제한된 데이터와 GTX 1650 Ti 4GB 환경에서 데이터 inventory, leakage-safe manifest,
ROI, visual encoder, temporal classifier, participant-disjoint CV, 단일 deployment/check model 및 실제
영상 inference를 한 번 완주해 설계와 구현의 유효성을 확인하는 것이 목적이었다.

실제로 선택된 구성은 **Experiment D: AI-Hub fine-tuned MobileNetV3-Small Encoder B + GRU**다.
Encoder는 ETRI Stage B 학습과 inference에서 frozen이며, 입력은 `T=64`, embedding은 `D=1024`,
GRU hidden size는 128이다. Phase 7의 primary metric인 5-fold mean Macro-F1은
`0.532176 ± 0.054575`였고, Phase 10 raw-video aggregate OOF Macro-F1은 `0.538337`이었다.
두 수치는 계산 단위가 다르므로 서로 대체하지 않는다.

Phase 10 raw-video OOF 평가는 ETRI Fixed Pilot 239개를 각 participant가 학습에 포함되지 않은 해당
fold checkpoint로 정확히 한 번씩 평가했다. 239/239 inference 성공, participant leakage 0,
duplicate 0, missing 0이었다. Cached OOF와 raw-video OOF는 sample set, 239개 prediction, 모든 metric과
confusion matrix가 동일했다. 확률에는 최대 `1.2517e-6`의 미세한 차이가 있어 사전 구현된 엄격한
`1e-6` 기준의 artifact verdict는 `DIFFERENCE_OBSERVED`로 보존한다.

Self-recorded 3개 영상도 Phase 9 single deployment/check model로 CUDA End-to-End 실행에 성공했다.
다만 의도한 복약과 음수 영상도 모두 기타로 예측됐으므로 기능적 연결은 검증됐지만 분류 정확성은
확인되지 않았다. 이 극소수 정성 결과로 metric을 계산하거나 tuning하지 않았다.

핵심 한계는 untouched final test 부재, CV를 선택과 성능 추정에 함께 사용한 점, 낮은 복약/음수
recall, 239개 Pilot subset, 대부분 partial ROI, ETRI domain에서 encoder를 fine-tune하지 않은 점,
외부·cross-batch 일반화 미검증이다. Independent Phase 10 Final Audit의 초기 verdict는
**PASS WITH REQUIRED FIXES**였고, BLOCKER는 0이었다. AI-Hub actor population 표현을
selected Pilot 192명과 candidate-pool split population 202명으로 구분하는 required documentation
fix 1건을 반영해 completion requirements를 충족했다. 따라서 현재 상태는 **Phase 10
COMPLETE / 1st Pilot COMPLETE**다. 이는 Baseline이 정의한 1차 Pilot 구현·평가·검증 절차의
완료이며 production, clinical performance 또는 광범위한 일반화 성능 달성을 뜻하지 않는다.

# 2. Pilot Objective and Scope

## 2.1 문제 정의

짧은 RGB 영상 클립을 다음 3개 class 중 하나로 분류한다.

| Index | Class | Target 의미 |
|---:|---|---|
| 0 | 복약 | 약 먹기 |
| 1 | 음수 | 물 마시기 |
| 2 | 기타 | 위 두 행동이 아닌 행동 |

AI-Hub의 `Drink_bever`와 `Drink_alcohol`은 Stage A의 음수 auxiliary visual class다. 최종 ETRI 음수
target은 A004 물 마시기이므로 두 의미를 동일한 최종 task label로 일반화하지 않는다.

## 2.2 범위와 제약

- 개발 환경: WSL 2, Python 3.12, PyTorch/torchvision, MediaPipe
- 기준 GPU: NVIDIA GeForce GTX 1650 Ti, 4GB VRAM
- Pilot 목적: 최고 성능이 아니라 전체 pipeline과 비교 설계의 실행 가능성 검증
- 제외: 3D CNN, Video Transformer, encoder ETRI fine-tuning, 대규모 hyperparameter search
- primary selection metric: participant-disjoint 5-fold mean Macro-F1

Baseline의 단계적 설계는 실제 구현에서도 유지됐다. 다만 Baseline 작성 시점의 계획과 달리 최종
`D=1024`, GRU hidden size 128, Phase 10 raw-video OOF artifact 구조 등은 실제 config와 checkpoint에서
구체화됐다.

# 3. Final Data Usage

## 3.1 Stage A — AI-Hub

AI-Hub inventory master는 18,420개 JSON/video candidate와 55,260개 JPG다. 1개 candidate에
`duplicate_frame_reference` warning이 있었고 18,419개가 usable이었다. Stage A에는 valid
`viewpoint_3` 후보에서 actor-disjoint로 선정한 400 videos, 1,200 frames가 사용됐다.

| Split | 복약 | 음수 | 기타 | Videos | Frames |
|---|---:|---:|---:|---:|---:|
| Train | 80 | 80 | 160 | 320 | 960 |
| Validation | 20 | 20 | 40 | 80 | 240 |
| 합계 | 100 | 100 | 200 | 400 | 1,200 |

- Actors(400개 selected Pilot video 기준): 192명, train 152 / validation 40, 교집합 없음
- 참고: actor-disjoint split은 전체 candidate pool 202명(train-split 162 / validation-split 40)을 기준으로 사전 확정됨
- Class mapping: `Take_pills → 복약`, `Drink_bever/Drink_alcohol → 음수 auxiliary`, 나머지 → 기타
- 기타에는 `Eat_food` hard negative 50 videos 포함
- 각 video의 JPG 3장은 같은 split에 유지

Stage A best checkpoint는 epoch 12의 validation video Macro-F1 `0.652049`로 선택됐다. 이 값은
AI-Hub auxiliary Stage A 평가이며 최종 ETRI 물 마시기 성능이 아니다.

## 3.2 Stage B — ETRI

ETRI Batch B inventory에는 RGB 6,589 clips와 30 participants가 있다. 1개 small RGB clip이 invalid로
격리되어 6,588개가 usable이었다. 전체 available Batch B를 학습한 것이 아니라 participant별 cap을
적용한 Fixed Pilot subset만 사용했다.

| Fold | Participants | 복약 | 음수 | 기타 | 합계 |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 12 | 12 | 24 | 48 |
| 1 | 6 | 12 | 12 | 24 | 48 |
| 2 | 6 | 12 | 12 | 24 | 48 |
| 3 | 6 | 12 | 12 | 24 | 48 |
| 4 | 6 | 11 | 12 | 24 | 47 |
| 합계 | 30 | 59 | 60 | 120 | 239 |

선정 cap은 participant당 복약 최대 2, 음수 최대 2, 기타 4이며 기타는 hard negative 2 + general
other 2를 목표로 했다. 실제 기타 120개는 hard negative 60, general other 60이다. P227만 유효한
복약 선정 clip이 1개여서 전체 복약 수가 59다. Fold는 Pilot selection 전에 participant 단위로
고정됐으며 한 participant의 모든 take는 동일 fold에 속한다.

Phase 7에서는 각 fold의 4개 fold를 train, 해당 1개 fold를 validation/OOF로 사용했다. Phase 9에서는
selected-valid 239개 전체로 Stage B만 새로 초기화해 학습했다.

## 3.3 Excluded / Reserved Data

- ETRI Batch A: 사용하지 않음
- ETRI Batch B의 미선정 candidate: manifest에 유지되지만 Stage B Pilot 학습/평가에는 사용하지 않음
- A045~A048 multi-person action: Pilot에서 제외
- Self-recorded: 학습, model selection, threshold 또는 preprocessing tuning에 사용하지 않음
- Self-recorded 3개: Phase 9 model의 functional/qualitative pipeline check에만 사용

# 4. Final Architecture and Training Flow

```text
Raw RGB Video
→ frame count validation
→ fixed_uniform T=64 sampling
→ MediaPipe face / hands / pose ROI
→ success / partial / full-frame fallback
→ BGR-to-RGB
→ Resize 224×224
→ float tensor + ImageNet normalization
→ frozen MobileNetV3-Small Encoder B
→ [T, 1024] frame embeddings
→ GRU(hidden=128, final hidden)
→ Linear(128, 3)
→ logits / softmax
→ 복약 / 음수 / 기타
```

Stage A Encoder B는 ImageNet pretrained MobileNetV3-Small에서 시작해 AI-Hub actor-disjoint Pilot로
학습했다. `features.11`, `features.12`, embedding head와 classifier를 학습했고 frozen BatchNorm
running-stat policy를 적용했다. Stage B에서는 classifier logits가 아니라 classifier 이전의
1024-dimensional visual representation을 사용하며 encoder parameter는 모두 frozen이다.

GRU는 frame별 visual representation의 순서 정보를 요약한다. 설정은 `batch_first=true`, 1 layer,
unidirectional, dropout 0, final hidden representation이다.

Tensor contract:

```text
Encoder input  : [B*T, 3, 224, 224]
Encoder output : [B*T, 1024]
Stage B input  : [B, 64, 1024]
Stage B output : [B, 3]
```

Normalization은 mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`다. ROI config SHA-256은
`7b692e3fa0260f745664869a58a958cf276d428cf693abda4449fcbadc71e381`, Stage A preprocessing config
SHA-256은 `0987566c7175a408f4d72504e77c16ec835eba722a52a98a03fe62e204e4e54f`다.

# 5. Model Selection — Phase 7 / Phase 8

Phase 7은 동일 Fixed Pilot Manifest, fold, sampling, ROI, normalization, T, D, loss 및 seed policy 아래
2×2 ablation을 수행했다.

| Experiment | Encoder | Temporal model | Macro-F1 mean ± std | 복약 Recall | 음수 Recall | 기타 Recall |
|---|---|---|---:|---:|---:|---:|
| A | ImageNet-only | Mean Pooling | 0.450650 ± 0.079355 | 0.339394 | 0.266667 | 0.783333 |
| B | AI-Hub fine-tuned | Mean Pooling | 0.449102 ± 0.087356 | 0.221212 | 0.300000 | 0.866667 |
| C | ImageNet-only | GRU | 0.518992 ± 0.042044 | 0.374242 | 0.300000 | 0.883333 |
| D | AI-Hub fine-tuned | GRU | **0.532176 ± 0.054575** | 0.357576 | 0.333333 | 0.900000 |

Phase 8 selection artifact는 primary metric `macro_f1_5fold_mean`이 가장 높은 Experiment D를 rank 1로
선택했다. numerical tie가 없어 secondary tie-break는 사용하지 않았다. D는 C보다 `+0.013184`,
B보다 `+0.083074` 높았다.

이 CV는 architecture/model selection의 근거이면서 Pilot 성능 추정 역할도 한다. 따라서
participant-disjoint OOF evidence이지만, model selection에 사용되지 않은 untouched final test와는
동일하지 않다.

# 6. Phase 9 Single Deployment Model

Phase 9에서는 선택된 D 구조의 Stage B GRU를 새로 초기화해 ETRI selected-valid 239개 전체에서
15 epochs 학습했다. Encoder B와 기존 Phase 6 embeddings는 고정됐고 Phase 7 fold checkpoint를
재사용하지 않았다.

| 항목 | 값 |
|---|---|
| 역할 | `deployment_check` |
| Training scope | all valid Pilot samples, 239 clips |
| Encoder | frozen AI-Hub fine-tuned Encoder B |
| Stage B | newly initialized GRU |
| Checkpoint | `<work_root>/checkpoints/phase9_deployment/phase9_deployment_full_pilot/deployment_check.pt` |
| SHA-256 | `8065c16515e27dd45f5621fbe350f9ebc8b90b19969bc41776cab49a0c303736` |
| MLflow run | `fe14b293c8dc4794936a326b4e1f5659`, FINISHED |

Phase 7 fold models는 held-out fold OOF evaluation용이다. 반면 Phase 9 single model은 fold model 중
하나를 임의로 배포하는 대신 모든 Pilot data를 활용해 inference와 pipeline check에 사용할 하나의
checkpoint를 제공한다. Phase 9의 `training_diagnostic_macro_f1`은 training population 자체의 진단값이며
independent performance로 보고하지 않는다.

# 7. Raw-video OOF End-to-End Evaluation Protocol

평가 population은 ETRI Fixed Pilot selected-valid 239개다. 각 row는 manifest `fold`와 같은 Phase 7
Experiment D checkpoint로 평가됐다. 예를 들어 `fold=2` sample은 D fold 2 model만 사용하며, 이 model의
training participants는 fold 2 participants를 포함하지 않는다.

- Total expected/evaluated: 239/239
- Inference failures: 0
- Duplicate/missing: 0/0
- Participant leakage: 0
- 각 sample 평가 횟수: 정확히 1회
- Input: 실제 ETRI raw MP4
- Embedding cache를 inference input으로 사용: 아니오
- Phase 9 deployment model 사용: 아니오
- Device: CPU

실제 path는 `video_frame_count → uniform_frame_indices → decode_sampled_frames →
preprocess_shared_frames/extract_roi → build_transform(training=False) → extract_frozen_embeddings →
fold-specific GRU → softmax`다. 이는 Phase 6 shared preprocessing을 재사용하며 별도 inference
preprocessing을 만들지 않는다.

Single-video shape는 `[64,3,224,224] → [64,1024] → [1,64,1024] → [1,3]`로 모든 sample에서
검증됐다.

# 8. Quantitative Results

## 8.1 Aggregate OOF

| Metric | Value |
|---|---:|
| OOF Macro-F1 | 0.538337 |
| Accuracy | 0.623431 |
| Macro Precision | 0.604227 |
| Macro Recall | 0.529755 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 복약 | 0.567568 | 0.355932 | 0.437500 | 59 |
| 음수 | 0.606061 | 0.333333 | 0.430108 | 60 |
| 기타 | 0.639053 | 0.900000 | 0.747405 | 120 |

## 8.2 Fold별 결과

| Fold | Macro-F1 | 복약 Recall | 음수 Recall | 기타 Recall | N |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.483333 | 0.166667 | 0.416667 | 0.875000 | 48 |
| 1 | 0.480423 | 0.166667 | 0.333333 | 0.958333 | 48 |
| 2 | 0.511081 | 0.416667 | 0.250000 | 0.875000 | 48 |
| 3 | 0.624478 | 0.583333 | 0.333333 | 0.916667 | 48 |
| 4 | 0.561563 | 0.454545 | 0.333333 | 0.875000 | 47 |

- 5-fold Macro-F1 mean: `0.532176`
- 5-fold Macro-F1 population std: `0.054575`
- 복약 Recall mean ± std: `0.357576 ± 0.165381`
- 음수 Recall mean ± std: `0.333333 ± 0.052705`
- 기타 Recall mean ± std: `0.900000 ± 0.033333`

Aggregate OOF Macro-F1은 239개 prediction을 한 번에 합쳐 계산한다. 5-fold mean ± std는 각 fold에서
별도로 계산한 Macro-F1 다섯 개의 평균과 population standard deviation(`ddof=0`)이다. Fold 크기와
class support가 완전히 같지 않으므로 두 값은 같을 필요가 없다.

# 9. Confusion Matrix and Error Analysis

GT × Prediction, class order는 복약 / 음수 / 기타다.

| GT \ Prediction | 복약 | 음수 | 기타 | Support |
|---|---:|---:|---:|---:|
| 복약 | 21 | 8 | 30 | 59 |
| 음수 | 9 | 20 | 31 | 60 |
| 기타 | 7 | 5 | 108 | 120 |

## Verified

- 복약 59개 중 30개, 음수 60개 중 31개가 기타로 분류됐다.
- 기타 recall은 0.90으로 높지만 복약과 음수 recall은 각각 0.356, 0.333이었다.
- 복약↔음수 직접 혼동은 복약→음수 8개, 음수→복약 9개였다.
- 오류의 중심은 두 target class가 기타로 흡수되는 패턴이다.

## Interpretation

Pilot model은 기타 class를 상대적으로 강하게 구분하지만 target action sensitivity가 낮다. 특히
음수 precision 0.606에 비해 recall 0.333인 결과는 음수로 예측할 때보다 실제 음수를 놓치는 문제가
더 크다는 뜻이다.

## Hypothesis

Target 행동의 짧은 핵심 순간, 작은 물체, participant/domain variation, class imbalance 또는 frozen
visual representation이 이 패턴에 기여했을 가능성이 있다. 이 원인들은 이번 Pilot에서 개별적으로
검증되지 않았으므로 사실로 단정하지 않는다.

# 10. Raw vs Cached OOF Consistency

| 항목 | 결과 |
|---|---|
| Raw / cached sample count | 239 / 239 |
| Sample set exact match | Yes |
| Prediction agreement | 239/239, 100% |
| Raw / cached aggregate Macro-F1 | 0.538337 / 0.538337 |
| Macro-F1 difference | 0.0 |
| Per-class metric difference | 모두 0.0 |
| Fold Macro-F1 difference | 모든 fold 0.0 |
| Confusion matrix equal | Yes |
| Probability max absolute difference | `1.2516975e-6` |
| Probability mean absolute difference | `2.7000561e-7` |
| Stored artifact verdict | `DIFFERENCE_OBSERVED` |

Stored verdict는 구현 시 고정된 probability tolerance `1e-6`를 최대 차이가 약 `0.252e-6` 초과했기
때문이다. 결과 확인 후 tolerance나 artifact verdict를 변경하지 않았다. Prediction과 metric 관점에서는
완전히 일치하며, probability에는 CPU 재계산의 미세한 수치 차이가 있다. 이 비교는 Phase 6 cache
생성 path와 Phase 10 raw-video path 사이에 prediction 또는 metric을 바꾸는 실질적 drift가 없음을
검증했다.

# 11. ROI Analysis

## 11.1 ETRI raw-video OOF

| 범위 | Total frames | Success | Partial | Fallback | Fallback ratio |
|---|---:|---:|---:|---:|---:|
| 전체 | 15,296 | 0 | 15,178 | 118 | 0.7714% |
| 복약 | 3,776 | 0 | 3,745 | 31 | 0.8210% |
| 음수 | 3,840 | 0 | 3,838 | 2 | 0.0521% |
| 기타 | 7,680 | 0 | 7,595 | 85 | 1.1068% |

ETRI inference는 거의 전부 partial ROI path에 의존했다. Success 0은 inference failure가 아니라
face/hands/pose 조합 중 일부 landmark만으로 contextual crop을 만든 status 의미다. Fallback frame도
full-frame crop으로 처리됐으며 어떤 sample도 제외되지 않았다. Phase 6 cache summary와 Phase 10 raw
ROI 합계는 정확히 일치한다.

## 11.2 Self-recorded

Self-recorded 192 sampled frames에서 success 7, partial 176, fallback 9였다. Medication 영상의
fallback 9 frames를 포함해 모든 영상이 최종 prediction까지 도달했다. 작은 표본이므로 ETRI와 ROI
비율을 정량 비교하지 않는다.

# 12. ETRI Functional Inference Check

Raw-video OOF 전에 Phase 9 single deployment/check model로 대표 ETRI 4개 영상을 CPU에서 실행했다.
이 JSON은 `/tmp` 임시 artifact이며 공식 OOF artifact가 아니다.

| Video | Intended/GT | Prediction | Confidence | ROI S/P/F | Status |
|---|---|---|---:|---:|---|
| `A003_P201_G006_H070.mp4` | 복약 | 복약 | 0.6716 | 0/64/0 | PASS |
| `A004_P201_G006_H070.mp4` | 음수 | 음수 | 0.4864 | 0/64/0 | PASS |
| `A020_P201_G010_H120.mp4` | 기타 general | 기타 | 0.9399 | 0/64/0 | PASS |
| `A010_P201_G007_H120.mp4` | 기타 hard negative | 기타 | 0.9604 | 0/64/0 | PASS |

네 prediction의 일치는 functional sample check일 뿐 소규모 accuracy나 model performance evidence로
사용하지 않는다.

# 13. Self-recorded Functional / Qualitative Check

Phase 9 deployment/check model로 `<work_root>/self_recorded/pipeline_check/`의 3개 영상을 CUDA에서
실행했다.

| Video | Intended | Prediction | Confidence | P(복약) | P(음수) | P(기타) | ROI S/P/F | Device | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `medication_01.mp4` | 복약 | 기타 | 0.6392 | 0.2095 | 0.1512 | 0.6392 | 1/54/9 | CUDA | PASS |
| `drinking_01.mp4` | 음수 | 기타 | 0.8251 | 0.0995 | 0.0754 | 0.8251 | 6/58/0 | CUDA | PASS |
| `other_01.mp4` | 기타 | 기타 | 0.6825 | 0.1633 | 0.1542 | 0.6825 | 0/64/0 | CUDA | PASS |

`PASS`는 video decode, sampling, ROI, Encoder B, GRU, softmax 및 result 생성이 crash 없이 완료됐다는
functional 의미다. 복약과 음수의 classification은 intended class와 일치하지 않았고 기타만 일치했다.
3개 sample로 Accuracy, Macro-F1, Precision 또는 Recall을 계산하지 않는다. 이 결과를 근거로 모델,
threshold 또는 preprocessing을 변경하지 않았다.

# 14. Runtime / Device Verification

| 검증 | Device | 결과 |
|---|---|---|
| Stage A production training | CUDA | 완료 |
| Phase 7 A/B/C/D production runs | GPU 실행 기록 존재 | 20/20 FINISHED |
| Phase 9 deployment training | GPU 실행 기록 존재 | FINISHED |
| Phase 10 representative ETRI inference | CPU | PASS |
| Phase 10 239 raw-video OOF | CPU | 239/239 PASS |
| Self-recorded 3-class pipeline check | CUDA | 3/3 functional PASS |

다음은 측정하지 않았다.

- production latency와 throughput
- systematic CPU/GPU probability reproducibility
- on-device latency
- peak memory benchmark
- 장시간 service 안정성

# 15. Final Pilot Findings

## 15.1 Verified

- AI-Hub actor-disjoint Stage A와 ETRI participant-disjoint Stage B data contract가 동작한다.
- Frozen MobileNetV3-Small embedding `[B,1024]`과 GRU `[B,64,1024] → [B,3]` contract가 동작한다.
- 2×2 ablation에서 GRU 구성 C/D가 Mean 구성 A/B보다 높은 5-fold mean Macro-F1을 기록했다.
- 고정된 selection rule은 Experiment D를 선택했다.
- 5개 D fold model로 raw MP4 239개 OOF를 leakage 없이 완주했다.
- Cache와 raw-video path는 239/239 prediction과 metric이 일치했다.
- Phase 9 single model은 ETRI와 self-recorded video에서 최종 label/confidence를 생성했다.
- Partial/fallback ROI는 sample을 삭제하지 않고 pipeline을 유지했다.

## 15.2 Interpretation

- Temporal GRU가 이 Pilot에서는 mean pooling보다 유리했다.
- 기타 class는 상대적으로 강하지만 복약/음수 sensitivity는 부족하다.
- Shared preprocessing 재사용은 cache/inference drift를 prediction 수준에서 통제했다.
- Self-recorded domain에서 pipeline은 작동했지만 두 target sample을 기타로 분류해 domain shift 또는
  target sensitivity 문제 가능성을 보여준다.

## 15.3 Not Verified

- Untouched independent final-test performance
- Phase 9 deployment model의 independent accuracy
- Batch A 또는 external dataset 일반화
- 일반 인구, 임상 또는 production 성능
- Threshold calibration과 event detection
- Latency, throughput, memory 및 on-device suitability
- Self-recorded 정량 성능

# 16. Limitations

1. **Untouched final test 부재**: participant-disjoint CV는 leakage를 막지만 model selection에도 사용됐다.
2. **선택과 추정의 결합**: Phase 7 CV가 구조 선택과 Pilot 성능 추정 양쪽 역할을 한다.
3. **Phase 9 independent test 부재**: 239개 전체로 학습한 single model의 training diagnostic은 공식
   performance가 아니다.
4. **Pilot subset 규모**: available Batch B 6,588 usable clips 중 239개만 사용했다.
5. **Class imbalance**: 기타 120 대 복약 59, 음수 60이다.
6. **Target recall**: 복약 0.356, 음수 0.333으로 기타 0.90보다 낮다.
7. **ETRI encoder adaptation 부재**: Encoder B는 AI-Hub에서 fine-tune됐고 ETRI에서는 frozen이다.
8. **Auxiliary label 차이**: AI-Hub 음수 auxiliary는 술/음료를 포함하지만 최종 target은 물 마시기다.
9. **ROI 의존**: ETRI sampled frames의 99.23%가 partial, 0.77%가 fallback이었다.
10. **Self-recorded 규모와 결과**: 3개뿐이며 복약/음수 intended sample을 기타로 예측했다.
11. **Device 범위**: raw OOF는 CPU, self-recorded는 CUDA였으나 체계적 device parity benchmark가 없다.
12. **External/cross-batch 미검증**: Batch A, external camera, 다양한 환경·인구 일반화를 평가하지 않았다.

# 17. Interpretation Boundaries

사용 가능한 표현:

- participant-disjoint 5-fold CV
- ETRI Batch B Pilot raw-video OOF End-to-End evaluation
- Pilot performance estimate
- ETRI/self-recorded functional pipeline check
- qualitative self-recorded check

사용하면 안 되는 표현:

- untouched final-test performance
- Phase 9 deployment model independent-test performance
- production 또는 clinical performance
- general population accuracy
- broad real-world generalization
- self-recorded quantitative performance

# 18. Pilot Completion Status

Independent Phase 10 Final Audit을 완료했다. 초기 verdict는 **PASS WITH REQUIRED FIXES**였으며,
BLOCKER 0, required documentation fix 1건은 AI-Hub actor population 표현을 selected Pilot
192명과 candidate-pool split population 202명으로 구분하는 내용이었다. 해당 fix를 반영해
completion requirements를 충족했으며, 최종 상태는 **Phase 10 COMPLETE / 1st Pilot
COMPLETE**다.

현재 알려진 최신 regression은 113 tests PASS, failed 0, skipped 0이다. Pilot COMPLETE는
Baseline에서 정의한 1차 Pilot 구현·평가·검증 절차를 완료했다는 뜻이며, untouched
independent final test, Phase 9 deployment model independent-test performance, production/clinical 성능
또는 광범위한 일반화 성능을 주장하지 않는다.

# 19. Artifact References

| Artifact logical path | 역할 | SHA-256 / ID | Count |
|---|---|---|---:|
| `<work_root>/manifests/pilot/ai_hub_pilot_manifest.jsonl` | Stage A fixed manifest | `9dbe8e3100c47a6e81be999309a11e202b6c6a7e5c5f20c924816a4e448e6e9e` | 18,420 rows / 400 selected |
| `<work_root>/manifests/pilot/etri_pilot_manifest.jsonl` | ETRI fixed manifest | `0c641e3301196afa92c4cf7b7cad28dfd9e21c5f88a5ebbe02b694110b3b4b93` | 6,589 rows / 239 selected |
| `<work_root>/checkpoints/stage_a/stage-a-20260829T170113531051Z-adf86795/best.pt` | Encoder B | `26393618a91db07826b6e9b1ba0beb30696ae92b9fd8c929d01054aed866cfd6` | epoch 12 |
| `<work_root>/embeddings/etri_pilot_t64_ab/manifest-0c641e33/summary.json` | Phase 6 cache summary | `2f7a1fc1d5b3bbe0333fb48e743f0fd502ce47b41f9820be0f2d145f77987614` | 239 |
| `<work_root>/checkpoints/phase7_ablation/phase7_D_fold0/best.pt` | D fold 0 | `b86cdfcc0858c7e1b8812f87ea29c1f6ed00066ddcad3db7b5f0267905dd0105` | 48 OOF |
| `<work_root>/checkpoints/phase7_ablation/phase7_D_fold1/best.pt` | D fold 1 | `3b4adae05f5b3f06583307637776e1184305e477d9bbfe3be76b399a29a68a13` | 48 OOF |
| `<work_root>/checkpoints/phase7_ablation/phase7_D_fold2/best.pt` | D fold 2 | `f8010653eb8b825f8ac7c729c1a15c0f89593944ff7529aa421e459923e1de02` | 48 OOF |
| `<work_root>/checkpoints/phase7_ablation/phase7_D_fold3/best.pt` | D fold 3 | `dd1a3a53241268d968af562fa2473958db84b227f185e0e6125e9278c4d7e93d` | 48 OOF |
| `<work_root>/checkpoints/phase7_ablation/phase7_D_fold4/best.pt` | D fold 4 | `9d1ff9af1fcdd5349abae2960196cf9b670e7545525f7a004d39a73c07c879cf` | 47 OOF |
| `<work_root>/checkpoints/phase7_ablation/phase7_D_aggregate/oof_predictions.csv` | Cached D OOF | `0d2e087dc6f1396616acb6dbd45ac10bfb9bda9bf8e5d40ee93e40661870d6b9` | 239 |
| `configs/phase8_selected_model.yaml` | Selection handoff | `8e9966a1cbacf235b4e63a7d7b21a69bb901cc9e2ecb3ee0a5f49a6867a13638` | D selected |
| `<work_root>/checkpoints/phase9_deployment/phase9_deployment_full_pilot/deployment_check.pt` | Single deployment/check model | `8065c16515e27dd45f5621fbe350f9ebc8b90b19969bc41776cab49a0c303736` | 239 training |
| `<work_root>/evaluations/phase10_raw_video_oof/raw_video_oof_predictions.csv` | Raw-video OOF predictions | `a0cc471d340310f4b50bf8c03c6998992a68cbb71954dee28e27b0183be97300` | 239 |
| `<work_root>/evaluations/phase10_raw_video_oof/raw_video_oof_metrics.json` | Raw OOF metrics/provenance | `09b7947323be9863ee50035b7281afc5d84102e96abde63fa18247db88ebd3bf` | 239 |
| `<work_root>/evaluations/phase10_raw_video_oof/raw_video_oof_confusion_matrix.csv` | Aggregate confusion matrix | `aaf700c79ff0701edc1eb06caba84a5fe77772a2ce9416c7fea1dfe2cb56cb1a` | 3×3 |
| `<work_root>/evaluations/phase10_raw_video_oof/fold_metrics.json` | Fold별 metrics | `da6766661f247a2424f179af5000264be46815cd6f13fe264d3a37374675f5ab` | 5 folds |
| `<work_root>/evaluations/phase10_raw_video_oof/raw_vs_cached_oof_consistency.json` | Cache/raw consistency | `f055316c2a77703f99d2fd3771307e585b433e1d179bc911f124285e91148115` | 239 |
| `<work_root>/evaluations/phase10_self_recorded/medication_01.json` | Self-recorded medication check | `0c4d48c914990f004635ca9bef60eb1376dc446fc9ba6f69a666ff51026fd56a` | 1 |
| `<work_root>/evaluations/phase10_self_recorded/drinking_01.json` | Self-recorded drinking check | `c68a27f27fd6aeb9b0ea57748bc3b81335a23a43428953091ad635bf34987134` | 1 |
| `<work_root>/evaluations/phase10_self_recorded/other_01.json` | Self-recorded other check | `c1846c62b6125ae7cfe4a5ffb5284bbde1bd2cdeed712d75e3908152edcd1152` | 1 |

## Source-of-Truth 주의사항

- `configs/phase8_selected_model.yaml`의 `phase9.deployment_checkpoint_created: false`는 Phase 9 실행 전
  handoff 상태다. 현재 존재하는 Phase 9 checkpoint provenance, MLflow FINISHED run과 STATUS를 우선해
  Phase 9 checkpoint가 생성된 것으로 판단했다.
- `STATUS.md`는 Independent Phase 10 Final Audit과 required documentation fix 반영 후 Phase 10 COMPLETE /
  1st Pilot COMPLETE 상태로 동기화되어 있다.
- Self-recorded artifact는 이전 raw OOF 작업 종료 후 workspace에 추가됐다. 이 문서는 현재 존재하는
  JSON 3개를 직접 읽어 기록했다.
