# Phase 8 — Model Structure Selection 결과

- 실행일: 2026-08-30
- 판정: **PASS — Phase 8 COMPLETE**
- 선택 artifact: `configs/phase8_selected_model.yaml`
- 입력: Phase 7 A/B/C/D participant-disjoint 5-fold production artifacts

## 검증 범위

Phase 8 CLI가 Phase 7 summary 값을 복사하지 않고 fold metric과 OOF prediction에서 metric을
독립 재계산했다. A/B/C/D는 각각 intended fold 0~4, OOF 239개 exact-once이며 missing/duplicate가
없다. Manifest selected-valid key set, OOF key set, participant/fold/label, fold별 train/validation
participant disjointness를 교차 검증했다. 네 실험의 manifest/fold/class mapping/T/D/ROI/sampling/
normalization/loss/training/seed/frozen-encoder provenance는 동일하며 의도된 encoder와 Stage B 축만
다르다. 재계산 결과는 Phase 7 fold 및 aggregate artifact와 tolerance `1e-12` 안에서 일치했다.

## 독립 재집계와 ranking

표준편차는 Phase 7과 동일하게 population standard deviation(`ddof=0`)이다.

| Rank | Exp | Encoder | Stage B | Macro-F1 mean ± std | 복약 Recall mean ± std | 음수 Recall mean ± std | 기타 Recall mean ± std |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | D | AI-Hub fine-tuned | GRU | 0.532176 ± 0.054575 | 0.357576 ± 0.165381 | 0.333333 ± 0.052705 | 0.900000 ± 0.033333 |
| 2 | C | ImageNet-only | GRU | 0.518992 ± 0.042044 | 0.374242 ± 0.073699 | 0.300000 ± 0.040825 | 0.883333 ± 0.040825 |
| 3 | A | ImageNet-only | Mean | 0.450650 ± 0.079355 | 0.339394 ± 0.092088 | 0.266667 ± 0.152753 | 0.783333 ± 0.173606 |
| 4 | B | AI-Hub fine-tuned | Mean | 0.449102 ± 0.087356 | 0.221212 ± 0.114732 | 0.300000 ± 0.154560 | 0.866667 ± 0.066667 |

## Confusion matrices

행은 true class, 열은 predicted class이며 순서는 `복약 / 음수 / 기타`다.

| Exp | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| A | `[[6,2,4],[4,3,5],[5,8,11]]` | `[[4,3,5],[4,5,3],[5,1,18]]` | `[[3,0,9],[1,0,11],[2,0,22]]` | `[[3,6,3],[3,3,6],[2,0,22]]` | `[[4,2,5],[4,5,3],[2,1,21]]` |
| B | `[[0,0,12],[0,4,8],[2,4,18]]` | `[[4,4,4],[2,2,8],[1,3,20]]` | `[[3,0,9],[3,1,8],[2,0,22]]` | `[[3,5,4],[1,6,5],[1,1,22]]` | `[[3,3,5],[1,5,6],[2,0,22]]` |
| C | `[[4,2,6],[3,4,5],[2,0,22]]` | `[[3,3,6],[4,3,5],[2,2,20]]` | `[[5,1,6],[4,3,5],[2,0,22]]` | `[[5,3,4],[3,4,5],[1,1,22]]` | `[[5,2,4],[6,4,2],[2,2,20]]` |
| D | `[[2,1,9],[1,5,6],[1,2,21]]` | `[[2,1,9],[1,4,7],[0,1,23]]` | `[[5,2,5],[4,3,5],[3,0,21]]` | `[[7,2,3],[1,4,7],[1,1,22]]` | `[[5,2,4],[2,4,6],[2,1,21]]` |

OOF aggregate confusion matrix:

- A: `[[20,13,26],[16,16,28],[16,10,94]]`
- B: `[[13,12,34],[7,18,35],[8,8,104]]`
- C: `[[22,11,26],[20,18,22],[9,5,106]]`
- D: `[[21,8,30],[9,20,31],[7,5,108]]`

## 2×2 효과 분석

| 비교 | 의미 | Δ Macro-F1 | Δ 복약 Recall | Δ 음수 Recall | Δ 기타 Recall |
|---|---|---:|---:|---:|---:|
| B − A | Mean에서 AI-Hub transfer | -0.001548 | -0.118182 | +0.033333 | +0.083333 |
| D − C | GRU에서 AI-Hub transfer | +0.013184 | -0.016667 | +0.033333 | +0.016667 |
| C − A | ImageNet에서 GRU | +0.068342 | +0.034848 | +0.033333 | +0.100000 |
| D − B | AI-Hub에서 GRU | +0.083074 | +0.136364 | +0.033333 | +0.033333 |
| D − A | 전체 two-stage 비교 | +0.081525 | +0.018182 | +0.066667 | +0.116667 |

## 선택 결정과 Phase 9 handoff

Primary policy인 5-fold mean Macro-F1 내림차순에 따라 **Experiment D**를 선택한다. D의 mean
Macro-F1 `0.5321757751479114`가 가장 높고 exact tie가 아니므로 secondary tie-break는 사용하지
않았다. 선택 대상은 CV fold checkpoint가 아니라 다음 구조/configuration이다.

- Encoder: frozen AI-Hub fine-tuned MobileNetV3-Small, Phase 5 best checkpoint
- Stage B: GRU + Linear (`input_size=1024`, `hidden_size=128`, one layer, unidirectional,
  dropout 0, final hidden representation)
- Input: T=64, D=1024, fixed-uniform sampling
- Manifest SHA-256: `0c641e3301196afa92c4cf7b7cad28dfd9e21c5f88a5ebbe02b694110b3b4b93`
- Fold definition SHA-256: `6d960834e34dc4b8f510a92d65a9d5c26b1fc13ff3f565dc97162ddf5b7fb99a`
- Phase 9 scope: ETRI Pilot selected-valid 239개 전체에서 Stage B를 새로 초기화해 재학습
- CV fold checkpoint 재사용 금지; encoder는 frozen 유지

## Limitation

Pilot은 239개 clip/30명 규모이며 class imbalance가 있다. D의 복약 Recall fold 표준편차
`0.165381`은 비교적 크고, 복약 Recall mean은 C보다 `0.016667` 낮다. 이 trade-off는 보존하지만
사전에 고정된 primary metric을 뒤집지 않는다. Phase 8에서는 학습, tuning, checkpoint 생성 또는
Phase 9 구현을 수행하지 않았다.
