# Pilot Retrospective & Roadmap

**복약 / 음수 / 기타 3-class 행동 인식 모델 — 1차 Pilot 회고 및 Post-Pilot 발전·고도화 방향**

- 프로젝트: `medication-drinking-action-recognition`
- 작성 기준일: 2026-08-31
- 문서 상태: **FINAL — 1st Pilot Retrospective & Post-Pilot Roadmap**
- 선행 상태: **Phase 10 COMPLETE / 1st Pilot COMPLETE**
- 다음 단계: **Full Experiment / Post-Pilot Planning — NOT STARTED**
- 상위 설계 기준: `docs/00_Pilot_Design_Baseline.md`
- 주요 근거:
  - `docs/05_Phase8_Structure_Selection_Result.md`
  - `docs/06_Pilot_Final_Evaluation.md`
  - `docs/07_Pilot_Development_Record.md`
  - `STATUS.md`
  - AI-Hub / ETRI 데이터 구조 및 EDA 문서

---

# 0. Document Purpose and Interpretation Rules

## 0.1 문서 목적

이 문서는 1차 Pilot의 개발 과정을 다시 기록하기 위한 문서가 아니다.

Phase 0~10에서 실제 수행한 작업, 생성된 artifact, Git·MLflow provenance와 구현 lifecycle은 `07_Pilot_Development_Record.md`에 기록되어 있으며, 공식 정량 평가와 confusion matrix, 오류 분석, limitation은 `06_Pilot_Final_Evaluation.md`에 기록되어 있다.

본 문서의 목적은 해당 evidence를 바탕으로 다음 질문에 답하는 것이다.

> **1차 Pilot을 통해 무엇이 검증되었고 무엇이 아직 검증되지 않았으며, 복약·음수 행동 인식 성능과 일반화 가능성을 높이기 위해 다음 실험에서는 무엇을 어떤 순서로 검증해야 하는가?**

따라서 본 문서는 다음 역할을 가진다.

1. 1차 Pilot의 기술적 회고
2. Pilot 결과에서 도출된 핵심 병목과 교훈 정리
3. 후속 개선 가설의 명시
4. Full Experiment의 우선순위와 검증 순서 정의
5. 향후 실제 제품형 행동 인식 모델로 확장하기 위한 중장기 방향 보존

본 문서 자체는 새로운 학습·평가 결과를 생성하지 않는다.

---

## 0.2 다른 문서와의 역할 구분

```text
00_Pilot_Design_Baseline.md
→ 1차 Pilot을 어떤 기준으로 설계할 것인가

05_Phase8_Structure_Selection_Result.md
→ 어떤 모델 구조가 Pilot에서 선택되었는가

06_Pilot_Final_Evaluation.md
→ 최종적으로 어떤 성능과 오류가 관측되었는가

07_Pilot_Development_Record.md
→ Pilot을 실제로 어떻게 구현·검증했는가

08_Pilot_Retrospective_and_Roadmap.md
→ 그래서 무엇을 배웠으며 다음에는 무엇을 검증할 것인가
```

본 문서는 기존 freeze 문서의 역사적 사실이나 공식 Pilot 결과를 소급 수정하지 않는다.

Post-Pilot 단계에서 새로운 설계가 확정될 경우 기존 Pilot Baseline을 덮어쓰지 않고 별도의 Full Experiment 설계 문서 또는 신규 Phase 문서에서 관리한다.

---

## 0.3 Interpretation Rules

본 문서에서는 내용을 다음 세 종류로 구분한다.

### Verified Evidence

실제 Pilot artifact, metric, config, checkpoint, manifest, MLflow, Git 또는 final documentation에서 직접 확인된 사실이다.

예:

```text
Experiment D 5-fold Macro-F1 mean = 0.532176
복약 aggregate OOF Recall = 0.355932
음수 aggregate OOF Recall = 0.333333
ETRI ROI partial = 15,178 / 15,296 sampled frames
```

### Interpretation

Verified Evidence를 바탕으로 한 기술적 해석이다.

예:

```text
현재 Pilot 조건에서는
temporal modeling의 효과가
AI-Hub fine-tuning의 추가 효과보다 크게 관측됐다.
```

Interpretation은 해당 Pilot 조건에서의 해석이며 일반적 사실이나 인과관계가 확정됐다는 의미가 아니다.

### Roadmap Hypothesis

다음 실험에서 검증해야 하는 가설이다.

예:

```text
ETRI domain에서 visual encoder 일부를 fine-tune하면
target representation이 개선될 가능성이 있다.
```

가설은 실제 비교 실험 전까지 성능 개선 사실로 표현하지 않는다.

---

## 0.4 결과 해석 경계

`1st Pilot COMPLETE`는 다음 전체 절차를 실제로 완주했다는 뜻이다.

```text
데이터 준비
→ manifest
→ ROI
→ Stage A
→ embedding
→ Stage B
→ 2×2 ablation
→ model selection
→ raw-video inference
→ participant-disjoint OOF evaluation
→ final audit
```

다음을 의미하지 않는다.

- production-ready model
- clinical validation 완료
- 실제 사용자 환경 일반화 확보
- 독립 final test 성능 확보
- 장시간 continuous stream에서의 event recognition 검증
- 최종 복약 순응도 판정 시스템 완성

이 해석 경계를 Post-Pilot에서도 유지한다.

---

# 1. Executive Retrospective

## 1.1 Overall Verdict

1차 Pilot은 **최종 성능 자체보다 전체 개발·실험·평가 구조를 검증하는 데 성공했다.**

제한된 데이터와 GTX 1650 Ti 4GB 환경에서도 다음 전체 흐름을 실제로 구축했다.

```text
Raw Data
→ Full Inventory
→ Leakage-safe Manifest
→ ROI Preflight
→ Stage A Visual Encoder
→ ETRI Embedding Cache
→ Mean / GRU 2×2 Ablation
→ Participant-disjoint 5-fold CV
→ Model Structure Selection
→ Deployment/Check Model
→ Raw-video OOF Evaluation
→ Self-recorded Pipeline Check
```

최종 선택 구조는:

```text
Raw RGB Video
→ Fixed-uniform Sampling (T=64)
→ MediaPipe Contextual ROI / Fallback
→ AI-Hub Fine-tuned MobileNetV3-Small Encoder B
→ [T, 1024] Embeddings
→ GRU (hidden=128)
→ Linear
→ 복약 / 음수 / 기타
```

였다.

그러나 target action을 실제 활용 수준으로 안정적으로 탐지하기에는 성능이 아직 충분하지 않았다.

Aggregate raw-video OOF:

| Class | Recall |
|---|---:|
| 복약 | 0.355932 |
| 음수 | 0.333333 |
| 기타 | 0.900000 |

즉 현재 가장 중요한 문제는 **기타는 상대적으로 잘 구분하지만, 실제 관심 대상인 복약과 음수를 놓치는 현상​**이다.

---

## 1.2 Pilot에서 가장 중요한 성과

### Engineering

- Raw Data immutable lifecycle 확립
- deterministic manifest 및 artifact provenance 확보
- actor / participant leakage 없는 평가 구현
- cache 기반 빠른 실험과 raw-video evaluation 연결
- Git / MLflow / config / checkpoint 추적 구조 구축
- unit → smoke → full execution → independent audit 절차 확립

### Modeling

- 경량 MobileNetV3-Small + GRU 구조가 실제 환경에서 실행 가능함을 확인
- Mean Pooling보다 GRU가 두 encoder 조건 모두에서 우수
- frozen embedding 기반 Stage B ablation이 효과적인 실험 전략임을 확인

### Evaluation

- raw-video OOF 239 / 239 inference 성공
- participant leakage = 0
- duplicate / missing / failure = 0
- cached OOF와 raw-video OOF prediction 및 metric 동일

---

## 1.3 Pilot에서 가장 중요한 미해결 문제

최종 OOF confusion matrix:

```text
GT \ Pred    복약   음수   기타
복약          21      8     30
음수           9     20     31
기타           7      5    108
```

가장 큰 오류 축은:

```text
복약 ↔ 음수
```

보다:

```text
복약 → 기타
음수 → 기타
```

다.

따라서 다음 단계의 중심 문제를 다음처럼 정의한다.

> **복약과 음수를 서로 구분하는 것에 앞서, 두 target action을 기타 행동으로부터 안정적으로 분리하는 능력을 높인다.**

---

## 1.4 Post-Pilot 기본 전략

Post-Pilot에서는 처음부터 더 큰 backbone이나 무거운 video architecture로 이동하지 않는다.

우선순위는 다음과 같다.

```text
평가 기준과 test policy 고정
↓
AI-Hub / ETRI 보유 데이터 활용 범위 확대
↓
target / hard-negative / other 구성 개선
↓
sampling / loss / augmentation 개선
↓
ROI / input representation 진단
↓
ETRI domain encoder adaptation
↓
temporal sampling / temporal architecture 확장
↓
독립 / cross-domain generalization 평가
↓
deployment optimization
```

현재 성능 병목은 모델 용량 부족 하나로 설명할 수 없으므로 데이터, input information, visual representation, temporal modeling을 분리 검증한다.

---

# 2. Pilot Objectives vs. Actual Outcomes

| Pilot Question | 검증 목표 | 실제 결과 | Verdict |
|---|---|---|---|
| 전체 pipeline을 구현할 수 있는가? | Raw → 학습 → 평가 → inference | Phase 0~10 완료 | **VALIDATED** |
| participant leakage 없이 평가 가능한가? | participant-disjoint CV | leakage 0 | **VALIDATED** |
| lightweight 2-stage 구조가 실행 가능한가? | CNN + temporal model | 4GB 환경에서 완료 | **VALIDATED** |
| temporal modeling이 도움이 되는가? | Mean vs GRU | C>A, D>B | **SUPPORTED** |
| AI-Hub fine-tuning이 ETRI에 도움이 되는가? | ImageNet vs AI-Hub FT | 제한적·구조 의존적 | **PARTIALLY SUPPORTED** |
| cache 결과가 raw-video에서 재현되는가? | raw/cache consistency | 239 prediction 동일 | **VALIDATED** |
| 복약·음수를 충분히 탐지하는가? | target sensitivity | Recall 약 0.36 / 0.33 | **NOT VALIDATED** |
| 실제 사용환경에서도 일반화하는가? | real-world generalization | 정량 평가 없음 | **NOT VALIDATED** |
| model selection과 독립된 final 성능이 있는가? | untouched final test | 없음 | **NOT VALIDATED** |
| cross-batch / external domain에 일반화하는가? | domain evaluation | 수행하지 않음 | **NOT VALIDATED** |

---

# 3. What the Pilot Successfully Validated

## 3.1 Data and Split Integrity

Pilot은:

```text
Raw
→ Full Candidate Inventory
→ split / fold
→ selection
→ Fixed Manifest
```

순서를 유지했다.

이를 통해:

- invalid sample provenance 유지
- deterministic selection
- 결과 이후 data composition 변경 방지
- actor / participant leakage 자동 검증
- Raw Data immutable 정책

을 실제로 유지할 수 있었다.

AI-Hub에서는 actor-disjoint split과 동일 video의 3 frame 동일 split을 유지했고, ETRI에서는 동일 participant의 모든 take가 같은 fold에 유지됐다.

이 정책은 Full Experiment에서도 변경하지 않는 핵심 계약으로 본다.

---

## 3.2 End-to-End Inference Feasibility

ETRI raw MP4 입력부터:

```text
decode
→ frame sampling
→ ROI
→ encoder
→ GRU
→ softmax
→ label
```

까지 239개 영상 전부 실행됐다.

```text
Expected      239
Evaluated     239
Failure         0
Duplicate       0
Missing         0
Leakage         0
```

따라서 전체 raw-video inference path의 구현 가능성은 검증됐다.

---

## 3.3 Two-stage Architecture Feasibility

다음 interface가 실제 학습과 inference에서 일관되게 유지됐다.

```text
Stage A
[B, 3, 224, 224]
→ MobileNetV3-Small
→ [B, 1024]

Stage B
[B, 64, 1024]
→ GRU
→ [B, 3]
```

frozen embedding cache를 통해 제한된 GPU에서도 Stage B를 반복 비교할 수 있었다.

후속 실험에서도:

```text
encoder / ROI 변경 없음
→ cache 기반 빠른 ablation

encoder / ROI 변경
→ embedding 재생성 또는 raw-video training
```

으로 구분하는 전략이 유효하다.

---

## 3.4 Temporal Modeling Value

2×2 결과:

| 비교 | Δ Macro-F1 |
|---|---:|
| C − A: ImageNet + GRU vs ImageNet + Mean | +0.068342 |
| D − B: AI-Hub FT + GRU vs AI-Hub FT + Mean | +0.083074 |

두 encoder 조건 모두에서 GRU가 Mean보다 우수했다.

### Interpretation

현재 Pilot에서는 frame representation의 단순 평균보다 **temporal order와 sequence context를 사용하는 것이 행동 분류에 더 유리했다.**

따라서 Post-Pilot의 temporal control baseline은 GRU로 유지한다.

이 결과가 더 복잡한 temporal model의 우위를 의미하지는 않는다.

---

## 3.5 Reproducibility and Provenance

Pilot에서 다음이 체계적으로 기록됐다.

- manifest hash
- fold
- config
- seed
- Git commit
- checkpoint
- encoder source
- sampling
- ROI
- normalization
- MLflow run
- metric artifact

raw-video OOF와 cached OOF는:

```text
sample set         동일
prediction         239 / 239 동일
metric             동일
confusion matrix   동일
```

이었다.

이는 후속 Full Experiment에서도 유지해야 하는 기준이다.

---

## 3.6 Development Process

Pilot에서 유효했던 절차:

```text
설계 고정
↓
unit test
↓
소수 sample smoke
↓
전체 실행
↓
artifact 검증
↓
independent audit
↓
required fix
↓
freeze
```

Phase 4에서 ROI 문제를 전체 embedding 생성 전 발견한 사례는 **대규모 실행 전에 작은 검증 gate를 두는 방식**의 가치를 보여준다.

---

# 4. Performance Gaps and What Remains Unvalidated

## 4.1 Low Target-action Sensitivity

Aggregate OOF:

| Class | Correct | → 다른 target | → 기타 |
|---|---:|---:|---:|
| 복약 | 21 / 59 | 8 | **30** |
| 음수 | 20 / 60 | 9 | **31** |

복약·음수의 약 절반이 기타로 흡수됐다.

### Interpretation

현재 decision boundary가 불확실한 target sample을 기타로 보내는 방향으로 형성됐을 가능성이 있다.

하지만 원인은 다음 중 하나 또는 조합일 수 있다.

- 데이터 규모
- other composition
- class balance
- loss
- ROI
- object information loss
- visual encoder domain gap
- temporal sampling
- model capacity

Pilot만으로 원인을 분리할 수 없으므로 후속 실험에서는 한 번에 하나의 주요 축만 변경한다.

---

## 4.2 Limited and Mixed AI-Hub Transfer Effect

AI-Hub fine-tuning 효과:

| 비교 | Δ Macro-F1 |
|---|---:|
| B − A: Mean | -0.001548 |
| D − C: GRU | +0.013184 |

복약 Recall도 D가 C보다 소폭 낮았다.

### Interpretation

AI-Hub fine-tuning은 완전히 무효라고 할 수 없지만, 현재 방식만으로 강한 domain transfer 효과가 확인됐다고 말할 수 없다.

가능한 원인 후보:

- AI-Hub ↔ ETRI domain 차이
- AI-Hub sparse 3-frame representation
- 음수 auxiliary label mismatch
- Stage A training 규모 제한
- ETRI encoder frozen 정책

은 모두 후속 실험 가설이다.

---

## 4.3 AI-Hub Temporal Information Limitation

현재 로컬 AI-Hub 데이터는 영상당:

```text
3 sampled JPG
+
JSON annotation
```

을 보유하며 original MP4는 로컬에 없다.

따라서 Stage A는 실제 temporal video pretraining이 아니라 visual frame representation 학습이다.

3개 frame은 같은 원본 video에서 나온 묶음이므로 split과 video-level evaluation에는 사용했지만, temporal sequence 전체를 학습할 수는 없다.

---

## 4.4 AI-Hub Supervision Underuse

현재 Pilot에서 AI-Hub의 모든 annotation 정보를 모델 supervision으로 활용하지 않았다.

추가 활용 가능한 자산:

- 3-frame group identity
- bbox
- `obj_name`
- action timeline metadata
- actor / demographic metadata
- camera / viewpoint information

특히 target bbox에는:

```text
복약
→ 알약
→ 약봉지

음수
→ 음료잔
→ 음료캔 / 병
```

과 같이 행동 판별과 직접적으로 관련된 object cue가 존재한다.

향후 해당 정보가 실제 target-vs-other 분리에 도움이 되는지 별도 실험이 필요하다.

---

## 4.5 ETRI Domain Adaptation Not Tested

ETRI Stage B에서는 visual encoder가 frozen이었다.

검증된 것은:

```text
ImageNet / AI-Hub representation
→ frozen
→ ETRI temporal classifier
```

이다.

검증되지 않은 것은:

```text
ETRI RGB
→ visual encoder partial fine-tuning
```

및:

```text
Encoder + GRU joint optimization
```

이다.

따라서 ETRI-specific visual adaptation은 높은 가치가 있는 후속 가설이다.

---

## 4.6 ROI Limitation

ETRI 15,296 sampled frames:

```text
success      0
partial      15,178
fallback       118
```

이었다.

`partial`은 failure가 아니며 contextual ROI는 정상 생성됐다.

하지만:

```text
ROI pipeline이 안정적임
≠
현재 ROI가 classification에 최적임
```

이다.

복약·음수에는:

```text
상체
손
얼굴 / 입
약 / 약봉지
컵 / 병
주변 context
```

가 동시에 중요할 수 있다.

현재 ROI가 작은 object나 global context를 제거하는지 직접 ablation이 필요하다.

---

## 4.7 Small Pilot Dataset

ETRI Pilot:

```text
239 clips
30 participants

복약 59
음수 60
기타 120
```

만 사용했다.

Batch B에는:

```text
6,589 clips
복약 A003 119
음수 A004 120
```

이 존재한다.

따라서 data scale-up 자체의 효과는 아직 검증되지 않았다.

---

## 4.8 Original Class Imbalance

ETRI 전체 데이터는 기타 비율이 압도적으로 높다.

Pilot에서는 약:

```text
복약 : 음수 : 기타
1 : 1 : 2
```

로 완화했다.

이는 Pilot comparison에는 적절했지만 실제 Full Experiment에서는:

```text
Target Recall
vs
Other False Positive
```

trade-off가 다시 중요해진다.

Training distribution과 evaluation distribution을 같은 개념으로 취급하지 않는다.

Training은 class-balanced 또는 hard-negative 중심 구성이 가능하지만, evaluation은 별도로 realistic distribution을 반영할 수 있다.

---

## 4.9 Untouched Independent Test Absence

현재 5-fold CV는:

```text
architecture selection
+
Pilot performance estimation
```

을 모두 수행했다.

따라서 leakage-safe OOF evidence이지만 selection과 독립된 untouched final test는 아니다.

Post-Pilot에서는 가능한 경우:

```text
Development CV
→ architecture / hyperparameter selection

Independent Test
→ 모든 선택 완료 후 평가
```

를 분리한다.

---

## 4.10 Generalization Not Validated

아직 평가하지 않은 영역:

- ETRI Batch A
- Batch B → Batch A
- 외부 dataset
- 실제 제품 촬영환경
- 새로운 camera position
- 새로운 사용자 집단
- continuous stream
- 실제 복약 event 단위

따라서 현재 Pilot 성능을 실제 제품 일반화 근거로 사용하지 않는다.

---

## 4.11 Self-recorded Check

Self-recorded 3개 영상은 pipeline을 모두 통과했으나 intended 복약·음수 sample도 기타로 예측됐다.

확인된 것은:

```text
pipeline integration
```

이며:

```text
real-world accuracy
```

가 아니다.

별도의 실제환경 데이터셋을 설계할 때까지 성능 근거로 사용하지 않는다.

---

# 5. Key Lessons Learned

## 5.1 Data Lessons

### Lesson 1 — Target 데이터뿐 아니라 Other의 구성이 핵심이다

기타는 단일 행동이 아니라 매우 다양한 행동 집합이다.

Full Experiment에서는 기타를 최소 다음처럼 구분해 관리한다.

```text
Hard Negative
+
General Other
+
Easy / Distant Other
```

random other만 늘리는 것이 항상 효과적인 것은 아니다.

---

### Lesson 2 — Participant diversity가 반복 sample 수보다 중요하다

일부 participant에는 반복 take가 많다.

동일 participant의 반복 데이터만 확대하면 sample 수는 증가하지만 participant diversity는 증가하지 않는다.

따라서 데이터 확장 시:

```text
participant diversity
+
within-participant repetitions
```

를 분리 기록한다.

---

### Lesson 3 — Training distribution과 evaluation distribution을 분리한다

Training에서는:

- undersampling
- balanced batch
- target-aware sampling
- hard-negative emphasis

를 사용할 수 있다.

Evaluation은 별도로:

```text
balanced diagnostic evaluation
+
more naturalistic distribution evaluation
```

로 나눌 수 있다.

---

### Lesson 4 — Dataset 간 label semantic 차이를 보존해야 한다

AI-Hub:

```text
Drink_bever
Drink_alcohol
```

은 최종 ETRI:

```text
A004 = 물 마시기
```

와 완전히 동일한 task label이 아니다.

AI-Hub 음수는 visual auxiliary supervision이다.

---

## 5.2 AI-Hub Stage A Lessons

현재 Stage A는 400 videos / 1,200 frames의 제한된 subset이었다.

따라서 AI-Hub transfer가 작았다는 결과는:

```text
AI-Hub 전체 데이터 활용 효과가 없다
```

는 뜻이 아니다.

확인된 것은 현재 Pilot Stage A 조건의 효과다.

향후에는 다음을 분리 검증할 가치가 있다.

- target 전체 확대
- broader other
- Eat_food 등 hard-negative 강화
- `Drink_bever only`
- `Drink_bever + Drink_alcohol`
- 3-frame group supervision
- bbox / object-aware supervision

---

## 5.3 ROI / Preprocessing Lessons

MediaPipe ROI는 pipeline 안정성에는 기여했다.

하지만 classification 관점에서는:

```text
detectable ROI
≠
best discriminative input
```

이다.

따라서 향후 ROI를 개선하기 전에 현재 ROI가 full frame보다 실제로 유리한지부터 검증한다.

---

## 5.4 Visual Representation Lessons

Experiment C와 D:

```text
C = 0.518992
D = 0.532176
```

의 차이는 크지 않았다.

따라서 현재 병목을 AI-Hub fine-tuning 하나로 설명할 수 없다.

향후 visual representation은:

```text
AI-Hub Stage A 강화
+
ETRI domain adaptation
+
input/ROI 개선
```

을 별도 축으로 비교한다.

---

## 5.5 Temporal Modeling Lessons

두 encoder 모두 GRU가 Mean보다 우수했다.

따라서 temporal order를 유지하는 구조는 Full Experiment에서도 control baseline으로 보존한다.

더 복잡한 temporal model은 이후 비교한다.

---

## 5.6 Object Information Lessons

복약과 음수는 작은 object cue가 중요한 행동이다.

현재 pipeline은 해당 object를 별도 detection task로 학습하지 않는다.

따라서 object-aware representation은 유망한 후보지만 반드시 baseline 대비 ablation으로 검증한다.

---

## 5.7 Unused ETRI Modality Lessons

현재 ETRI에서 모델 입력에 직접 사용하지 않은 정보:

- Skeleton
- Body Index

가 존재한다.

이들은 반드시 최종 제품 입력으로 사용해야 하는 modality가 아니다.

### Skeleton

water-drinking motion 보조 feature 또는 RGB motion representation의 진단용으로 사용할 수 있다.

### Body Index

실제 제품에서 사용하기보다 offline diagnostic / supervision 용도로 활용 가능하다.

예:

```text
Body Index person mask
vs
MediaPipe ROI
```

를 비교해 ROI 품질을 평가할 수 있다.

---

## 5.8 Experiment Design Lessons

한 실험에서 여러 축을 동시에 변경하지 않는다.

예:

```text
데이터 확대
+
ROI 변경
+
loss 변경
+
encoder fine-tuning
```

을 동시에 수행하면 개선 원인을 알 수 없다.

각 실험은 가능한 한 하나의 주요 hypothesis를 검증한다.

---

## 5.9 Evaluation Lessons

유지할 기준:

- participant-disjoint
- Macro-F1
- per-class Recall / Precision / F1
- confusion matrix
- fold variance
- aggregate OOF와 fold mean 구분

Post-Pilot에서는 추가로:

- independent test
- seed stability
- confidence / calibration
- target false-negative 분석
- same-domain vs cross-domain evaluation

을 검토한다.

---

## 5.10 Engineering Lessons

다음 운영 원칙은 유지한다.

```text
Raw immutable
Manifest frozen
Config-driven
Artifact hash
MLflow
Git provenance
Unit test
Smoke test
Independent audit
No silent overwrite
```

---

# 6. Post-Pilot Problem Definition

## 6.1 Primary Problem

> **복약과 음수 target action이 기타 class로 흡수되는 오류를 줄이고 target sensitivity를 개선한다.**

현재 가장 큰 문제는:

```text
복약 / 음수
→ 기타
```

이다.

---

## 6.2 Secondary Objectives

### Other false positive control

```text
Target Recall ↑
+
Other false positive 통제
```

를 함께 본다.

### Participant generalization

새로운 participant에 대한 성능 유지.

### Domain generalization

촬영환경·batch·camera 조건 변화에서 성능 유지.

### Lightweight deployment feasibility

모델 개선 과정에서도 경량 구조를 우선한다.

---

## 6.3 Post-Pilot Objective

> **현재 leakage-safe lightweight pipeline을 보존하면서 AI-Hub·ETRI 데이터를 더 충분히 활용하고, target-vs-other representation을 개선하여 복약·음수 Recall을 높인 뒤 독립 평가를 통해 participant 및 domain generalization을 검증한다.**

---

# 7. Improvement Hypotheses

본 장은 Verified Evidence가 아니라 후속 실험 가설이다.

---

## H1. ETRI Data Scale Hypothesis

### Hypothesis

Pilot보다 더 많은 ETRI target clip과 다양한 other를 사용하면 target representation과 participant variability를 더 잘 학습할 수 있다.

### Test

```text
Pilot-size dataset
vs
Expanded Batch B dataset
```

model / ROI / loss는 고정한다.

---

## H2. AI-Hub Stage A Scale Hypothesis

### Hypothesis

현재 400-video Stage A보다 더 많은 AI-Hub target / hard-negative / other를 사용하면 ETRI에 전달되는 visual representation이 개선될 가능성이 있다.

### Candidates

```text
A0 Current 400-video Stage A baseline
A1 Full/expanded target + matched other
A2 Expanded hard negatives
A3 Larger stratified viewpoint_3 subset
```

---

## H3. AI-Hub Drinking Label Hypothesis

### Hypothesis

AI-Hub 음수 auxiliary class 정의가 ETRI 물 마시기 transfer에 영향을 줄 수 있다.

### Compare

```text
Drink_bever only
vs
Drink_bever + Drink_alcohol
```

이를 통해 proxy label 범위가 transfer에 미치는 영향을 확인한다.

---

## H4. AI-Hub 3-frame Group Hypothesis

### Hypothesis

같은 original video에서 나온 3개 frame을 독립 frame으로만 학습하는 것보다 group identity를 활용하면 video-level visual representation consistency가 개선될 수 있다.

### Candidate

```text
Independent frame CE baseline
vs
3-frame grouped / consistency-aware learning
```

구체적 loss는 별도 설계 후 검증한다.

---

## H5. Object-aware AI-Hub Supervision Hypothesis

### Hypothesis

AI-Hub의 bbox와 object label을 auxiliary supervision으로 활용하면 복약·음수의 작은 target object representation이 개선될 수 있다.

후보:

- bbox crop representation
- object auxiliary head
- global + object crop feature
- object-aware sampling

---

## H6. Hard-negative Composition Hypothesis

### Hypothesis

random other보다 target과 시각적으로 유사한 hard negative를 체계적으로 노출하는 것이 target-vs-other boundary 개선에 더 효과적일 수 있다.

후보 action군:

- 음식 먹기
- 얼굴 주변 손 동작
- 양치
- 전화
- 컵/병을 다루지만 마시지 않는 행동
- 작은 물체를 입 주변으로 가져가는 행동

정확한 list는 inventory와 EDA를 기준으로 확정한다.

---

## H7. Epoch-wise Other Sampling Hypothesis

### Hypothesis

고정된 작은 other subset보다 epoch마다 넓은 other candidate에서 재샘플링하면 negative diversity를 더 많이 학습할 수 있다.

예:

```text
Target
→ 고정 또는 높은 sampling probability

Other
→ epoch-wise resampling

Other composition
→ hard negative x%
→ general other y%
```

---

## H8. Class-balance / Loss Hypothesis

후보:

```text
Standard CE
Class-weighted CE
Balanced Sampler
Target-aware Batch Sampler
Focal Loss
```

sampler와 loss를 처음부터 동시에 변경하지 않는다.

---

## H9. Visual Augmentation Hypothesis

### Hypothesis

적절한 visual augmentation은 participant / camera / lighting 변화에 대한 robustness를 높일 수 있다.

후보:

- mild brightness / contrast
- color jitter
- scale / crop
- small affine
- horizontal flip, 행동 의미가 유지되는 경우에 한함

작은 약/컵 object가 사라질 정도의 강한 crop은 피한다.

---

## H10. Temporal Augmentation Hypothesis

후보:

- temporal jitter
- random start
- sampled frame drop
- interval perturbation
- multi-window crop

fixed-uniform baseline 이후 비교한다.

---

## H11. ROI / Input Representation Hypothesis

### Hypothesis

현재 contextual ROI보다 더 넓은 context 또는 global+local representation이 target-vs-other 분리에 도움이 될 수 있다.

후보:

```text
R0 Current ROI
R1 Full Frame
R2 Broader Upper-body ROI
R3 Global + Local Crop
R4 Person + Hand/Object Region
```

우선 R0 / R1 / R2처럼 단순한 비교부터 수행한다.

---

## H12. ETRI Encoder Adaptation Hypothesis

후보:

```text
E0 Frozen Encoder B
E1 Head / embedding adaptation
E2 Last block fine-tuning
E3 Last N blocks
E4 Progressive unfreezing
E5 Low-LR joint Encoder + GRU
```

E5는 앞 단계에서 partial fine-tuning 효과가 확인된 이후 검토한다.

---

## H13. Temporal Sampling Hypothesis

후보:

```text
T=32
T=64 baseline
T=96
T=128
```

및:

```text
fixed uniform
random temporal crop
multi-window
```

를 비교한다.

---

## H14. Alternative Temporal Model Hypothesis

후보:

- GRU baseline
- LSTM
- TCN / 1D temporal convolution
- lightweight temporal attention
- TSM-style lightweight approach

데이터·visual representation 병목을 먼저 해결한 뒤 수행한다.

---

## H15. Alternative Task Formulation Hypothesis

현재:

```text
복약 / 음수 / 기타
→ direct 3-class
```

가 계속 target→other 문제를 보일 경우 후순위로 다음 formulation을 검토할 수 있다.

### Hierarchical

```text
Stage 1
Target vs Other

Stage 2
복약 vs 음수
```

### One-vs-Rest

```text
Medication vs Rest
Drinking vs Rest
```

이 구조가 실제로 더 나은지는 별도 검증이 필요하며 현재 우선순위는 낮다.

---

## H16. Calibration / Decision-layer Hypothesis

classifier 성능이 안정된 이후:

- class-specific threshold
- target-vs-other threshold
- probability calibration
- uncertainty / reject option

을 검토한다.

이는 feature/model learning과 별개인 **decision-layer optimization**으로 분리한다.

---

## H17. Multimodal Hypothesis

후순위 후보:

```text
RGB
+
Skeleton
```

또는:

```text
RGB
+
object feature
```

Skeleton은 복약 object cue를 대체하지 않으므로 RGB 보조 modality로 본다.

---

# 8. Full Experiment Roadmap

## 8.1 Track A — Evaluation and Dataset Foundation

대규모 학습 전 먼저 고정한다.

1. Full Experiment population
2. development participants
3. independent test participants 또는 test role
4. Batch A 역할
5. target / hard-negative / other taxonomy
6. class mapping
7. primary / secondary metrics
8. artifact naming
9. seed policy
10. no-tuning test policy

원칙:

```text
test / split policy
먼저 고정
↓
training data 구성
↓
model experiment
```

---

## 8.2 Track B — AI-Hub Stage A Expansion

목표:

```text
Current 400-video Stage A
→ stronger visual pretraining baseline
```

실험 후보:

1. target sample 확대
2. broader other pool
3. Eat_food 등 hard-negative 확대
4. `Drink_bever only`
5. `Drink_bever + Drink_alcohol`
6. 3-frame grouped training
7. bbox/object-aware auxiliary supervision

각 실험에서 ETRI transfer 성능을 별도로 확인한다.

AI-Hub Stage A metric 자체만으로 최종 개선을 판단하지 않는다.

---

## 8.3 Track C — ETRI Data Expansion

ETRI Batch B에서:

- target 사용 범위 확대
- participant coverage 유지
- hard-negative 확대
- general other 확대

를 수행한다.

첫 실험에서는 current Experiment D 구조를 최대한 유지해 **data scale effect 자체를 분리**한다.

---

## 8.4 Track D — Other Composition / Sampling / Loss

추천 순서:

```text
D0 Fixed current-style other subset
↓
D1 Expanded hard-negative composition
↓
D2 Epoch-wise other resampling
↓
D3 Class-weighted CE
↓
D4 Balanced / target-aware sampler
↓
D5 Focal Loss
```

한 실험에서 여러 방법을 동시에 추가하지 않는다.

---

## 8.5 Track E — Augmentation

데이터와 sampling baseline이 확정된 뒤:

### Visual

- mild photometric augmentation
- scale / crop
- small affine

### Temporal

- jitter
- random start
- frame interval perturbation

을 분리 비교한다.

---

## 8.6 Track F — ROI / Input Representation Diagnostic

Encoder adaptation 전에 먼저 확인하는 것을 권장한다.

첫 비교:

```text
Current ROI
vs
Full Frame
vs
Broader Context ROI
```

가능하면 동일 frozen encoder 조건에서 수행한다.

목적은:

> **현재 ROI가 필요한 정보를 제거하고 있는가?**

를 먼저 확인하는 것이다.

필요 시 Body Index를 offline reference로 사용해 person foreground / crop 품질을 진단할 수 있다.

---

## 8.7 Track G — Visual Encoder Adaptation

ROI/input baseline이 정리된 이후 수행한다.

### AI-Hub Encoder 비교

- current Encoder B
- expanded Stage A encoder
- object-aware Stage A encoder

### ETRI adaptation

```text
Frozen
vs
last block fine-tuning
vs
last N blocks
vs
progressive unfreezing
```

GTX 1650 Ti 환경에서는:

- AMP
- small batch
- gradient accumulation
- limited trainable block

을 우선 검토한다.

---

## 8.8 Track H — Temporal Sampling

visual input과 encoder가 고정된 상태에서:

```text
T=32 / 64 / 96 / 128
```

및 sampling policy를 비교한다.

---

## 8.9 Track I — Temporal Architecture

GRU를 control baseline으로 유지한다.

후보 temporal model은 동일 데이터 / encoder / ROI / sampling에서 비교한다.

성능 개선이 작다면 모델 단순성을 우선한다.

---

## 8.10 Track J — Alternative Formulation / Calibration

직접 3-class 모델이 개선 후에도 target→other 병목을 크게 유지할 경우에만 진행한다.

후보:

- hierarchical target-vs-other → target subtype
- one-vs-rest
- threshold / calibration

이는 초기 Full Experiment의 우선순위가 아니다.

---

## 8.11 Track K — Multimodal / Additional Supervision

후순위:

- RGB + Skeleton
- RGB + object feature
- Body Index offline supervision
- teacher/student representation

실제 제품에서 사용할 수 없는 modality는 최종 inference dependency로 만들지 않는 방향을 우선한다.

---

# 9. Evaluation and Generalization Roadmap

## 9.1 Same-domain Generalization

목적:

> 같은 ETRI Batch B domain에서 새로운 participant에게 일반화되는가?

가능한 경우 held-out participant test를 사용한다.

---

## 9.2 Development CV

모델 선택과 tuning에는 participant-disjoint CV를 유지한다.

동일 participant의 모든 take는 동일 group에 유지한다.

---

## 9.3 Independent Final Test

최종 후보 선택 후 test를 1회 평가한다.

test 결과를 보고:

- threshold
- preprocessing
- hyperparameter
- model

을 다시 변경하지 않는 것이 원칙이다.

---

## 9.4 Cross-domain Generalization

ETRI Batch A는 Batch B와 촬영 protocol이 다르므로:

```text
Train / Select on Batch B
→ Evaluate on Batch A
```

는 same-domain final test가 아니라 **cross-domain robustness test**로 해석한다.

---

## 9.5 Product-oriented External Dataset

실제 환경용 self-recorded dataset을 새로 만든다면 사전에 protocol을 정의한다.

포함:

- participant
- camera position
- lighting
- 복약 행동 정의
- 물 마시기 정의
- hard negative
- clip/event boundary
- train / validation / test separation
- 개인정보 및 데이터 관리

현재 3개 pipeline-check 영상은 이 데이터셋과 별도로 유지한다.

---

## 9.6 Evaluation Distribution

가능하면 두 종류를 분리한다.

### Balanced Diagnostic Set

target model capability 비교용.

### Naturalistic / Realistic Set

실제 기타 행동 비율이 높은 환경을 더 가깝게 반영.

둘을 혼합해서 하나의 metric으로 해석하지 않는다.

---

## 9.7 Error Analysis

기본 오류 축:

```text
복약 → 기타
음수 → 기타
기타 → 복약
기타 → 음수
복약 ↔ 음수
```

가능한 slice:

- participant
- action category
- camera height
- take
- clip duration
- ROI status
- confidence
- hard-negative type

---

## 9.8 Metrics

기본:

- Macro-F1
- Accuracy
- Macro Precision
- Macro Recall
- per-class Precision
- per-class Recall
- per-class F1
- confusion matrix
- fold mean ± std

추가:

- PR curve / PR-AUC
- confidence distribution
- calibration
- false-negative rate
- false-positive rate

---

## 9.9 Repeated-seed Stability

초기 screening은 기존처럼 고정 seed를 사용할 수 있다.

하지만 최종 후보 간 차이가 작다면:

```text
fixed-fold
+
multiple seeds / repeated runs
```

를 검토한다.

특히:

- mean performance
- std
- fold consistency
- seed consistency

를 함께 확인한다.

작은 single-run 차이를 구조적 개선으로 과해석하지 않는다.

---

# 10. Engineering / Deployment Roadmap

## 10.1 Inference Performance Benchmark

안정된 모델이 확보된 뒤:

- decode time
- preprocessing time
- encoder latency
- temporal model latency
- total latency
- peak VRAM / RAM
- CPU / GPU 비교

를 기록한다.

---

## 10.2 Model Export

필요 시:

```text
PyTorch
→ ONNX
→ target runtime
```

으로 확장한다.

---

## 10.3 Compression

후보:

- quantization
- pruning
- knowledge distillation
- smaller backbone

현재 target performance가 안정되기 전에는 우선하지 않는다.

---

## 10.4 Clip Classification → Event Recognition

현재 모델의 task:

```text
short clip
→ 복약 / 음수 / 기타
```

실제 제품에서는 장기적으로:

```text
continuous video
→ window / candidate segment
→ action classification
→ temporal aggregation
→ medication event
→ 복약 여부 / 기록
```

이 필요하다.

따라서 현재 3-class classifier는 최종 복약관리 시스템 전체가 아니라 **perception module의 핵심 baseline**이다.

continuous event recognition은 clip classifier의 성능과 일반화가 안정된 이후 별도 프로젝트 단계로 설계한다.

---

## 10.5 Decision Layer

향후 제품 요구사항이 구체화되면:

```text
복약 미탐 비용
vs
오탐 알림 비용
```

을 정의하고 threshold / reject / confidence policy를 설계한다.

---

# 11. Prioritized Execution Plan

## P0 — Pilot Baseline Preservation

보존:

- Fixed Pilot Manifest
- Phase 7 A/B/C/D artifacts
- Experiment D checkpoints
- Phase 8 selection artifact
- Phase 9 deployment/check model
- Phase 10 OOF artifacts
- Final Evaluation
- Development Record
- Git / MLflow provenance

새 실험은 기존 공식 artifact를 overwrite하지 않는다.

---

## P1 — Full Experiment Evaluation & Test Design

먼저 확정:

1. development population
2. test population
3. Batch A 역할
4. target taxonomy
5. hard-negative taxonomy
6. metrics
7. seed policy
8. artifact naming

이 단계가 끝나기 전에 대규모 tuning을 시작하지 않는다.

---

## P2 — Existing Data Utilization Expansion

### AI-Hub

- Stage A target 확대
- other / hard-negative 확대
- drinking label ablation
- 3-frame information
- bbox / object supervision

### ETRI

- Batch B target 확대
- broader other
- participant diversity 유지

현재 보유 데이터를 더 충분히 사용하는 것이 큰 architecture 변경보다 먼저다.

---

## P3 — Expanded-data D Baseline

현재 Experiment D 구조를 가능한 한 유지한 채 확대 데이터에서 baseline을 재구축한다.

목적:

> 데이터 규모 자체의 효과를 측정한다.

---

## P4 — Other Sampling / Hard Negative / Loss / Augmentation

순서 예:

```text
hard-negative composition
↓
epoch-wise other resampling
↓
class weighting / sampler
↓
focal loss
↓
augmentation
```

---

## P5 — ROI / Input Representation Diagnostic

```text
Current ROI
vs
Full Frame
vs
Broader Context
```

를 먼저 비교한다.

정보가 preprocessing 단계에서 사라지는지 확인한 뒤 encoder fine-tuning으로 이동한다.

---

## P6 — Visual Encoder Improvement

1. stronger AI-Hub Stage A encoder
2. ETRI partial fine-tuning
3. progressive unfreezing
4. 필요 시 low-LR joint training

순으로 검토한다.

---

## P7 — Temporal Sampling

- T
- temporal window
- jitter
- multi-window

비교.

---

## P8 — Lightweight Temporal Alternatives

GRU 이후에만:

- LSTM
- TCN
- temporal attention
- TSM-style approach

을 비교한다.

---

## P9 — Alternative Formulation / Decision Layer

target→other 병목이 충분히 해결되지 않는 경우:

- hierarchical classification
- one-vs-rest
- calibration
- threshold

을 검토한다.

---

## P10 — Independent / Cross-domain Validation

최종 후보에 대해:

```text
Same-domain participant test
+
Batch A cross-domain
+
별도 real-world dataset
```

순으로 확장한다.

---

## P11 — Deployment Optimization

성능 후보가 확정된 뒤:

- latency
- memory
- ONNX
- quantization
- streaming

을 진행한다.

---

# 12. Exit Criteria and Decision Gates

## 12.1 Data Foundation Gate

PASS 조건:

- participant leakage = 0
- target / hard-negative / general-other provenance 명확
- manifest deterministic
- train / validation / test 역할 명확
- invalid 처리 명확
- dataset statistics artifact 존재

---

## 12.2 Baseline Reproduction Gate

PASS:

- training 완료
- prediction completeness
- checkpoint reload
- metric recomputation
- artifact provenance
- inference 정상

---

## 12.3 Data Expansion Gate

다음을 함께 판단한다.

- Macro-F1
- 복약 Recall
- 음수 Recall
- 기타 false positive
- fold variance

accuracy만으로 판단하지 않는다.

---

## 12.4 Stage A Improvement Gate

새 AI-Hub Stage A encoder를 채택하려면:

```text
AI-Hub 자체 validation 향상
```

뿐 아니라:

```text
ETRI downstream improvement
```

근거가 있어야 한다.

Stage A 자체 성능만 높고 ETRI transfer가 개선되지 않으면 후속 baseline으로 채택하지 않을 수 있다.

---

## 12.5 ROI Gate

확인:

- target Recall
- other false positive
- Macro-F1
- fallback
- inference cost

ROI detection 성공률과 classification metric을 분리한다.

---

## 12.6 Encoder Adaptation Gate

Frozen 대비:

- mean performance
- target Recall
- fold stability
- seed stability
- compute cost

를 비교한다.

소폭 성능 차이만으로 더 복잡한 training을 채택하지 않는다.

---

## 12.7 Temporal Model Gate

GRU보다 복잡한 구조는 동일 조건에서 실제 개선이 있을 때만 채택한다.

성능이 유사하면:

1. target Recall
2. variance
3. model complexity
4. latency
5. memory

순으로 판단한다.

---

## 12.8 Generalization Gate

최종 후보는 가능한 경우 다음 중 둘 이상을 포함해 검증한다.

- same-domain held-out participant
- cross-batch
- external / product-oriented dataset

평가 데이터로 tuning하지 않는다.

---

## 12.9 Numerical Success Targets

현재 문서에서는 임의의 절대값:

```text
Macro-F1 ≥ 0.8
```

같은 기준을 확정하지 않는다.

우선 요구:

```text
baseline 대비 재현 가능한 개선
+
복약 Recall 개선
+
음수 Recall 개선
+
기타 false positive 관리
+
participant leakage 0
```

제품 요구사항이 확정되면 별도의 operational KPI를 설정한다.

---

# 13. Final Takeaways

1차 Pilot의 가장 큰 성과는 높은 최종 성능이 아니라 **복약·음수 행동 인식 문제를 실제 데이터와 제한된 하드웨어에서 처음부터 끝까지 검증 가능한 형태로 구현했다는 점**이다.

핵심 결론은 다음과 같다.

1. 전체 raw-video pipeline과 leakage-safe evaluation은 성공적으로 구축됐다.
2. temporal modeling은 현재 task에서 의미 있는 효과를 보였다.
3. AI-Hub fine-tuning의 추가 효과는 제한적이었지만 현재 Stage A 규모와 supervision만으로 전체 transfer 가능성을 판단할 수는 없다.
4. 가장 큰 성능 병목은 복약·음수가 기타로 흡수되는 target sensitivity 문제다.
5. 다음 단계에서는 기존 데이터의 활용 범위를 먼저 확대하고 hard-negative / sampling / loss를 개선해야 한다.
6. ROI에서 정보가 손실되는지 확인한 뒤 ETRI-specific encoder adaptation을 수행하는 것이 합리적이다.
7. 큰 video architecture보다 현재 lightweight baseline에서 각 병목을 분리해 검증하는 것이 우선이다.
8. independent participant test와 cross-domain evaluation을 분리해 일반화 성능을 검증해야 한다.
9. Skeleton / Body Index / object annotation은 반드시 final inference modality로 사용할 필요는 없지만 auxiliary supervision과 offline diagnostic에 활용할 수 있다.
10. 현재 clip classifier는 장기적으로 continuous video에서 복약 event를 판정하는 제품 pipeline의 perception baseline으로 활용할 수 있다.

Post-Pilot의 기본 전략:

```text
Pilot baseline 보존
↓
Evaluation / Test 설계
↓
AI-Hub + ETRI 데이터 활용 확대
↓
Expanded-data D baseline
↓
Hard Negative / Sampling / Loss / Augmentation
↓
ROI / Input Representation 진단
↓
Visual Encoder Adaptation
↓
Temporal Sampling
↓
Lightweight Temporal Model
↓
Independent / Cross-domain Validation
↓
Deployment Optimization
↓
Continuous Event Recognition
```

1차 Pilot은 종료됐지만 현재 구조는 폐기 대상이 아니다.

향후 Full Experiment의 목적은 Pilot을 새로 만드는 것이 아니라, **현재 재현 가능한 Experiment D baseline을 control로 유지하면서 어떤 변화가 실제로 복약·음수 인식 성능 개선을 만드는지 하나씩 검증하는 것**이다.

---

# Appendix A. Evidence → Interpretation → Action Matrix

| Verified Evidence | Interpretation | Recommended Action |
|---|---|---|
| D Macro-F1 `0.532176` > B `0.449102` | GRU temporal modeling 효과 큼 | GRU baseline 유지 |
| C − A `+0.068342` | ImageNet에서도 temporal 효과 | temporal order 유지 |
| D − B `+0.083074` | AI-Hub FT에서도 temporal 효과 | Mean 후순위 |
| D − C `+0.013184` | current AI-Hub transfer 이득 제한적 | Stage A 확대 + ETRI adaptation |
| B − A `-0.001548` | transfer 효과가 구조 의존적 | Encoder B 절대 우위로 해석 금지 |
| 복약 Recall `0.355932` | target sensitivity 부족 | target/object/encoder 개선 |
| 음수 Recall `0.333333` | target sensitivity 부족 | temporal/hard-negative 개선 |
| 기타 Recall `0.900000` | target이 기타로 흡수 | target-vs-other 중심 분석 |
| 복약 30/59 → 기타 | medication detection 병목 | medication-specific cue 강화 |
| 음수 31/60 → 기타 | drinking detection 병목 | motion/context 강화 |
| AI-Hub Stage A 400 videos | transfer scale 미검증 | AI-Hub 확대 |
| AI-Hub original MP4 미보유 | true video pretraining 불가 | 조건부 MP4 확보 검토 |
| AI-Hub bbox/object 미활용 | object supervision 여지 | bbox/object ablation |
| ETRI 239 subset | data scale 미검증 | Batch B 확대 |
| encoder frozen | ETRI adaptation 미검증 | partial fine-tuning |
| ROI partial 15,178/15,296 | current ROI 의존 | full-frame/context ablation |
| Skeleton 미사용 | motion auxiliary 미검증 | 후순위 multimodal |
| Body Index 미사용 | ROI reference 가능 | offline diagnostic |
| raw/cache 239 prediction 동일 | pipeline consistency 확보 | cache 적극 재활용 |
| participant leakage 0 | split policy 유효 | Full Experiment 유지 |
| untouched final test 없음 | 최종 일반화 불확실 | independent test |
| Batch A 미사용 | domain shift 불확실 | cross-batch test |
| self-recorded pipeline 성공 | integration 가능 | functional test 유지 |
| self target도 기타 예측 | real-world 성능 불확실 | 별도 external dataset |

---

# Appendix B. Underused / Unused Data Assets

| Asset | 현재 Pilot 활용 | Post-Pilot 후보 활용 |
|---|---|---|
| AI-Hub 3-frame group | 동일 split / video eval | group consistency / grouped training |
| AI-Hub bbox | ROI 참고 수준 | object auxiliary supervision |
| AI-Hub obj_name | 직접 모델 입력 아님 | medication/drink object representation |
| AI-Hub timeline metadata | 직접 temporal 학습 불가 | original MP4 확보 시 segment sampling |
| AI-Hub full target candidate | 일부만 사용 | Stage A scale-up |
| ETRI unused Batch B clips | 미선정 | Full Experiment 확대 |
| ETRI Batch A | 미사용 | cross-domain evaluation |
| ETRI Skeleton | 미사용 | RGB auxiliary motion branch |
| ETRI Body Index | 미사용 | ROI / person-mask offline reference |
| A045~A048 | Pilot 제외 | multi-person association 이후 재검토 |

---

# Appendix C. Conditional Medium-term Research

## Original AI-Hub Video Acquisition

현재 로컬에는 original MP4가 없다.

향후 확보 가능하면:

```text
timeline.start / end
↓
action segment extraction
↓
video sampling
↓
temporal pretraining
```

을 검토할 수 있다.

이는 storage / preprocessing / GPU 비용이 크므로 즉시 우선순위는 아니다.

---

## Multi-person Other Reintroduction

Pilot에서 제외한 A045~A048은 향후:

```text
person association
+
main participant selection
```

이 안정적으로 가능해진 이후 기타 행동 다양성 확대 관점에서 재도입할 수 있다.

---

## Heavy Video Models

후순위 연구 후보:

- 3D CNN
- Video Transformer
- SlowFast
- larger backbone

현재 lightweight 접근에서 데이터와 representation 병목을 충분히 검증한 이후 비교한다.

---

# Appendix D. Frozen Pilot Control

Post-Pilot 실험의 기준 control은 **Experiment D**다.

세부 architecture, checkpoint, config와 official Pilot metric은 다음 문서를 기준으로 한다.

- `docs/05_Phase8_Structure_Selection_Result.md`
- `docs/06_Pilot_Final_Evaluation.md`
- `docs/07_Pilot_Development_Record.md`
- `configs/phase8_selected_model.yaml`

핵심 reference metric:

```text
Phase 7 5-fold Macro-F1
0.532176 ± 0.054575

Phase 10 aggregate raw-video OOF Macro-F1
0.538337

Aggregate Recall
복약 0.355932
음수 0.333333
기타 0.900000
```

이 baseline과 기존 공식 artifact는 후속 실험 결과로 소급 수정하지 않는다.