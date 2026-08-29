# 모델 구현 Reference 가이드 — 1차 Pilot

**복약 / 음수 / 기타 3-class 행동 인식 모델 — 모델·학습·평가·추론 코드 참고 기준**

- 작성일: 2026-08-29
- 문서 상태: **Model Implementation Reference — 1차 Pilot 구현 보조 기준**
- 프로젝트명: `medication-drinking-action-recognition`
- 상위 설계 문서: `docs/00_Pilot_Design_Baseline.md`
- 적용 범위: **Phase 5 Stage A ~ Phase 10 Pipeline Integration**
- 목적: 공식 라이브러리와 사전 리서치한 외부 구현을 **어디까지, 어떤 우선순위로 참고할지** 정의
- 주의: 이 문서는 Design Baseline의 모델·데이터·평가 정책을 변경하지 않는다.

---

# 0. 문서 역할

## 0.1 목적

이 문서는 1차 Pilot에서 모델·학습·평가·inference 코드를 작성할 때 다음을 명확히 하기 위한 구현 참고 가이드다.

1. 어떤 자료를 **Primary implementation source**로 사용할지
2. 어떤 GitHub repository를 **구조·패턴 참고용**으로 사용할지
3. 어떤 연구 구현을 **방법론·설계 근거용**으로만 사용할지
4. 외부 구현에서 **채택 가능한 범위와 채택하지 않을 범위**
5. Coding Agent가 모델 코드를 작성하기 전에 확인할 순서
6. 외부 코드 직접 재사용과 단순 참고를 구분하는 기록 원칙

이 문서의 핵심 목적은 **외부 코드를 그대로 복사하는 것**이 아니다.

현재 프로젝트의 Design Baseline을 기준으로:

```text
공식 API / 검증된 구현 패턴 확인
→ 현재 프로젝트 인터페이스와 제약에 맞게 재구성
→ 테스트와 provenance 기록
```

의 방식으로 구현하기 위한 기준이다.

---

## 0.2 Design Baseline과의 관계

모델·학습·평가·추론의 상위 정책은 항상:

```text
docs/00_Pilot_Design_Baseline.md
```

을 따른다.

본 문서는 다음을 변경할 권한이 없다.

- Class taxonomy
- AI-Hub / ETRI 데이터 역할
- Stage A / Stage B 구조
- MobileNetV3-Small 선택
- ImageNet-only vs AI-Hub fine-tuned 비교
- ETRI에서 encoder frozen 정책
- Mean Pooling vs GRU 비교
- Fixed Pilot Manifest
- participant-disjoint 5-fold CV
- Primary metric = 5-fold mean Macro-F1
- self_recorded 사용 제한
- 1차 Pilot 제외 설계

외부 Reference와 Design Baseline이 충돌하면:

```text
Design Baseline 우선
```

이다.

---

# 1. Reference 우선순위

모델·학습·평가·inference 코드를 구현할 때 참고 우선순위는 다음과 같다.

```text
1. 현재 사용자 요청
        ↓
2. docs/00_Pilot_Design_Baseline.md
        ↓
3. AGENTS.md
        ↓
4. 현재 프로젝트의 기존 config / interface / tests
        ↓
5. PyTorch / torchvision 공식 문서·공식 구현
        ↓
6. 본 문서에서 승인한 외부 GitHub Reference
        ↓
7. 일반적인 구현 관례
```

### 핵심 원칙

- 공식 PyTorch / torchvision API를 우선한다.
- 외부 GitHub repository는 **본 문서에 명시된 범위에서만** 참고한다.
- 외부 repository의 모델 구조를 이유로 Pilot 아키텍처를 임의 변경하지 않는다.
- 외부 repository의 dataset split, label taxonomy, transform, loss, threshold를 현재 프로젝트에 자동 적용하지 않는다.
- 외부 구현의 성능 수치를 현재 Pilot의 기대 성능으로 해석하지 않는다.
- 이미 프로젝트 내부에 동일 기능이 구현되어 있으면 중복 구현보다 기존 interface와의 일관성을 우선한다.

---

# 2. Reference 분류 체계

본 프로젝트에서는 외부 Reference의 사용 수준을 세 단계로 구분한다.

## 2.1 ADOPT

공식 라이브러리의 검증된 API를 현재 프로젝트에서 직접 사용한다.

예:

```text
torchvision.models.mobilenet_v3_small
torchvision.models.MobileNet_V3_Small_Weights
torch.nn.GRU
torch.nn.Linear
torch.nn.CrossEntropyLoss
torch.no_grad
state_dict
```

ADOPT는 **API를 사용하는 것**을 의미하며, 외부 프로젝트의 전체 코드를 복사한다는 의미가 아니다.

---

## 2.2 ADAPT

외부 repository의 코드 구조·함수 분리·orchestration·방법론을 참고하되 현재 프로젝트 요구사항에 맞게 새로 구현한다.

예:

```text
video load
→ preprocessing
→ model load
→ inference
→ result formatting
```

같은 책임 분리 방식.

---

## 2.3 DO NOT ADOPT

현재 Pilot과 충돌하거나 범위를 불필요하게 넓히는 구조는 채택하지 않는다.

예:

- 3D CNN
- MC3 / R3D 기반 end-to-end video model
- SlowFast
- optical-flow Two-Stream
- TensorFlow training pipeline 직접 이식
- 외부 dataset split 정책
- 외부 class taxonomy
- 외부 threshold 기반 event detector
- 별도 근거 없는 architecture 변경

---

# 3. Reference Matrix

| Reference | 분류 | 현재 Pilot 역할 | 참고/채택 범위 | 채택하지 않는 범위 |
|---|---|---|---|---|
| **PyTorch 공식 문서/API** | ADOPT | Stage B 및 공통 training/inference 구현 기준 | `nn.GRU`, `nn.Linear`, CE, freeze, `eval`, `no_grad`, `state_dict` | custom GRU 재구현, 공식 API와 불필요하게 중복되는 구현 |
| **torchvision 공식 문서/API** | ADOPT | Stage A visual encoder 구현 기준 | MobileNetV3-Small, ImageNet weights, preprocessing metadata, classifier/feature interface 확인 | 외부 backbone으로 임의 변경 |
| **dronefreak/human-action-classification** | ADAPT | action-recognition software structure 참고 | training / inference 모듈 분리, video inference orchestration, device/model loading 패턴, CLI/API 책임 분리 | MC3-18, R3D-18, 3D CNN pipeline, UCF/HMDB transform/split, repo taxonomy |
| **prouast/deep-intake-detection** | ADAPT — methodology only | 섭취 행동 인식 방법론·temporal modeling 근거 | appearance feature의 중요성, 2D CNN + temporal 모델 비교 관점, warm-start 개념, 평가 설계 참고 | TensorFlow 코드 직접 이식, TFRecord, optical flow, SlowFast, Two-Stream, event threshold detector 직접 적용 |

---

# 4. Primary Reference — PyTorch / torchvision

## 4.1 역할

현재 Pilot의 실제 모델 구현은 **PyTorch / torchvision 공식 API를 Primary implementation source로 사용한다.**

공식 문서:

- PyTorch: https://docs.pytorch.org/
- torchvision: https://docs.pytorch.org/vision/
- MobileNetV3-Small:
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html
- GRU:
  https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html

검토 기준일:

```text
2026-08-29
```

---

## 4.2 MobileNetV3-Small

Design Baseline:

```text
MobileNetV3-Small
+
ImageNet pretrained
```

을 그대로 따른다.

구현 시 공식 API를 기준으로 확인한다.

```python
torchvision.models.mobilenet_v3_small(...)
torchvision.models.MobileNet_V3_Small_Weights
```

현재 torchvision 공식 문서에서는:

```text
MobileNet_V3_Small_Weights.DEFAULT
=
MobileNet_V3_Small_Weights.IMAGENET1K_V1
```

로 정의되어 있다.

### 구현 시 참고할 항목

- pretrained weights loading
- classifier 구조
- feature extraction 위치
- 입력 normalization
- resize / crop 기준
- pretrained weights와 transform의 대응 관계
- 모델 parameter freeze / unfreeze

### 주의

공식 weights의 inference transform은 구현 참고 기준이다.

Stage A training augmentation은:

```text
Baseline의 "최소 합리적 augmentation"
+
Phase 5 config
```

에서 별도로 정의한다.

즉 공식 inference transform을 그대로 training augmentation 전체로 간주하지 않는다.

---

## 4.3 Stage A embedding 정의

Design Baseline의 Stage A ↔ Stage B contract:

```text
Stage A Input : [B, 3, H, W]
Stage A Output: [B, D]

Stage B Input : [B, T, D]
```

를 만족해야 한다.

Embedding은 **3-class classifier logits가 아니라 classifier 이전의 visual representation**을 사용한다.

구현 전 반드시 실제 torchvision MobileNetV3-Small 구조를 확인하고:

- embedding extraction 위치
- `D`
- classifier 입력 dimension

을 코드와 config/test에서 일관되게 고정한다.

Encoder A와 Encoder B는 반드시 동일한 embedding interface를 가져야 한다.

```text
Encoder A
ImageNet pretrained

Encoder B
ImageNet pretrained
→ AI-Hub fine-tuned
```

두 encoder의 Stage B 입력 shape은 동일해야 한다.

---

## 4.4 Fine-tuning

Stage A에서 어떤 block까지 학습할지는 Design Baseline에서:

```text
일부 block / head
→ 구현 시 확정
```

상태다.

따라서 다음을 지킨다.

- freeze/unfreeze 범위를 config로 표현
- 코드에 특정 block 수를 분산 하드코딩하지 않음
- optimizer에는 실제 trainable parameter만 전달
- 학습 전 trainable parameter summary 확인 가능하게 구현
- 선택한 범위는 MLflow/config에 기록

공식 PyTorch의 transfer learning 패턴을 우선 참고한다.

---

## 4.5 GRU

Stage B Candidate:

```text
[B, T, D]
↓
GRU
↓
sequence representation
↓
Linear
↓
3-class
```

구현은:

```python
torch.nn.GRU
```

를 사용한다.

Design Baseline의 Stage B input이 `[B, T, D]`이므로 구현에서는 `batch_first=True` 사용을 우선 검토한다.

다만 다음 값은 Phase 7 config에서 확정한다.

- `hidden_size`
- `num_layers`
- `dropout`
- sequence representation 선택 방식

Design Baseline에서 `GRU hidden = 64~128 후보`이므로 임의로 과도한 hidden size를 선택하지 않는다.

### 금지

- GRU cell 자체를 새로 재구현
- LSTM으로 임의 변경
- bidirectional GRU를 근거 없이 기본값으로 추가
- attention layer를 기본 구조에 추가
- temporal Transformer로 변경

이들은 현재 2×2 Pilot 비교 범위를 바꾼다.

---

# 5. External Reference A — dronefreak/human-action-classification

## 5.1 Repository

```text
Repository:
https://github.com/dronefreak/human-action-classification

License:
Apache License 2.0

검토 기준일:
2026-08-29
```

이 repository는 PyTorch 기반 human action classification 프로젝트이며 현재 다음 성격의 pipeline을 제공한다.

- image action recognition
- MediaPipe 기반 pose 처리
- video action recognition
- training pipeline
- inference pipeline
- CLI / Python API
- checkpoint/model loading

---

## 5.2 현재 Pilot에서의 역할

이 repository의 역할은:

> **Software implementation pattern reference**

이다.

즉 이 repository를 **현재 Pilot의 model architecture authority로 사용하지 않는다.**

가장 중요한 참고 영역은:

```text
프로젝트/모듈 책임 분리
training orchestration
video inference orchestration
model loading
device 처리
preprocess → model → output 흐름
CLI와 내부 Python API 분리
```

이다.

---

## 5.3 ADAPT 범위

다음 유형의 구현 패턴을 참고할 수 있다.

### A. Inference orchestration

```text
video input
→ decode / sampling
→ preprocessing
→ model loading
→ forward
→ probability / label
→ output formatting
```

현재 프로젝트에서는 이를:

```text
Input Video
→ Frame Sampling
→ ROI / Fallback
→ Encoder
→ Stage B
→ Probability
→ Label
```

에 맞게 재구성한다.

### B. 책임 분리

예:

```text
model definition
training
evaluation
inference
CLI
checkpoint loading
```

을 하나의 대형 script에 몰아넣지 않는 방식.

### C. Inference 기본 패턴

- device 명시
- `model.eval()`
- gradient 비활성화
- checkpoint load
- preprocessing과 model 호출 분리
- 결과 변환 분리

---

## 5.4 DO NOT ADOPT 범위

다음은 현재 Pilot에서 가져오지 않는다.

```text
MC3-18
R3D-18
3D CNN video pipeline
UCF-101 전용 frame 수 / resize / normalization
HMDB51 전용 설정
repo의 class taxonomy
repo의 dataset split
repo의 benchmark 값을 현재 기대 성능으로 사용하는 것
```

이유:

Design Baseline에서 이미:

- 3D CNN 제외
- Video Transformer 제외
- TSM 등 video end-to-end 구조 제외
- MobileNetV3-Small + Stage B 구조 사용

으로 확정되어 있기 때문이다.

---

## 5.5 MediaPipe 관련 주의

해당 repository가 MediaPipe를 사용하더라도 현재 프로젝트의 ROI 정책은:

```text
Phase 4 ROI Preflight에서 확정된 현재 프로젝트 ROI 구현
```

을 기준으로 한다.

외부 repository의 pose/ROI 방식이 현재 ROI 결과와 다르더라도:

```text
현재 프로젝트 ROI 정책 우선
```

이다.

Phase 5 이후 ROI 로직을 이 repository에 맞춰 재설계하지 않는다.

---

# 6. External Reference B — prouast/deep-intake-detection

## 6.1 Repository

```text
Repository:
https://github.com/prouast/deep-intake-detection

License:
MIT

검토 기준일:
2026-08-29
```

Repository 목적:

```text
video에서 food / drink intake gesture detection
```

이다.

공개 구현은 TensorFlow 기반이며 다음과 같은 모델군을 포함한다.

- 2D CNN
- CNN-LSTM
- 3D CNN
- Two-Stream
- SlowFast
- ResNet 계열 변형

또한 CNN-LSTM 등을 2D CNN checkpoint에서 warm-start하는 구조를 제공한다.

---

## 6.2 현재 Pilot에서의 역할

이 repository는:

> **Methodology / research reference**

로 사용한다.

현재 프로젝트와 문제 도메인이 완전히 같지는 않지만:

```text
섭취 행동
+
RGB appearance
+
temporal context
```

를 다룬다는 점에서 설계 근거로 의미가 있다.

---

## 6.3 참고 가능한 방법론

### A. Appearance information의 중요성

공개 결과에서는 frame appearance를 사용하는 모델이 optical-flow-only 모델보다 강한 결과를 보인다.

현재 Pilot에서:

```text
RGB를 주 입력으로 사용
```

하는 방향과 정합된다.

다만 이 결과를 현재 데이터셋에서의 성능 보장으로 해석하지 않는다.

---

### B. Temporal context 비교

2D CNN, CNN-LSTM, 3D CNN, SlowFast 등 temporal context를 다루는 구조를 비교한다.

현재 Pilot에서는 이 아이디어를 축소하여:

```text
Mean Pooling + Linear
VS
GRU + Linear
```

로 temporal modeling의 효과를 검증한다.

즉 **temporal context를 별도 실험으로 검증한다는 방법론**을 참고한다.

---

### C. Warm-start / staged training 개념

repository에는 2D CNN에서 CNN-LSTM 등으로 warm-start하는 구성이 있다.

현재 Pilot과 완전히 같은 stage 정의는 아니지만:

```text
visual representation을 먼저 확보
→ temporal model이 이를 사용
```

한다는 staged-learning 관점은 참고할 수 있다.

현재 프로젝트의 실제 Stage 정의는 반드시 Design Baseline을 따른다.

```text
Stage A
AI-Hub visual encoder training

Stage B
ETRI frozen embedding
→ Mean / GRU clip classifier
```

외부 repository의 stage 정의로 교체하지 않는다.

---

## 6.4 직접 적용하지 않는 항목

다음은 현재 Pilot에서 사용하지 않는다.

```text
TensorFlow training code 직접 이식
TFRecord pipeline
TV-L1 optical flow
Two-Stream
SlowFast
3D CNN
frame-level binary intake detector
threshold search 기반 event detection
OREBA dataset split
OREBA evaluation protocol 그대로 적용
```

특히 이 repository의 task는 frame-level intake detection 성격이 강하고,
현재 Pilot은:

```text
clip-level
복약 / 음수 / 기타
3-class classification
```

이므로 평가 단위가 다르다.

---

# 7. Phase별 Reference 사용 지침

## Phase 5 — Stage A Visual Encoder

### Primary

```text
torchvision 공식 MobileNetV3-Small
PyTorch transfer learning 공식 패턴
Design Baseline Stage A
```

### Secondary

```text
dronefreak/human-action-classification
→ training module 분리
→ model loading / evaluation 구조 참고
```

### 구현 시 확인

- MobileNetV3-Small ImageNet weights
- classifier 교체
- embedding extraction 위치
- trainable block
- preprocessing / normalization
- frame-level evaluation
- video-level logit mean evaluation
- checkpoint provenance

### 참고하지 않음

- dronefreak 3D CNN
- deep-intake TensorFlow model code
- external dataset split

---

## Phase 6 — ETRI Embedding Cache

### Primary

```text
Design Baseline Stage A ↔ Stage B interface
torchvision MobileNetV3-Small 실제 구조
PyTorch inference pattern
```

### Secondary

```text
dronefreak inference orchestration
```

### 구현 시 확인

```text
[B, 3, H, W]
→ encoder
→ [B, D]

clip
→ T frames
→ [T, D]
```

cache provenance 필수:

- encoder type
- checkpoint ID
- preprocessing config
- normalization
- ROI version
- sampling config
- T
- D
- source clip key

### 금지

- classifier logits를 embedding으로 저장
- Encoder A/B가 서로 다른 embedding definition 사용
- ROI/sampling config를 cache metadata에 남기지 않음

---

## Phase 7 — 2×2 Ablation

### Primary

```text
torch.nn.Linear
torch.nn.GRU
Design Baseline 2×2 조건
```

### Conceptual

```text
deep-intake-detection
→ temporal context를 별도 비교하는 방법론
```

### 반드시 동일하게 유지

- Fixed Pilot Manifest
- participant fold
- T
- ROI
- normalization
- sampling
- loss baseline
- seed policy
- encoder frozen

### 금지

- Exp별 augmentation/ROI/sampling 변경
- GRU 실험만 별도 encoder fine-tuning
- 결과를 보고 fold 변경
- Mean baseline 삭제

---

## Phase 8 — 구조 선택

외부 repository의 metric이나 benchmark로 모델을 선택하지 않는다.

Primary:

```text
ETRI participant-disjoint 5-fold mean Macro-F1
```

Secondary:

1. 복약 Recall
2. 음수 Recall
3. std
4. 단순성
5. latency / memory

Design Baseline 기준을 그대로 따른다.

---

## Phase 9 — Pilot deployment/check model

Reference:

```text
PyTorch checkpoint / state_dict 패턴
현재 프로젝트 model factory / config
```

선택된 구조를 ETRI Pilot 전체 valid sample로 재학습한다.

외부 reference를 이유로 구조를 변경하지 않는다.

Deployment/check model의 training-set 성능을 Pilot 정량 성능으로 보고하지 않는다.

---

## Phase 10 — Pipeline Integration

### Primary

```text
Design Baseline Final inference pipeline
현재 project modules
```

### Structural Reference

```text
dronefreak/human-action-classification
→ video inference orchestration
→ model load / preprocessing / prediction 책임 분리
```

최종 흐름:

```text
Input Video
↓
Frame Sampling
↓
MediaPipe ROI / Fallback
↓
Selected Frozen Encoder
↓
Embedding Sequence
↓
Selected Stage B
↓
Softmax / Probability
↓
복약 / 음수 / 기타
```

필수:

- `model.eval()`
- inference gradient 비활성화
- 동일 normalization
- 동일 ROI policy
- 동일 sampling
- 동일 class mapping
- checkpoint/config provenance validation

---

# 8. 코드 구조 참고 원칙

외부 repository의 파일 구조를 그대로 복제하지 않는다.

현재 프로젝트에서는 역할 중심으로 다음 수준의 분리를 권장한다.

```text
src/
├── data / preprocessing 관련 모듈
├── model 관련 모듈
├── training 관련 모듈
├── evaluation 관련 모듈
└── inference 관련 모듈

scripts/
├── train entry point
├── evaluate entry point
└── inference entry point
```

실제 파일명과 세부 구조는 기존 프로젝트 구조를 우선한다.

### 원칙

- reusable logic은 `src/`
- 실행 entry point는 `scripts/`
- config는 `configs/`
- 테스트는 `tests/`
- 대용량 cache/checkpoint는 SSD Work
- 코드에 실제 절대경로 하드코딩 금지

---

# 9. Interface 우선 원칙

외부 reference보다 **현재 프로젝트의 interface contract**를 우선한다.

핵심 contract:

```text
Stage A Input
[B, 3, H, W]

Stage A Embedding
[B, D]

Stage B Input
[B, T, D]

Stage B Output
[B, 3]
```

구현 시 다음을 테스트한다.

- batch dimension
- temporal dimension
- embedding dimension
- class index
- dtype
- device
- checkpoint compatibility

모델 내부 변경으로 이 contract가 달라져야 한다면 Coding Agent가 임의 변경하지 않고 먼저 변경 사유를 보고한다.

---

# 10. 외부 코드 재사용·라이선스 정책

## 10.1 단순 참고

다음은 코드 직접 재사용으로 보지 않는다.

- 공식 API 사용
- architecture 개념 참고
- module 책임 분리 방식 참고
- 함수 흐름 참고
- 일반적인 PyTorch training pattern 사용

---

## 10.2 코드 직접 재사용

외부 repository의 코드를 verbatim 또는 실질적으로 변형하여 가져오는 경우 최소 다음을 기록한다.

```text
repository
source file
license
commit SHA
참고한 line / function
현재 project에서의 사용 위치
변형 내용
```

필요하면 별도:

```text
THIRD_PARTY_NOTICES.md
```

를 생성한다.

---

## 10.3 현재 승인된 license 정보

검토 기준일 2026-08-29:

```text
dronefreak/human-action-classification
→ Apache License 2.0

prouast/deep-intake-detection
→ MIT License
```

코드 직접 재사용 전에는 repository의 현재 LICENSE와 대상 revision을 다시 확인한다.

---

# 11. Revision / Commit 고정 원칙

외부 GitHub repository는 시간이 지나면서 변경될 수 있다.

따라서 **실제로 특정 코드 파일을 참고하여 구현할 경우** 다음을 작업 기록에 남긴다.

```text
Repository:
Revision / Commit SHA:
Reviewed date:
Relevant file:
Reference type: ADAPT / direct reuse
Use:
```

단순 README 수준의 아이디어 참고에는 매번 commit pinning을 강제하지 않지만,
실제 코드 구조·함수 구현을 구체적으로 참고했다면 SHA 기록을 권장한다.

---

# 12. 구현 시 금지 사항

Coding Agent는 외부 Reference를 근거로 다음을 임의 수행하지 않는다.

- MobileNetV3-Small → 다른 backbone 변경
- Mean / GRU 외 temporal architecture 추가
- encoder를 ETRI에서 joint fine-tuning
- 3D CNN 추가
- SlowFast 추가
- optical flow branch 추가
- attention / Transformer 추가
- T 기본값 임의 변경
- split/fold 변경
- loss baseline 변경
- threshold gating 추가
- self_recorded 기반 tuning
- Pilot Manifest 변경
- 외부 데이터셋 class mapping 적용
- 외부 benchmark 기준으로 성능 판단

필요성이 발견되면:

```text
현재 구현 유지
→ issue / limitation / follow-up candidate로 기록
```

하고 Full Experiment 또는 별도 변경 문서 대상으로 남긴다.

---

# 13. Reference를 이용한 의사결정 규칙

구현 중 외부 자료에서 더 좋은 방식이 발견된 경우 다음 순서를 따른다.

```text
1. 현재 Baseline과 충돌하는가?
   ├─ Yes → 적용하지 않음
   │        → 후속 실험 후보로 기록
   │
   └─ No
       ↓
2. 현재 Phase의 interface와 호환되는가?
   ├─ No → 임의 적용하지 않음
   └─ Yes
       ↓
3. 구현 단순성 / 재현성 / 테스트 가능성을 개선하는가?
   ├─ No → 적용하지 않음
   └─ Yes
       ↓
4. 공식 API로 해결 가능한가?
   ├─ Yes → 공식 API 우선
   └─ No → 승인된 외부 reference를 ADAPT
```

---

# 14. Coding Agent 사전 체크리스트

모델·학습·평가·inference 관련 코드를 작성하기 전에:

- [ ] 현재 Phase를 `STATUS.md`에서 확인했다.
- [ ] `docs/00_Pilot_Design_Baseline.md`의 해당 section을 확인했다.
- [ ] `AGENTS.md`의 현재 작업 규칙을 확인했다.
- [ ] 본 문서의 해당 Phase Reference를 확인했다.
- [ ] 공식 PyTorch / torchvision API를 먼저 확인했다.
- [ ] 기존 project interface / tests를 확인했다.
- [ ] 외부 GitHub는 승인된 범위에서만 참고한다.
- [ ] 외부 구현 때문에 Baseline을 변경하지 않는다.
- [ ] tensor shape contract를 확인했다.
- [ ] 데이터 split / manifest를 변경하지 않는다.
- [ ] 직접 코드 재사용 시 license / source / revision을 기록한다.

---

# 15. 작업 완료 후 기록 기준

모델 관련 작업 결과 보고에는 가능하면 다음을 포함한다.

```text
Implementation source:
- PyTorch / torchvision official API

External references consulted:
- repository
- relevant module / concept
- ADAPT 범위

Not adopted:
- 확인했지만 현재 Pilot 범위상 채택하지 않은 항목

Project implementation:
- 생성 / 수정 파일
- interface
- tests
- config

Baseline impact:
- none
또는
- 변경 필요 사항 발견
```

Baseline impact가 `none`이면 Design Baseline은 수정하지 않는다.

---

# 16. Project Code ↔ Reference 추적 예시

실제 파일이 생성되면 아래 표를 작업 결과에 맞게 갱신하거나 별도 구현 기록에 남길 수 있다.

| Project 구현 | Primary Reference | Secondary Reference | 참고 범위 |
|---|---|---|---|
| Stage A visual encoder | torchvision MobileNetV3-Small | dronefreak HAC | 공식 모델 API / module 분리 |
| Stage A training | PyTorch 공식 training pattern | dronefreak HAC | freeze / optimizer / training orchestration |
| ETRI embedding extractor | torchvision + PyTorch | dronefreak HAC | feature extraction / inference orchestration |
| Mean classifier | PyTorch | - | mean + Linear |
| GRU classifier | `torch.nn.GRU` | deep-intake methodology | sequence model / temporal 비교 |
| CV evaluation | scikit-learn + project policy | deep-intake methodology | metric 관점만 참고 |
| Final inference | PyTorch + Design Baseline | dronefreak HAC | orchestration / module 책임 분리 |

이 표는 구현 파일명이 실제로 확정된 뒤 구체화한다.

---

# 17. Reference별 핵심 요약

## PyTorch / torchvision

```text
역할
→ 실제 구현의 기준

ADOPT
→ MobileNetV3-Small
→ ImageNet weights
→ GRU
→ Linear
→ CE
→ freeze / eval / no_grad / state_dict

원칙
→ 공식 API 우선
```

---

## dronefreak/human-action-classification

```text
역할
→ software implementation pattern reference

ADAPT
→ project/module structure
→ training/inference orchestration
→ preprocessing → model → output 책임 분리
→ model/device/checkpoint handling

DO NOT ADOPT
→ 3D CNN architecture
→ MC3 / R3D
→ UCF/HMDB pipeline
```

---

## prouast/deep-intake-detection

```text
역할
→ methodology / intake-recognition research reference

ADAPT
→ appearance 중심 접근의 근거
→ temporal context 비교 관점
→ staged / warm-start learning 개념
→ 평가 방법론 아이디어

DO NOT ADOPT
→ TensorFlow code
→ TFRecord
→ optical flow
→ Two-Stream
→ SlowFast
→ 3D CNN
→ threshold event detector
```

---

# 18. Phase 5 시작 전 최종 기준

Phase 5 Stage A 구현은 다음 기준으로 시작한다.

```text
Design Baseline
    ↓
MobileNetV3-Small
ImageNet pretrained
AI-Hub Pilot fine-tuning

Implementation
    ↓
torchvision official model/weights
PyTorch official training APIs

Structure reference
    ↓
dronefreak/human-action-classification
필요한 범위만 ADAPT

Methodology reference
    ↓
prouast/deep-intake-detection
코드 직접 이식 X
```

Phase 5 구현에서 외부 Reference의 존재는 **모델 구조를 다시 선택하기 위한 근거가 아니라**,
이미 확정된 Pilot 구조를 더 안정적이고 재현 가능하게 구현하기 위한 참고 기준이다.

---

# 부록 A. 현재 Pilot 모델 기준 요약

```text
AI-Hub
↓
MediaPipe ROI / Fallback
↓
MobileNetV3-Small
↓
Stage A fine-tuning
↓
AI-Hub Fine-tuned Encoder

ETRI
↓
T=64 uniform frame sampling
↓
ROI
↓
Frozen Encoder
↓
Frame Embedding Sequence
↓
┌──────────────────────┐
│ Mean + Linear        │
│         VS           │
│ GRU + Linear         │
└──────────────────────┘
↓
복약 / 음수 / 기타
```

2×2:

| Encoder | Stage B |
|---|---|
| ImageNet-only | Mean |
| AI-Hub fine-tuned | Mean |
| ImageNet-only | GRU |
| AI-Hub fine-tuned | GRU |

이 구조는 본 문서의 외부 Reference로 변경하지 않는다.

---

# 부록 B. Reference URL

## Official

- PyTorch Documentation
  https://docs.pytorch.org/

- torchvision Documentation
  https://docs.pytorch.org/vision/

- torchvision MobileNetV3-Small
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html

- PyTorch GRU
  https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html

## External GitHub

- dronefreak/human-action-classification
  https://github.com/dronefreak/human-action-classification

- prouast/deep-intake-detection
  https://github.com/prouast/deep-intake-detection

---

# 부록 C. 변경 관리

이 문서는 Design Baseline과 달리 **구현 Reference 가이드**이므로 다음 상황에서 갱신할 수 있다.

- 공식 PyTorch / torchvision API 사용 기준 변경
- 실제 참고한 GitHub revision / file 기록 추가
- 새로운 승인 Reference 추가
- 기존 Reference의 참고 범위 수정
- Phase별 구현에서 실제로 사용한 reference 추적 정보 추가

다만 다음 변경은 이 문서만 수정해서 처리하지 않는다.

- 모델 architecture 변경
- 데이터 역할 변경
- split/evaluation 정책 변경
- 2×2 실험 구조 변경
- self_recorded 정책 변경

위 변경은 Design Baseline의 변경 관리 원칙을 따른다.
