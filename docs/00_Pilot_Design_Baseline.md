# 복약-음수 행동 인식 AI 비전 모델 — 최종 1차 Pilot Design Baseline

**복약 / 음수 / 기타 3-class 영상 행동 인식 모델 — 설계·학습·평가·통합 기준**

- 최초 작성일: 2026-08-27
- 최종 개정일: 2026-08-29
- 문서 상태: **Final Design Baseline — 1차 Pilot 구현 기준**
- 프로젝트명: `medication-drinking-action-recognition`
- WSL Project Root: `/home/user/projects/medication-drinking-action-recognition`
- Host: Windows
- 개발·실행 환경: WSL 2 / Ubuntu 26.04.1 LTS
- GPU: NVIDIA GeForce GTX 1650 Ti, VRAM 4GB
- 상위 목적: 복약 / 물 마시기 / 기타 행동을 짧은 영상에서 분류하는 경량 비전 모델 Pilot 구현

---

# 0. 문서 역할과 운영 원칙

## 0.1 문서 목적

이 문서는 **복약 / 음수 / 기타 3-class 영상 행동 인식 모델의 1차 Pilot 개발을 위한 공식 설계 기준**이다.

1차 Pilot의 목적은 전체 데이터 규모에서 최고 성능을 확보하는 것이 아니라, 축소된 대표 데이터로 다음 전체 흐름을 한 번 완주하여 설계와 구현의 유효성을 빠르게 검증하는 것이다.

```text
환경·경로·Git 기준선 확정
→ Full Candidate Inventory
→ Fixed Pilot Manifest
→ ROI Preflight
→ Stage A Visual Encoder
→ ETRI Embedding Cache
→ Stage B Clip Classifier
→ 핵심 2×2 Ablation
→ participant-disjoint 5-fold CV End-to-End 평가
→ 선택 구성으로 Pilot deployment/check model 생성
→ ETRI + self_recorded 전체 inference pipeline 검증
→ 오류·한계·Full Experiment 확장 항목 정리
```

이 문서는 다음 역할을 가진다.

1. 1차 Pilot 구현의 상위 기술 기준
2. 데이터 전처리·학습·평가·통합 문서의 Reference
3. 모델·데이터·평가 정책에 대한 의사결정 기록
4. Pilot 완료 후 Full Dataset 실험으로 확장하기 위한 기준점
5. 향후 발표자료·포트폴리오의 기술 근거

이 문서는 실제 학습 결과를 기록하는 문서가 아니다. 실제 실험 결과와 변경된 의사결정은 별도 문서와 MLflow에 기록한다.

---

## 0.2 Pilot과 Full Experiment 구분

### 1차 Pilot — 현재 구현 범위

```text
축소된 대표 데이터
→ 전체 개발 흐름 구현
→ Stage A/B 학습·평가
→ 핵심 2×2 Ablation
→ 전체 inference pipeline 통합
→ ETRI participant-disjoint 5-fold CV End-to-End 평가
→ ETRI 실제 inference script 검증
→ self_recorded/pipeline_check 실행
→ 결과 정성 확인
```

Pilot 성능은 **최종 제품 성능 또는 광범위한 일반화 성능의 증거로 해석하지 않는다.**

### Full Experiment — Pilot 이후

```text
전체 target candidate
+
넓은 other pool
↓
데이터 규모 확대
↓
sampling / loss / augmentation / hyperparameter 실험
↓
최종 성능 및 일반화 검증
```

---

## 0.3 용어 정의

### 음수 class

본 프로젝트에서 **음수 class의 실제 목표 행동은 물 마시기**다.

- ETRI A004 = 물 마시기 → 최종 음수 target
- AI-Hub Drink_bever / Drink_alcohol → Stage A 시각 특징 학습용 auxiliary positive

AI-Hub의 술·음료 데이터를 사용한다고 해서 최종 class 의미를 “모든 음료 마시기”로 확장하지 않는다.

### End-to-End

본 문서의 **End-to-End 평가/검증**은:

```text
영상 입력
→ frame sampling
→ ROI
→ encoder
→ Stage B
→ 최종 3-class 출력
```

의 **전체 inference path**를 의미한다.

CNN과 temporal model을 한 번에 공동 최적화하는 **end-to-end video training**을 의미하지 않는다.

---

## 0.4 문서 운영 원칙

확정된 기준 문서는 해당 시점의 의사결정 기록으로 보존한다.

- Design Baseline / EDA / 데이터 구조 / 환경 기준 문서는 기준 시점 기록으로 보존
- 의미 있는 설계 변경은 기존 문서를 덮어쓰기보다 **새 버전 또는 변경 문서**로 기록
- 단순 오타·링크·표현 정리는 기존 문서 수정 가능
- `README.md`, `AGENTS.md`, `STATUS.md`는 최신 상태를 유지하는 living document
- 실험 결과는 별도 결과 문서와 MLflow에 기록
- 변경 문서에는 가능하면 `변경 전 / 변경 후 / 이유 / 영향 범위 / 적용 시점`을 남긴다

---

# 1. 현재 프로젝트 상태

## 1.1 완료 상태

- 학습용 데이터 확보
- AI-Hub / ETRI 데이터 구조 분석
- 복약 / 음수 / 기타 관점 EDA
- 모델 아키텍처 리서치
- 1차 모델 아키텍처 설계
- Pilot 핵심 정책 정의
- SSD Raw / Working 실제 경로 확정
- WSL 2 / Ubuntu / VS Code 개발환경 구축
- 최종 WSL Project Root 확정
- 최종 Project Root에서 Python 3.12 `.venv` 재구축
- VS Code Python Interpreter 재연결
- PyTorch CUDA / MediaPipe / OpenCV / MLflow 설치·import 검증
- GTX 1650 Ti CUDA 실제 tensor 연산 검증
- `.python-version` / `requirements-lock.txt` 생성
- AI-Hub / ETRI 구조·EDA 문서 `docs/` 배치
- MLflow DB를 `runtime/mlflow/mlflow.db`로 이동
- `.gitignore` 최종 검토
- 개발환경 구축 기록 문서 작성
- 최종 Design Baseline 검토

## 1.2 개발 시작 직전 남은 작업

```text
최종 Design Baseline을 docs/에 배치
→ git init -b main
→ git status / git check-ignore 검증
→ git add -n . dry-run
→ 프로젝트 기준선 첫 commit
→ Pilot 구현 시작
```

---

# 2. 개발 환경·경로·Git 기준

## 2.1 Host / WSL

- Host OS: Windows
- 개발·실행: WSL 2
- Linux Distribution: Ubuntu 26.04.1 LTS (Resolute)
- IDE: Windows VS Code + WSL
- 주 개발 터미널: VS Code 내부 WSL Ubuntu terminal
- Python Interpreter: `<WSL_PROJECT_ROOT>/.venv/bin/python`

## 2.2 Python / GPU

- Python 관리: `uv`
- 프로젝트 Python: 3.12.14
- `.python-version`: `3.12`
- 가상환경: `<WSL_PROJECT_ROOT>/.venv`
- PyTorch: 2.13.0+cu126
- Torchvision: 0.28.0+cu126
- CUDA runtime: 12.6
- MediaPipe: 1.0.1
- OpenCV: 5.0.0
- NumPy: 2.5.2
- Pandas: 2.3.3
- Scikit-learn: 1.9.0
- MLflow: 3.15.2
- 설치 상태 snapshot: `requirements-lock.txt`

GPU:

- NVIDIA GeForce GTX 1650 Ti
- VRAM 4GB
- `torch.cuda.is_available() == True`
- CUDA tensor 연산 성공 확인

---

## 2.3 실제 경로

| 논리 경로 | Windows | WSL |
|---|---|---|
| `<SSD_RAW_ROOT_AI_HUB>` | `D:\AI 도약과정\데이터\data_raw\057.일상생활_영상_데이터\01.데이터` | `/mnt/d/AI 도약과정/데이터/data_raw/057.일상생활_영상_데이터/01.데이터` |
| `<SSD_RAW_ROOT_ETRI>` | `D:\AI 도약과정\데이터\data_raw\고령자 일상행동인식 3차원 영상 데이터셋(리빙랩)` | `/mnt/d/AI 도약과정/데이터/data_raw/고령자 일상행동인식 3차원 영상 데이터셋(리빙랩)` |
| `<SSD_WORK_ROOT>` | `D:\AI 도약과정\데이터\data_workspace\medication_drinking_action_data` | `/mnt/d/AI 도약과정/데이터/data_workspace/medication_drinking_action_data` |
| `<WSL_PROJECT_ROOT>` | `\\wsl.localhost\Ubuntu\home\user\projects\medication-drinking-action-recognition` | `/home/user/projects/medication-drinking-action-recognition` |

### 경로 사용 원칙

- Raw 데이터는 immutable source
- Raw 데이터 수정·삭제·rename 금지
- 파생 데이터는 `<SSD_WORK_ROOT>`에 저장
- 코드·config·manifest·문서는 `<WSL_PROJECT_ROOT>`에서 관리
- Python/config에서는 WSL `/mnt/d/...` 경로 사용
- 공백·한글 경로는 shell에서 따옴표로 감싼다
- 코드에 절대경로를 하드코딩하지 않는다

권장 path config:

```text
configs/
├── paths.example.yaml     # Git 관리
└── paths.local.yaml       # Git 제외
```

`paths.local.yaml`은 현재 PC의 실제 절대경로를 저장한다.

---

## 2.4 프로젝트 구조

개발 시작 기준 구조:

```text
<WSL_PROJECT_ROOT>/
├── .venv/                         # Git 제외
├── configs/
├── docs/
│   ├── 00_Pilot_Design_Baseline.md
│   ├── 01_개발환경_구축_기록.md
│   └── AI-Hub / ETRI 구조·EDA reference
├── manifests/
├── scripts/
├── src/
├── tests/
├── runtime/                       # Git 제외
│   └── mlflow/
│       └── mlflow.db
├── .gitignore
├── .python-version
├── requirements-lock.txt
├── AGENTS.md
├── README.md
└── STATUS.md
```

상위 디렉토리는 첫 구현 단위에 맞춰 생성하고, 불필요한 빈 하위 디렉토리를 미리 과도하게 만들지 않는다.

---

## 2.5 Git 관리 기준

### Git 관리 대상

- `src/`
- `scripts/`
- `configs/` 중 공유 config
- `docs/`
- `tests/`
- `manifests/`
- `.python-version`
- `requirements-lock.txt`
- `README.md`
- `AGENTS.md`
- `STATUS.md`
- 소형 metadata / experiment definition

### Git 제외 대상

최소:

```text
.venv/
runtime/
mlflow.db
mlruns/
mlartifacts/
.env
configs/paths.local.yaml
logs/
*.log
cache/
embeddings/
checkpoints/
*.pt
*.pth
*.ckpt
*.onnx
```

Raw / Working 대용량 데이터와 SSD 산출물은 Git에 넣지 않는다.

---

# 3. 프로젝트 목표와 1차 Pilot 범위

## 3.1 최종 목표

짧은 영상 클립을 입력받아 다음 중 하나를 출력하는 경량 영상 행동 인식 모델을 구현한다.

```text
복약
음수 = 물 마시기
기타
```

GTX 1650 Ti 4GB 수준에서 학습·추론 가능한 구조를 우선한다.

---

## 3.2 1차 Pilot 범위

- 입력: 짧은 영상 클립
- 출력: 3-class logits / probability / label
- Stage A: AI-Hub 이미지 기반 visual encoder 학습
- Stage B: ETRI Batch B 기반 clip-level classifier 학습
- Transfer 비교: ImageNet-only vs AI-Hub fine-tuned
- Temporal 비교: Mean Pooling + Linear vs GRU
- 핵심 2×2 ablation
- ETRI participant-disjoint 5-fold CV End-to-End 평가
- 선택된 구성으로 전체 inference pipeline 통합
- self_recorded 영상에서 기능적 pipeline check 및 정성 확인

---

## 3.3 self_recorded 사용 범위

직접 촬영 데이터:

```text
<SSD_WORK_ROOT>/self_recorded/pipeline_check/
```

사용:

- 전체 inference pipeline 동작 검증
- 최종 label / confidence 생성 확인
- 기능적 오류 확인
- 결과 정성 확인

사용 금지:

- 학습
- validation
- test metric 산출
- ROI 방식 선정
- ROI Preflight
- frame sampling 방식 결정
- threshold 결정
- hyperparameter tuning
- model selection
- preprocessing tuning
- self_recorded 결과를 보고 fine-tuning

self_recorded에서 발견된 **기능적 코드 오류**는 수정할 수 있다.

---

# 4. 데이터 역할과 Class Taxonomy

## 4.1 최종 Class Taxonomy

```text
복약
├─ ETRI A003
└─ AI-Hub Take_pills

음수 = 물 마시기 행동
├─ ETRI A004 물 마시기
└─ AI-Hub Stage A auxiliary positive
   ├─ Drink_bever
   └─ Drink_alcohol

기타
├─ ETRI 나머지 행동
│  └─ Stage B에서 A045~A048 제외
└─ AI-Hub 나머지 102개 행동
```

AI-Hub의 Drink_bever / Drink_alcohol은 **물 마시기와 유사한 손-용기-입 시각 패턴을 학습하기 위한 proxy**다.

물 이외 음료 일반화는 현재 필수 목표가 아니다.

---

## 4.2 ETRI 데이터 역할

ETRI-Activity3D-LivingLab:

| 데이터 | 형식 | 클립 수 |
|---|---|---:|
| RGB | MP4 | 8,622 |
| 3D Skeleton | CSV | 8,622 |
| Body Index | PNG | 8,622 |
| Depth Map | 미보유 | — |

Batch:

| Batch | 참가자 | 클립 수 | Pilot |
|---|---|---:|---|
| A | P01~P20 | 2,033 | 제외 |
| B | P201~P230 | 6,589 | 사용 |

Stage B는 Batch B만 사용한다.

Batch B raw:

- 복약 A003: 119
- 음수 A004: 120
- 기타: 6,350

기타 후보에서 제외:

- A045~A048 다중 인물 행동
- 손상 / 비정상 RGB
- validation에서 invalid 판정된 sample

---

## 4.3 ETRI Pilot subset

목표:

| 클래스 | 선택 원칙 | 예상 수 |
|---|---|---:|
| 복약 | participant당 최대 2개 | 약 59 |
| 음수 | participant당 최대 2개 | 약 60 |
| 기타 | participant당 약 4개 | 약 120 |
| 합계 | 약 1:1:2 | 약 239 |

원칙:

- 30 participant를 가능하면 모두 유지
- P201~P206의 12 take도 target class당 최대 2개
- P201~P206은 가능하면 H070/H120 모두 포함
- P207~P230의 기존 2 take 우선 활용
- 기타는 가능하면 hard negative 2 + 일반 기타 2
- seed 고정
- selection reason manifest 기록
- 동일 participant의 모든 clip은 같은 fold

실제 수치는 inventory 검증 후 확정한다.

---

## 4.4 AI-Hub 데이터 역할

현재 로컬본:

- JSON / video instance: 18,420
- JPG: 55,260
- bbox: 55,558
- 행동 클래스: 105
- 원본 MP4: 현재 로컬 미보유

### Candidate Inventory master 기준

**AI-Hub Full Candidate Inventory의 master는 현재 로컬에 실제 존재하는 Training + Validation JSON/file 18,420건이다.**

메타데이터 xlsx는 원본 전체 20,517건 기준이므로 **보조 metadata source**로만 사용한다.

```text
Local Training + Validation JSON/File Inventory
↓
18,420 local samples
↓
metadata xlsx와 교집합 join
↓
viewpoint_3
↓
class mapping
↓
actor-disjoint split
```

로컬에 실제 존재하지 않는 metadata row를 candidate로 만들지 않는다.

---

## 4.5 AI-Hub Stage A mapping

```text
Take_pills (#3)      → 복약
Drink_bever (#2)     → 음수 auxiliary positive
Drink_alcohol (#5)   → 음수 auxiliary positive
나머지               → 기타
```

viewpoint_3 기준:

| Class | Video | Frame |
|---|---:|---:|
| 복약 | 111 | 333 |
| 음수 auxiliary | 225 | 675 |
| 기타 | 10,652 | 31,956 |
| 합계 | 10,988 | 32,964 |

---

## 4.6 AI-Hub Pilot subset

목표:

| 클래스 | Pilot video | 예상 JPG | 선택 원칙 |
|---|---:|---:|---|
| 복약 | 약 100 | 약 300 | actor 다양성 우선 |
| 음수 auxiliary | 약 100 | 약 300 | Drink_bever / Drink_alcohol 모두 포함 |
| 기타 | 약 200 | 약 600 | Eat_food 포함 + 다양한 기타 |
| 합계 | 약 400 | 약 1,200 | 약 1:1:2 |

원칙:

- viewpoint_3만 사용
- actor-disjoint split을 **먼저** 확정
- split 내부에서 Pilot subset 선택
- 동일 video의 JPG 3장은 항상 함께 선택
- 복약 / 음수는 actor 다양성 우선
- 동일 actor 과도한 반복 방지
- 기타는 Eat_food를 필수 hard negative 후보로 포함
- seed / selection reason 기록

---

# 5. EDA 핵심 결과와 설계 반영

## 5.1 클래스 불균형

원본 candidate는 기타가 압도적으로 많다.

Pilot에서는 최종 데이터 분포를 재현하기보다 전체 pipeline 완주를 우선하므로 약 1:1:2로 축소한다.

Pilot 기본:

```text
Fixed Pilot Manifest
+
Standard CrossEntropy
```

Pilot 필수 기본값에서 제외:

- class-weighted CE
- focal loss
- epoch-wise 대규모 other undersampling

이들은 Pilot 결과 이후 또는 Full Experiment에서 검토한다.

---

## 5.2 Viewpoint

AI-Hub 복약 / 음수 관련 class는 3인칭 고정 시점에 존재한다.

설계:

```text
Stage A = viewpoint_3 only
```

기타에 1인칭을 섞어 viewpoint 자체를 class 단서로 학습하게 하지 않는다.

---

## 5.3 Leakage

### AI-Hub

split unit:

```text
video / JSON instance
```

group:

```text
actor
```

동일 video의 JPG 3장은 같은 split.

### ETRI

group:

```text
participant
```

동일 participant의 모든 take는 같은 fold.

속도를 위해 leakage 방지 규칙을 완화하지 않는다.

---

## 5.4 Hard Negative

### AI-Hub

대표:

- Eat_food

Drink_alcohol은 현재 auxiliary positive로 사용한다.

### ETRI

대표 후보:

- A040 두 손으로 얼굴 비비기
- A016 머리 빗기
- A021 안경 쓰기/벗기
- A035 전화 통화
- A017 드라이어 사용
- A043 어깨 주무르기
- A014 화장
- A010 이 닦기
- A012 세수
- A013 얼굴 닦기
- A038 담배
- A001 포크로 음식 먹기

Pilot other에 의도적으로 일부 포함한다.

---

## 5.5 RGB / Skeleton

ETRI EDA에서 복약은 skeleton 단독으로 기타와 분리하기 어렵다.

따라서 현재 Pilot의 주 입력은 RGB다.

Skeleton은:

- EDA
- hard-negative 선정
- 향후 multimodal candidate

로 사용한다.

---

## 5.6 Sequence length / Frame Sampling

ETRI 타깃 clip은 중앙값 약 170 frame 수준이며, 64~96 frame 균일 sampling이 현실적인 시작점이다.

Pilot 기준:

```text
Smoke test       : T = 16
기본 Pilot 실험  : T = 64
Optional ablation: T = 96
```

기본 sampling:

```text
fixed-T uniform temporal sampling
```

fixed-FPS는 필요 시 후속 비교한다.

---

# 6. Manifest와 데이터 검증 기준

## 6.1 Full Candidate Manifest → Pilot Manifest

원본을 삭제하지 않는다.

```text
Raw
↓
Full Candidate Inventory
↓
validation / exclusion flag
↓
split / fold
↓
Pilot selection
↓
Fixed Pilot Manifest
↓
실제 preprocessing / training
```

`pilot_selected=false`인 candidate도 inventory에서 유지한다.

---

## 6.2 경로 저장 정책

Manifest에는 PC별 절대경로 자체를 핵심 식별자로 저장하지 않는다.

기본:

```text
dataset
root_key
relative_path
```

실행 시:

```text
absolute_path = path_config[root_key] / relative_path
```

로 조립한다.

AI-Hub 3 frame도 가능하면 `frame_relative_paths`로 기록한다.

---

## 6.3 ROI status

enum:

```text
pending
success
partial
fallback
```

- inventory 최초 생성: `pending`
- ROI 실행 후: `success / partial / fallback`

ROI 실패 sample을 자동 삭제하지 않는다.

---

## 6.4 필수 leakage / consistency 자동 검증

### AI-Hub

반드시 코드로 확인:

```text
train actor ∩ val actor = ∅
train actor ∩ test actor = ∅       # test를 별도로 둘 경우
동일 video의 3 JPG는 같은 split
pilot_selected sample은 valid=true
```

### ETRI

반드시 코드로 확인:

```text
각 fold train participant ∩ val participant = ∅
동일 participant의 모든 take는 동일 fold
A045~A048은 pilot_selected=false
invalid sample은 pilot_selected=false
```

Manifest validation 실패 시 학습 단계로 넘어가지 않는다.

---

# 7. ROI Preflight Gate

## 7.1 목적

MediaPipe ROI가 실제 데이터에서 충분히 작동하는지 확인한 뒤 Pilot preprocessing을 시작한다.

Preflight 이전에 Pilot 전체 데이터를 일괄 처리하지 않는다.

---

## 7.2 Preflight 데이터

사용:

1. AI-Hub viewpoint_3 대표 이미지
2. ETRI Batch B 대표 frame

범주:

- 복약
- 음수
- Eat_food
- 일반 기타
- ETRI hard negative

사용하지 않음:

- self_recorded

---

## 7.3 ROI 처리

| 상태 | 의미 | 처리 |
|---|---|---|
| success | 의도한 ROI 생성 | landmark ROI |
| partial | 손/얼굴 일부 검출 | 검출 영역 기반 확장 ROI |
| fallback | 정상 ROI 실패 | full-frame 또는 확정 fallback |

원칙:

- 실패 sample 삭제 금지
- class별 fallback rate 기록
- 특정 class에서 실패율이 높으면 ROI 정책 재검토
- ROI Preflight PASS 후에만 Pilot 전처리 진행

ROI 합격 기준의 세부 수치는 Preflight 직전 전처리 가이드에서 확정한다.

---

# 8. 전체 모델 아키텍처

## 8.1 구조

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
┌──────────────────────────────┐
│ Stage B                      │
│                              │
│ Mean Pooling + Linear        │
│             VS               │
│ GRU + Linear                 │
└──────────────────────────────┘
↓
복약 / 음수 / 기타
```

GRU를 자동으로 최종 모델로 간주하지 않는다.

---

## 8.2 Stage A ↔ Stage B interface

- Stage A Input: `[B, 3, H, W]`
- Stage A Output: `[B, D]`
- Stage B Input: `[B, T, D]`
- Stage B Output: `[B, 3]`

H/W, D는 구현 시 확정한다.

T 기본값은 64.

---

## 8.3 Visual Backbone

Baseline:

```text
MobileNetV3-Small
+
ImageNet pretrained
```

선택 이유:

- 4GB VRAM 환경
- 작은 target dataset
- transfer learning 활용
- Stage B에서 embedding cache 가능

---

# 9. Stage A — Visual Encoder

## 9.1 목적

AI-Hub로 복약 / 음수 auxiliary / 기타의 frame-level 시각 특징을 학습한다.

Stage A 모델의 역할:

1. AI-Hub frame classifier
2. ETRI frame embedding extractor

---

## 9.2 데이터 흐름

```text
Local AI-Hub JSON/File Inventory 18,420
↓
metadata xlsx intersection join
↓
viewpoint_3
↓
class mapping
↓
actor-disjoint split
↓
split 내부 Pilot selection
↓
약 400 video / 약 1,200 JPG
↓
ROI / resize / normalization
↓
MobileNetV3-Small fine-tuning
```

---

## 9.3 Stage A 평가

### Frame-level

- Macro-F1
- per-class Precision
- per-class Recall
- Confusion Matrix

### Video-level

동일 video의 3 frame logits:

```text
frame logits
↓
mean
↓
video prediction
```

평가:

- video-level Macro-F1
- per-class Recall
- Confusion Matrix

Stage A 모델 상태 판단에서는 video-level 결과를 중요하게 본다.

Stage A의 음수 성능은 auxiliary class 학습 상태이며 최종 물 마시기 성능이 아니다.

---

## 9.4 Stage A Transfer 대상

두 encoder를 만든다.

### Encoder A

```text
ImageNet pretrained MobileNetV3-Small
AI-Hub fine-tuning 없음
```

### Encoder B

```text
ImageNet pretrained MobileNetV3-Small
↓
AI-Hub Pilot fine-tuning
```

두 encoder의 실질적 가치는 ETRI Stage B 성능으로 판단한다.

---

# 10. Stage B — Clip-level Classifier

## 10.1 입력

ETRI Batch B Pilot clip을 T frame으로 sampling하고 각 frame을 encoder로 embedding한다.

```text
video
↓
T=64 uniform sampling
↓
ROI
↓
frozen encoder
↓
[B, T, D]
```

---

## 10.2 핵심 규칙 — Encoder Freeze

**2×2 Pilot Ablation에서는 visual encoder를 ETRI 학습 중 고정(frozen)한다.**

즉:

```text
ImageNet-only Encoder
→ frozen
→ ETRI embedding cache

AI-Hub Fine-tuned Encoder
→ frozen
→ ETRI embedding cache
```

ETRI에서 학습하는 것은 Stage B classifier다.

이 원칙을 사용하는 이유:

1. ImageNet-only vs AI-Hub transfer 비교를 깨끗하게 유지
2. 동일 encoder embedding을 재사용
3. 4GB VRAM 제약 대응
4. 2×2 비교 조건의 공정성 확보

ETRI 기반 encoder fine-tuning은 1차 Pilot 필수 범위에서 제외한다.

---

## 10.3 Embedding Cache

ETRI Pilot clip의 frame embedding은 `<SSD_WORK_ROOT>` 아래에 cache한다.

Git 제외.

cache metadata에는 최소 다음 provenance를 남긴다.

- encoder type
- encoder checkpoint ID
- preprocessing config
- normalization
- ROI config/version
- frame sampling config
- T
- D
- source clip key

---

## 10.4 Stage B 후보

### Baseline A — Mean Pooling + Linear

```text
[B, T, D]
↓
mean over T
↓
Linear
↓
3-class
```

최소 non-temporal baseline.

### Candidate B — GRU + Linear

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

Temporal modeling 효과를 검증한다.

---

# 11. 핵심 2×2 Ablation

|  | Mean Pooling + Linear | GRU + Linear |
|---|---|---|
| ImageNet-only Encoder | Exp A | Exp C |
| AI-Hub Fine-tuned Encoder | Exp B | Exp D |

분석:

- B vs A → AI-Hub Stage A transfer 효과
- C vs A → ImageNet embedding에서 temporal 효과
- D vs B → AI-Hub embedding에서 temporal 효과
- D vs A → 전체 2-stage 설계의 종합 효과

모든 실험은 동일하게 유지:

- Fixed Pilot Manifest
- participant folds
- ROI policy
- T
- frame sampling
- normalization
- class mapping
- loss baseline
- seed 정책
- encoder frozen

---

# 12. ETRI Split·평가·모델 선택

## 12.1 Split

```text
ETRI Batch B Pilot
↓
participant Group 5-Fold
```

- participant 30명 가능하면 전부 유지
- fold당 약 6명
- 동일 participant의 모든 clip/take는 같은 fold
- clip random split 금지

---

## 12.2 “held-out” 표현 기준

각 fold의 validation participant는 해당 fold training에서 제외되므로 **fold-level held-out**이다.

그러나 4개 구조를 동일 5-fold 결과로 비교·선택하므로, 이 Pilot에는 별도의 완전히 untouched final test set을 두지 않는다.

따라서 최종 정량 결과는 다음 명칭을 사용한다.

```text
ETRI participant-disjoint 5-fold CV End-to-End evaluation
```

별도의 “final held-out test 성능”으로 과장하지 않는다.

---

## 12.3 Primary model-selection metric

핵심 2×2 구조 선택의 primary metric:

```text
5-fold mean Macro-F1
```

반드시 같이 기록:

- Macro-F1 mean ± std
- 복약 Recall mean ± std
- 음수 Recall mean ± std
- 기타 Recall mean ± std
- fold별 Confusion Matrix
- out-of-fold aggregate Confusion Matrix

동일하거나 매우 근접한 경우 secondary 기준:

1. 복약 Recall
2. 음수 Recall
3. 안정성(std)
4. 모델 단순성
5. inference latency / memory

Pilot 결과를 본 뒤 primary metric을 바꾸지 않는다.

---

## 12.4 End-to-End 정량 평가

각 fold에서:

```text
Video
↓
Frame Sampling
↓
ROI / Fallback
↓
Encoder
↓
Stage B
↓
복약 / 음수 / 기타
```

실제 inference path로 평가한다.

기록:

- Macro-F1
- class Precision / Recall
- Confusion Matrix
- OOF prediction
- ROI status
- error type

embedding cache 기반 Stage B 성능뿐 아니라 실제 inference script 경로를 최소 1회 이상 검증한다.

---

# 13. Error Analysis

주요 오류 축:

- 복약 → 기타
- 복약 → 음수
- 음수 → 기타
- 음수 → 복약
- 기타 → 복약
- 기타 → 음수

추가 분석:

- Eat_food
- ETRI 손/얼굴 hard negative
- `roi_status=partial`
- `roi_status=fallback`
- AI-Hub → ETRI domain shift
- participant별 편차
- fold별 편차

Self-recorded failure는 정량 데이터에 합치지 않고 기능적 또는 정성적 failure case로만 기록한다.

---

# 14. Pilot deployment/check model과 Pipeline Integration

## 14.1 목적

5-fold CV에서 모델 구조를 선택한 뒤 self_recorded pipeline check에 사용할 단일 Stage B checkpoint가 필요하다.

따라서:

```text
2×2 participant-disjoint 5-fold CV
↓
구조 선택
↓
선택된 encoder / Stage B 구조 고정
↓
ETRI Pilot 전체 valid sample로 Stage B 재학습
↓
Pilot deployment/check checkpoint 생성
↓
ETRI inference check
↓
self_recorded/pipeline_check
```

Encoder는 선택된 frozen encoder를 사용한다.

---

## 14.2 중요한 평가 원칙

Pilot 전체 ETRI로 재학습한 deployment/check model의 training-set 성능은 **정량 평가 결과로 사용하지 않는다.**

정량 모델 비교와 성능 보고:

```text
participant-disjoint 5-fold CV 결과
```

Pipeline check / 데모용 checkpoint:

```text
Pilot 전체 데이터로 재학습한 deployment/check model
```

두 역할을 분리한다.

---

## 14.3 Integration Test

확인:

- checkpoint 경로
- encoder / Stage B 조합
- embedding dimension
- tensor shape
- label index
- normalization
- ROI / fallback
- frame sampling
- Stage A/B interface
- manifest / config provenance
- 예외 처리

---

## 14.4 Final inference pipeline

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

추론:

- `model.eval()`
- `torch.no_grad()`
- 학습과 동일 normalization
- 동일 ROI policy
- 동일 sampling
- 동일 class mapping

---

# 15. Experiment Tracking·재현성

## 15.1 MLflow

MLflow local DB:

```text
runtime/mlflow/mlflow.db
```

Git 제외.

첫 실험 전 확정:

- tracking URI
- artifact store
- experiment naming
- run naming
- checkpoint/artifact naming

---

## 15.2 각 run에 기록할 최소 정보

- Git commit hash
- config
- seed
- manifest version / hash
- encoder type
- encoder checkpoint
- Stage B type
- fold
- T
- sampling
- ROI config
- normalization
- loss
- LR
- batch size
- epoch
- metrics
- confusion matrix
- error artifact

---

## 15.3 재현성

고정/기록:

- Pilot selection seed
- split/fold seed
- train seed
- manifest
- config
- package snapshot
- Git commit

실험 비교 시 데이터와 fold를 임의로 바꾸지 않는다.

---

# 16. Pilot 실행 순서

## Phase 0 — Repository Baseline

```text
Final Design Baseline / 환경 기록 docs 배치
→ git init -b main
→ ignore 검증
→ dry-run staging
→ baseline commit
```

---

## Phase 1 — 프로젝트 골격

```text
configs/
manifests/
scripts/
src/
tests/
README.md
AGENTS.md
STATUS.md
```

필요한 만큼 생성.

---

## Phase 2 — Full Candidate Inventory

### AI-Hub

```text
local JSON/file scan
→ 18,420 inventory
→ metadata xlsx intersection join
→ viewpoint_3 filter
→ class mapping
→ actor validation
```

### ETRI

```text
Batch B inventory
→ file validation
→ A045~A048 flag
→ target/hard-negative mapping
→ participant validation
```

---

## Phase 3 — Fixed Pilot Manifest

```text
split / fold 먼저 확정
→ split 내부 Pilot selection
→ seed / reason 기록
→ leakage validation
→ manifest freeze
```

---

## Phase 4 — ROI Preflight

```text
AI-Hub + ETRI representative samples
→ MediaPipe ROI
→ success / partial / fallback
→ visual check
→ failure analysis
→ PASS
```

PASS 이전에 전체 Pilot preprocessing 금지.

---

## Phase 5 — Stage A

```text
AI-Hub Pilot
→ preprocessing
→ MobileNetV3-Small fine-tuning
→ frame evaluation
→ video evaluation
→ Encoder B checkpoint
```

Encoder A는 ImageNet-only.

---

## Phase 6 — ETRI Embedding

두 encoder 각각:

```text
ETRI Pilot video
→ T=64 sampling
→ ROI
→ frozen encoder
→ embedding cache
```

---

## Phase 7 — 2×2 Ablation

```text
A: ImageNet + Mean
B: AI-Hub + Mean
C: ImageNet + GRU
D: AI-Hub + GRU
```

각각 participant 5-fold.

---

## Phase 8 — 구조 선택

Primary:

```text
5-fold mean Macro-F1
```

Secondary 기준으로 recall / std / latency / 단순성 확인.

---

## Phase 9 — Pilot deployment/check model

```text
선택 구조
→ ETRI Pilot 전체 valid data
→ Stage B 재학습
→ single deployment/check checkpoint
```

정량 성능 보고에 사용하지 않음.

---

## Phase 10 — Pipeline Integration

```text
ETRI inference script check
→ self_recorded/pipeline_check
→ 기능적 오류 수정
→ 결과 정성 확인
```

Self-recorded 결과에 맞춘 모델 튜닝 금지.

---

# 17. Pilot 완료 기준

다음을 모두 만족하면 Pilot 완료:

- Git 기준선 생성
- Full Candidate Inventory 생성
- Fixed Pilot Manifest 생성
- Manifest leakage/consistency validation PASS
- AI-Hub + ETRI ROI Preflight 완료
- ROI fallback 동작 확인
- Stage A 학습·평가 완료
- ImageNet-only / AI-Hub fine-tuned encoder 준비
- ETRI embedding cache 생성
- Mean Pooling / GRU 모두 구현
- 핵심 2×2 participant-disjoint 5-fold CV 완료
- 5-fold mean ± std / OOF 결과 확보
- 선택 모델 구조 확정
- Pilot deployment/check model 생성
- 실제 End-to-End inference script 실행 가능
- ETRI 전체 pipeline check 완료
- self_recorded/pipeline_check 완료
- Error Analysis 기록
- 성능 한계 / 구현 한계 / Full Experiment 확장 항목 기록
- MLflow에 실험 이력 기록

성능이 낮더라도 pipeline이 정상 완주되고 failure 원인을 분석할 수 있다면 Pilot의 핵심 목적은 달성한 것으로 본다.

---

# 18. 1차 Pilot에서 제외하는 설계

| 항목 | 이유 |
|---|---|
| Full Dataset 전체 preprocessing | Pilot 검증 전 비용 과다 |
| 3D CNN | GPU 부담 |
| Video Transformer | 모델 규모·학습 비용 |
| TSM 등 video end-to-end 구조 | 현재 4GB 환경 부담 |
| Skeleton-only | 복약 vs 기타 구분 한계 |
| Encoder + temporal joint fine-tuning | 2×2 transfer 비교와 cache 전략 단순화 |
| ETRI Batch A | Batch protocol domain 차이 |
| A045~A048 | multi-person association 필요 |
| Threshold gating | 우선 직접 3-class 학습 |
| 광범위 hyperparameter sweep | Pilot 완주 우선 |
| self_recorded 기반 tuning | 평가 오염 방지 |

---

# 19. Pilot에서 실험으로 결정할 사항

| 항목 | 현재 정책 | 우선순위 |
|---|---|---|
| Encoder | ImageNet-only vs AI-Hub fine-tuned | 필수 |
| Stage B | Mean Pooling vs GRU | 필수 |
| T | 64 기본, 96 optional | 선택 |
| T=16 | smoke test | 구현 검증 |
| Sampling | fixed-T uniform 기본 | 기본 |
| Batch size | VRAM에 맞춰 확정 | 구현 시 |
| LR | validation 기반 | 구현 시 |
| Epoch | 10~15 후보 | 구현 시 |
| GRU hidden | 64~128 후보 | 구현 시 |
| Stage A fine-tuning 범위 | 일부 block / head | 구현 시 |
| Augmentation | 최소 합리적 설정 | 구현 시 |
| ROI Preflight pass 기준 | 전처리 가이드에서 확정 | Pilot 전 |
| Weighted CE | Standard CE 이후 | 후속 |
| Focal loss | Standard CE 이후 | 후속 |

Pilot 필수 실험:

1. ImageNet-only + Mean Pooling
2. AI-Hub fine-tuned + Mean Pooling
3. ImageNet-only + GRU
4. AI-Hub fine-tuned + GRU

---

# 20. Pilot 이후 Full Experiment

우선순위:

1. AI-Hub Stage A target / other 사용 범위 확대
2. ETRI Batch B participant별 clip cap 완화
3. 전체 target + 넓은 other pool 활용
4. epoch-wise other sampling 도입
5. hard-negative 비율 실험
6. class-weighted CE / focal loss ablation
7. T / sampling / augmentation 세부 실험
8. 최종 일반화 검증

조건부 후속 실험:

```text
Drink_bever only
VS
Drink_bever + Drink_alcohol
```

AI-Hub auxiliary mapping 효과를 별도로 확인할 수 있다.

추가 고도화 후보:

- ETRI Batch A
- AI-Hub 원본 MP4
- Object-aware feature
- RGB + Skeleton fusion
- 더 강한 temporal model
- 더 강한 visual backbone
- Quantization
- Pruning
- On-device 최적화
- Multi-person association 후 A045~A048 재도입

현재 `self_recorded/pipeline_check` 영상은 이후에도 기본적으로 학습 데이터로 전용하지 않는다.

---

# 부록 A. 확정 사항 요약

## A-1. 데이터

- AI-Hub local JSON/file 18,420이 inventory master
- metadata xlsx는 intersection join용
- AI-Hub Stage A는 viewpoint_3 only
- ETRI Stage B는 Batch B only
- Pilot ratio 약 1:1:2
- 원본 삭제 없이 manifest selection

## A-2. Split

- AI-Hub: actor-disjoint
- ETRI: participant 5-fold Group CV
- 동일 video 3 JPG 동일 split
- 동일 participant 모든 take 동일 fold

## A-3. Model

- MobileNetV3-Small
- ImageNet pretrained
- Stage B에서 encoder frozen
- Mean Pooling + Linear vs GRU + Linear

## A-4. 평가

- Primary: 5-fold mean Macro-F1
- mean ± std
- class Recall
- OOF Confusion Matrix
- 별도 untouched final test를 주장하지 않음

## A-5. Pipeline

- ROI Preflight 필수
- self_recorded는 pipeline check only
- CV 후 전체 Pilot 데이터로 deployment/check Stage B 재학습
- deployment/check model training 성능은 정량 결과로 사용하지 않음

---

# 부록 B. Manifest 필수 후보 column

## B-1. AI-Hub

필수:

- `dataset`
- `root_key`
- `relative_path`
- `frame_relative_paths`
- `video_id`
- `source_split`
- `actor`
- `viewpoint`
- `original_class`
- `target_class`
- `target_role`
- `split`
- `is_hard_negative`
- `valid`
- `exclusion_reason`
- `pilot_selected`
- `pilot_selection_reason`
- `pilot_seed`
- `roi_status`

`target_role` 예:

```text
direct_target
auxiliary_positive
other
```

Drink_bever / Drink_alcohol:

```text
target_class = 음수
target_role  = auxiliary_positive
```

---

## B-2. ETRI

필수:

- `dataset`
- `root_key`
- `relative_path`
- `batch`
- `participant`
- `action`
- `target_class`
- `fold`
- `height`
- `take`
- `is_multi_person`
- `is_hard_negative`
- `valid`
- `exclusion_reason`
- `pilot_selected`
- `pilot_selection_reason`
- `pilot_seed`
- `roi_status`

---

## B-3. 공통 원칙

- absolute path 대신 `root_key + relative_path`
- `pilot_selected=false` candidate도 유지
- selection seed 기록
- selection reason 기록
- exclusion reason 기록
- `roi_status` 초기값 `pending`
- Pilot / Full 전환은 가능하면 config 차이로 제어

---

# 부록 C. Pilot 시작 전 자동 검증 체크리스트

## C-1. Environment

- [ ] `pwd`가 최종 Project Root
- [ ] `.venv/bin/python`
- [ ] Python 3.12.x
- [ ] CUDA available
- [ ] GPU GTX 1650 Ti
- [ ] SSD Raw path readable
- [ ] SSD Work path writable

## C-2. Git

- [ ] `.venv/` ignored
- [ ] `runtime/` ignored
- [ ] `runtime/mlflow/mlflow.db` ignored
- [ ] `configs/paths.local.yaml` ignored
- [ ] checkpoint/cache ignored
- [ ] docs/config/manifest는 추적 가능

## C-3. AI-Hub Manifest

- [ ] local JSON count 검증
- [ ] metadata intersection 검증
- [ ] viewpoint_3 count 검증
- [ ] actor unique 재검증
- [ ] class mapping 검증
- [ ] actor overlap = 0
- [ ] 동일 video 3 JPG same split
- [ ] Pilot sample 목표량 확인

## C-4. ETRI Manifest

- [ ] Batch B count 검증
- [ ] participant list 검증
- [ ] A003 / A004 count 검증
- [ ] invalid RGB flag
- [ ] A045~A048 제외
- [ ] participant fold overlap = 0
- [ ] 동일 participant all takes same fold
- [ ] hard-negative 구성 확인

## C-5. ROI

- [ ] `pending → success/partial/fallback`
- [ ] class별 fallback rate
- [ ] ETRI / AI-Hub 모두 visual check
- [ ] Preflight PASS 전 full Pilot preprocessing 금지

---

# 부록 D. 데이터 역할 요약

| 데이터 | 학습 | ROI Preflight | Pipeline 검증 | End-to-End 정량 평가 | 정성 확인 |
|---|---:|---:|---:|---:|---:|
| AI-Hub viewpoint_3 | Stage A | O | - | Stage A 자체 평가만 | - |
| ETRI Batch B | Stage B | O | O | **participant-disjoint 5-fold CV** | O |
| self_recorded | X | X | O | X | O |

```text
AI-Hub
→ visual encoder 학습

ETRI Batch B
→ temporal classifier 학습
→ participant-disjoint 5-fold End-to-End 평가
→ 전체 inference pipeline 검증

self_recorded
→ 전체 pipeline 기능 검증
→ 결과 정성 확인
→ 학습 / 튜닝 / 정량 평가에는 사용하지 않음
```

---

# 부록 E. Reference 문서

`docs/`에 다음 문서를 함께 유지한다.

- `00_Pilot_Design_Baseline.md` — 본 문서
- `01_개발환경_구축_기록.md`
- AI-Hub 데이터 구조 문서
- AI-Hub EDA 문서
- ETRI 데이터 구조 문서
- ETRI 복약/물마시기/기타 EDA 문서

이 문서를 Git 첫 기준선 commit 시점의 공식 Pilot 설계 기준으로 사용한다.
