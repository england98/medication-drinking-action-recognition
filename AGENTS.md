# AGENTS.md

# 복약-음수 행동 인식 AI 비전 모델 — Coding Agent 작업 지침

- 프로젝트: `medication-drinking-action-recognition`
- 상위 기준 문서: `docs/00_Pilot_Design_Baseline.md`
- 개발환경 기록: `docs/01_개발환경_구축_기록.md`
- 문서 성격: **안정적인 Coding Agent 정책 문서 (필요 시에만 갱신)**
- 최종 정리일: 2026-08-29

---

# 0. 문서 목적

이 문서는 Coding Agent가 본 프로젝트에서 코드를 작성·수정할 때 따라야 하는 작업 규칙을 정의한다.

Coding Agent는 이 문서를 단순 참고사항이 아니라 **작업 수행 규칙**으로 취급해야 한다.

모델·데이터·평가·실험 정책에 관한 최상위 기술 기준은 다음 문서다.

```text
docs/00_Pilot_Design_Baseline.md
```

환경·경로에 관한 현재 기준은 다음 문서다.

```text
docs/01_개발환경_구축_기록.md
```

이 문서와 상위 기준 문서가 충돌하면 다음 우선순위를 따른다.

```text
1. 사용자의 현재 명시적 지시
2. docs/00_Pilot_Design_Baseline.md
3. AGENTS.md
4. STATUS.md        # 현재 작업 상태 / 다음 작업
5. README.md        # 프로젝트의 안정적인 개요
6. 구현 편의상 가정
```

상위 설계를 임의로 변경하지 않는다.

---

# 1. 역할 분담

## 1.1 Coding Agent가 담당하는 작업

Coding Agent는 다음을 수행할 수 있다.

- Python 코드 작성
- 기존 코드 수정
- 코드 리팩터링
- config 작성
- manifest 생성 로직 작성
- validation 코드 작성
- unit test 작성
- smoke test용 코드 작성
- README / STATUS / 개발 문서 보조 작성
- 실행 명령어 작성
- 예상 산출물 및 정상 조건 설명
- 짧고 안전한 정적·동적 검증
- 오류 로그 분석
- 사용자 실행 결과를 바탕으로 후속 수정

## 1.2 사용자가 직접 수행하는 작업

다음 작업은 원칙적으로 사용자가 직접 실행한다.

- 전체 데이터셋 scan
- 대규모 inventory 생성 실행
- 전체 Pilot preprocessing
- ROI Preflight 실제 전체 실행
- Stage A 모델 학습
- ETRI embedding 전체 생성
- Stage B 모델 학습
- participant-disjoint 5-fold CV
- 2×2 Ablation
- 대규모 inference
- End-to-End 정량 평가
- self_recorded pipeline check
- 장시간 GPU 작업
- 대량 파일 생성 작업
- 장시간 CPU / I/O 작업

Coding Agent는 위 작업을 수행하는 코드를 작성하고, 사용자가 직접 실행할 수 있도록 명령어를 제공한다.

---

# 2. Coding Agent 실행 허용 범위

Coding Agent는 다음과 같이 **짧고 안전하며 재현 가능한 검증**은 직접 실행할 수 있다.

예:

- `python -m py_compile`
- import 확인
- config parsing 확인
- 단위 테스트
- 작은 mock 데이터 기반 테스트
- 실제 데이터 1~3개 샘플 수준의 smoke test
- tensor shape 확인
- manifest schema validation의 소규모 테스트
- path existence/readability 확인
- 몇 초~수십 초 내 종료되는 검증

원칙:

```text
빠름
+
데이터 파괴 위험 없음
+
대량 산출물 없음
+
GPU 장시간 사용 없음
```

이면 허용한다.

반대로 다음은 사용자 승인 없이 실행하지 않는다.

- 전체 데이터 전수 scan
- 대규모 전처리
- 수백/수천 개 파일 생성
- 모델 학습
- 5-fold CV
- embedding 전체 생성
- 장시간 inference
- 패키지 대규모 설치/업그레이드
- 환경 재구성

---

# 3. 금지 사항

Coding Agent는 다음을 하지 않는다.

## 3.1 Raw Data 변경 금지

Raw 데이터는 immutable source다.

금지:

- 삭제
- rename
- overwrite
- 원본 파일 내부 수정
- 폴더 구조 변경
- Raw 위치에 파생 결과 저장

Raw 경로:

```text
AI-Hub
/mnt/d/AI 도약과정/데이터/data_raw/057.일상생활_영상_데이터/01.데이터

ETRI
/mnt/d/AI 도약과정/데이터/data_raw/고령자 일상행동인식 3차원 영상 데이터셋(리빙랩)
```

파생 데이터는 Working 영역을 사용한다.

```text
/mnt/d/AI 도약과정/데이터/data_workspace/medication_drinking_action_data
```

## 3.2 환경 임의 변경 금지

사용자 승인 없이 다음을 하지 않는다.

- Python 버전 변경
- `.venv` 삭제/재생성
- 패키지 upgrade/downgrade
- CUDA / PyTorch 재설치
- system package 설치
- MLflow DB 이동
- Project Root 변경

## 3.3 Git 위험 작업 금지

사용자 승인 없이 다음을 하지 않는다.

- commit
- push
- branch 삭제
- reset --hard
- clean -fd
- force push
- history rewrite

Git 관련 작업이 필요한 경우 명령어와 목적을 먼저 제시한다.

## 3.4 설계 임의 변경 금지

다음은 Baseline 정책을 따른다.

- class taxonomy
- AI-Hub / ETRI 역할
- split unit
- actor / participant leakage 방지
- Pilot subset 정책
- ROI Preflight gate
- Stage A / Stage B 구조
- encoder freeze
- 2×2 ablation
- primary metric
- self_recorded 사용 범위

변경 필요성이 보이면 코드를 임의로 바꾸지 말고 다음을 보고한다.

```text
현재 기준
→ 발견된 문제
→ 변경 제안
→ 이유
→ 영향 범위
→ 사용자 승인 필요 여부
```

---

# 4. 경로 사용 규칙

코드에 PC별 절대경로를 직접 하드코딩하지 않는다.

기본 구조:

```text
configs/
├── paths.example.yaml
└── paths.local.yaml
```

- `paths.example.yaml`: Git 관리
- `paths.local.yaml`: 실제 PC 경로, Git 제외

코드에서는 다음 형태를 우선 사용한다.

```text
root_key + relative_path
```

Manifest에도 절대경로 대신 다음 정보를 기록한다.

```text
dataset
root_key
relative_path
```

AI-Hub 3개 JPG는 가능하면:

```text
frame_relative_paths
```

로 저장한다.

---

# 5. 프로젝트 구조 규칙

기준 구조:

```text
<WSL_PROJECT_ROOT>/
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

원칙:

- 필요한 디렉토리는 실제 구현 단위에 맞춰 생성한다.
- 불필요한 빈 하위 디렉토리를 과도하게 만들지 않는다.
- 실행 산출물은 `runtime/` 또는 SSD Working 영역에 둔다.
- Git에 대용량 파일을 넣지 않는다.

---

# 6. Git 관리 규칙

## 6.1 Git 관리 대상

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
- 소형 metadata
- experiment definition

## 6.2 Git 제외 대상

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
*:Zone.Identifier
```

대용량 데이터는 Git에 넣지 않는다.

---

# 7. 데이터 처리 원칙

## 7.1 기본 흐름

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
preprocessing / training
```

원본 candidate를 삭제하지 않는다.

`pilot_selected=false`인 데이터도 inventory에 유지한다.

## 7.2 AI-Hub

기준:

- Local Training + Validation JSON/File 18,420건이 inventory master
- metadata xlsx는 intersection join용
- Stage A는 `viewpoint_3` only
- actor-disjoint split
- 동일 video의 JPG 3장은 같은 split
- class mapping은 Baseline 준수

## 7.3 ETRI

기준:

- Stage B는 Batch B only
- participant-disjoint 5-fold
- 동일 participant의 모든 take는 같은 fold
- A045~A048은 Pilot에서 제외
- invalid sample은 `pilot_selected=false`
- multi-person 관련 정책은 Baseline 준수

---

# 8. Manifest validation 규칙

Manifest validation 실패 시 다음 단계로 넘어가지 않는다.

## 8.1 AI-Hub 필수 검증

```text
train actor ∩ val actor = ∅
train actor ∩ test actor = ∅      # test를 별도로 둘 경우
동일 video의 JPG 3개는 같은 split
pilot_selected sample은 valid=true
```

## 8.2 ETRI 필수 검증

```text
각 fold train participant ∩ val participant = ∅
동일 participant 모든 take는 동일 fold
A045~A048은 pilot_selected=false
invalid sample은 pilot_selected=false
```

검증 실패 시 오류를 숨기거나 자동 보정하지 말고 명시적으로 실패시킨다.

---

# 9. ROI 처리 규칙

ROI Preflight는 필수 gate다.

Preflight 이전에 전체 Pilot 데이터를 일괄 처리하지 않는다.

ROI status:

```text
pending
success
partial
fallback
```

원칙:

- ROI 실패 sample 자동 삭제 금지
- fallback 동작 필수
- class별 fallback rate 기록
- self_recorded는 ROI Preflight에 사용하지 않음
- Preflight PASS 후에만 전체 Pilot preprocessing 진행

---

# 10. 모델 구현 기준

## 10.1 Stage A

Baseline:

```text
MobileNetV3-Small
+
ImageNet pretrained
```

역할:

- AI-Hub frame classifier
- ETRI frame embedding extractor

Stage A 데이터:

```text
AI-Hub viewpoint_3
```

## 10.2 Stage B

입력:

```text
[B, T, D]
```

기본 T:

```text
64
```

후보:

```text
Mean Pooling + Linear
GRU + Linear
```

## 10.3 Encoder Freeze

2×2 Pilot Ablation에서는 ETRI 학습 중 visual encoder를 고정한다.

```text
ImageNet-only Encoder
→ frozen
→ ETRI embedding

AI-Hub Fine-tuned Encoder
→ frozen
→ ETRI embedding
```

ETRI encoder fine-tuning은 1차 Pilot 필수 범위가 아니다.

---

# 11. 2×2 Ablation 기준

필수 실험:

```text
A. ImageNet-only + Mean Pooling
B. AI-Hub fine-tuned + Mean Pooling
C. ImageNet-only + GRU
D. AI-Hub fine-tuned + GRU
```

동일 조건 유지:

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

# 12. 평가 기준

Primary metric:

```text
5-fold mean Macro-F1
```

기록:

- Macro-F1 mean ± std
- 복약 Recall mean ± std
- 음수 Recall mean ± std
- 기타 Recall mean ± std
- fold별 Confusion Matrix
- out-of-fold aggregate Confusion Matrix

성능이 비슷한 경우 secondary 기준:

1. 복약 Recall
2. 음수 Recall
3. std
4. 모델 단순성
5. inference latency / memory

Pilot 결과를 본 뒤 primary metric을 변경하지 않는다.

---

# 13. self_recorded 사용 규칙

경로:

```text
<SSD_WORK_ROOT>/self_recorded/pipeline_check/
```

허용:

- 전체 inference pipeline 동작 확인
- label / confidence 생성 확인
- 기능적 코드 오류 확인
- 정성적 확인

금지:

- 학습
- validation
- test metric
- ROI 방식 선정
- sampling 방식 결정
- threshold tuning
- hyperparameter tuning
- model selection
- preprocessing tuning
- self_recorded 결과를 보고 fine-tuning

---

# 14. MLflow / 재현성 규칙

각 실험 run에는 가능하면 다음을 기록한다.

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

실험 비교 중 데이터와 fold를 임의로 바꾸지 않는다.

---

# 15. 작업 시작 절차

Coding Agent는 새 작업을 시작할 때 다음 순서를 따른다.

```text
1. 현재 요청 확인
2. docs/00_Pilot_Design_Baseline.md 관련 항목 확인
3. STATUS.md 현재 Phase 확인
4. 필요한 기존 코드/문서 확인
5. 변경 범위 최소화
6. 코드 작성
7. 짧은 검증
8. 변경 내용 보고
9. 사용자 실행 명령 제공
10. 예상 산출물 / PASS 기준 제공
```

---

# 16. 작업 완료 보고 형식

각 작업 완료 후 최소 다음을 보고한다.

## 변경 파일

```text
생성:
- ...

수정:
- ...

삭제:
- ...
```

## 구현 내용

```text
- ...
- ...
```

## Coding Agent가 직접 수행한 검증

```text
- command
- result
```

## 사용자가 실행할 명령

```bash
...
```

## 예상 산출물

```text
...
```

## PASS 기준

```text
...
```

## Living Document 갱신

```text
STATUS.md: 수정 / 수정 불필요
README.md: 수정 / 수정 불필요
AGENTS.md: 사용자 승인 후 수정 / 수정 불필요
```

필요한 경우 각 판단 이유를 간단히 기록한다.

## 남은 이슈 / 주의사항

```text
...
```

---

# 17. 문서 관리 규칙

문서는 역할을 명확히 분리한다.

| 문서 | 역할 | 기본 갱신 원칙 |
|---|---|---|
| `AGENTS.md` | Coding Agent 작업 규칙 | 공통 규칙이 바뀔 때만 |
| `README.md` | 프로젝트의 안정적인 개요 | 구조·사용법·주요 인터페이스가 바뀔 때만 |
| `STATUS.md` | 현재 작업 진행 상태의 **Single Source of Truth** | 실제 작업 상태가 바뀔 때 |
| `docs/00_Pilot_Design_Baseline.md` | Pilot 상위 설계 기준 | 의미 있는 설계 변경 시 새 버전/변경 문서 우선 |
| EDA / 데이터 구조 / 환경 기록 | 기준 시점의 Reference | 원칙적으로 보존 |

일상적인 진행 상황은 `README.md`나 `AGENTS.md`에 중복 기록하지 않고 `STATUS.md`에서만 관리한다.

확정된 기준 문서는 해당 시점의 기록으로 보존한다.

- Design Baseline / EDA / 데이터 구조 / 환경 기준 문서는 기준 시점 기록
- 의미 있는 설계 변경은 기존 문서를 덮어쓰기보다 새 변경 문서 또는 새 버전으로 기록
- 단순 오타·링크·표현 정리는 기존 문서 수정 가능
- 실험 결과는 별도 결과 문서와 MLflow에 기록

설계 변경 문서를 만들 경우 가능하면 다음을 기록한다.

```text
변경 전
변경 후
이유
영향 범위
적용 시점
```


## 17.1 Living Document 갱신 규칙

Coding Agent는 작업 완료 후 실제 프로젝트 상태와 문서 상태가 불일치하지 않도록
필요한 Living Document를 함께 확인하고 갱신한다.

### STATUS.md

`STATUS.md`는 현재 작업 상태의 **Single Source of Truth**다.

다음 경우에는 반드시 갱신한다.

- Phase 또는 작업 단계가 변경된 경우
- 체크리스트 항목이 완료된 경우
- 중요한 구현 작업이 완료된 경우
- 새로운 blocker / issue가 발생한 경우
- 다음 작업 또는 우선순위가 변경된 경우

작업 완료 후 최소 다음 항목이 실제 상태와 일치하는지 확인한다.

```text
현재 상태
완료
현재 해야 할 작업
다음 Phase / 다음 작업
```

단순한 코드 정리나 상태 변화가 없는 소규모 수정만 수행한 경우에는
불필요하게 STATUS.md를 수정하지 않는다.

### README.md

`README.md`에는 현재 Phase, 오늘 할 일, 최근 완료 작업 같은 상세 진행 상태를 중복 기록하지 않는다.
상세 진행 상태는 `STATUS.md`로 연결한다.

다음 경우 필요한 부분만 갱신한다.

- 프로젝트 실행 방법이 추가되거나 변경된 경우
- 주요 디렉토리 구조가 변경된 경우
- 새로운 주요 기능 또는 pipeline이 추가된 경우
- 사용자 또는 개발자가 알아야 할 명령어가 변경된 경우
- 프로젝트 사용 방법에 영향을 주는 변경이 발생한 경우

내부 구현 세부사항이나 사용자 사용 방식에 영향을 주지 않는 변경만 있는 경우에는
README.md를 수정하지 않아도 된다.

### AGENTS.md

`AGENTS.md`는 작업 중 매번 갱신하는 문서가 아니다.

다음 경우에만 변경을 검토한다.

- Coding Agent의 작업 권한이 변경된 경우
- 사용자와 Coding Agent의 역할 분담이 변경된 경우
- 공통 작업 절차가 변경된 경우
- 금지사항 또는 승인 절차가 변경된 경우
- 프로젝트 전반에 적용되는 코딩·검증·실행 규칙이 변경된 경우

**AGENTS.md의 작업 권한, 금지사항, 역할 분담, 승인 절차를 변경하는 수정은
사용자의 명시적 승인 없이 수행하지 않는다.**

일반적인 기능 구현, 버그 수정, 실험 코드 작성만으로는 AGENTS.md를 수정하지 않는다.

### 공통 원칙

- 문서를 갱신하기 위한 갱신은 하지 않는다.
- 실제 프로젝트 상태가 변경된 경우에만 필요한 문서를 수정한다.
- 확정된 Design Baseline / EDA / 데이터 구조 / 개발환경 기록 문서는 임의로 수정하지 않는다.
- 의미 있는 설계 변경은 기존 기준 문서를 덮어쓰기보다 새 변경 문서 또는 새 버전으로 기록한다.
- 작업 완료 보고에 어떤 Living Document를 수정했는지 또는 수정하지 않은 이유를 명시한다.

---

# 18. 현재 작업 상태 참조

현재 Phase, 완료 작업, 진행 중 작업, 다음 작업, blocker는 이 문서에 기록하지 않는다.

Coding Agent는 작업 시작 시 반드시 다음 문서를 확인한다.

```text
STATUS.md
```

현재 작업 상태가 `README.md` 또는 과거 문서와 다를 경우,
상위 설계 기준에 위배되지 않는 범위에서 `STATUS.md`를 현재 상태 기준으로 사용한다.
