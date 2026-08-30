# STATUS.md

# Project Status

- 프로젝트: `medication-drinking-action-recognition`
- 현재 Phase: **Phase 8 — COMPLETE**
- 이전 Phase: **Phase 7 — COMPLETE**
- 다음 Phase: **Phase 9 — READY TO START**
- 최종 업데이트: 2026-08-30
- 문서 성격: **현재 작업 상태의 Single Source of Truth**

---

## 1. 현재 상태

개발환경, Data Readiness, Git Repository Baseline 구축이 완료되었다.
Phase 1 프로젝트 최소 골격과 안전한 path infrastructure 구축을 완료했다.
Phase 2의 AI-Hub Full Candidate Inventory 생성·validation 및 전체 scan을 완료했다.
AI-Hub Inventory 결과는 **PASS_WITH_WARNINGS**다.
ETRI Batch B Full Candidate Inventory 생성·validation 코드와 소수 MP4 smoke test를 완료했으며,
전체 scan 결과는 **PASS_WITH_WARNINGS**다. Phase 2 inventory를 입력으로 actor-disjoint split과
participant-disjoint 5-fold를 먼저 고정한 뒤 Pilot subset을 선택했다. Fixed Pilot Manifest 생성,
leakage/consistency validation 및 결정성 검증을 완료했다. Phase 4의 deterministic representative
selector, MediaPipe ROI, AI-Hub/ETRI loader, visual output, report 및 CLI 구현과 mock validation을
완료했다. WSL의 `libGLESv2.so.2` prerequisite 해결 후 실제 AI-Hub 1-frame smoke도 통과했다.
hand-only partial의 작은 crop 문제를 원본 크기 기반 contextual ROI로 보완하고 동일 sample 재검증을
완료했다. 사용자 전체 64-frame 실행에서 ETRI fallback 68.75%와 class 편차가 확인되어 PASS하지
않았다. no-face/no-hand 경로에 Pose contextual ROI를 추가하고 ETRI fallback 3-frame smoke를
통과했다. 이후 사용자가 동일한 64-frame representative set을 재실행하고 visual review를 완료했다.
최종 결과는 success 4, partial 60, fallback 0이며 ROI는 얼굴·상체·손과 행동 context를 보존했다.
Phase 4 최종 판정은 **PASS_WITH_WARNINGS**다. Phase 5 Stage A 구현, GPU smoke, 본 학습,
frame/video 평가, best checkpoint reload, Encoder A/B interface 검증 및 독립 Final Audit를
완료했다. 최종 판정은 **PHASE 5 COMPLETE — READY FOR PHASE 6**이다. Phase 6에서는
Fixed ETRI Pilot 239개 전체에 대한 frozen Encoder A/B embedding cache 생성, full-cache
validation gate 및 독립 Final Audit를 완료했다. 최종 상태는 **Phase 6 COMPLETE — Phase 7
READY TO START**이다. Phase 7의 immutable cache dataset, Mean/GRU Stage B, 2×2 experiment
definition, participant-disjoint fold guard, 공통 trainer, metric/OOF/aggregation/checkpoint/MLflow 및
CLI 구현을 완료하고 실제 239-cache validation-only를 통과했다. 독립 Pre-run Audit 결과는
**PASS_WITH_WARNINGS — READY FOR SMOKE AFTER NOTED CHECKS**였고, 이후 Mean/GRU/MLflow smoke와
최종 regression을 모두 통과했다. commit `60c72bc125b0cc2fe94bd9500858192e3edaf521`의 clean
working tree에서 A/B/C/D × 5-fold 20개 production run, OOF/aggregation 및 Post-run Independent
Final Audit를 완료했다. Phase 8에서 fold/OOF evidence completeness, participant leakage,
experiment fairness와 metric aggregation을 독립 검증하고 고정된 primary metric으로 Experiment D
(AI-Hub fine-tuned Encoder + GRU)를 선택했다. 최종 상태는 **Phase 8 COMPLETE / Phase 9 READY
TO START**이며 Phase 9 구현·학습은 아직 시작하지 않았다.

현재 위치:

```text
개발환경 구축
↓
개발환경 Audit
↓
Repository 정리
↓
Git baseline 검증 / commit
↓
Phase 1 프로젝트 골격       완료
↓
Full Candidate Inventory    완료
↓
Fixed Pilot Manifest        완료
↓
ROI Preflight               완료 (PASS_WITH_WARNINGS)
↓
Stage A Visual Encoder      완료
↓
ETRI Embedding Cache        완료 (PASS_WITH_WARNINGS)
↓
2×2 Ablation                완료 (PASS_WITH_WARNINGS)
↓
Structure Selection         완료
↓
Pilot deployment/check model ← 다음 (READY TO START)
```

---

## 2. 완료

### 데이터 / 분석

- [x] AI-Hub 데이터 확보
- [x] ETRI 데이터 확보
- [x] AI-Hub 데이터 구조 분석
- [x] ETRI 데이터 구조 분석
- [x] AI-Hub 복약 / 음수 / 기타 관점 EDA
- [x] ETRI 복약 / 물마시기 / 기타 관점 EDA
- [x] 모델 아키텍처 리서치
- [x] 1차 Pilot 모델 아키텍처 설계

### 개발환경

- [x] WSL 2 / Ubuntu 개발환경 구축
- [x] 최종 WSL Project Root 확정
- [x] Python 3.12 `.venv` 재구축
- [x] VS Code Python Interpreter 연결
- [x] PyTorch / Torchvision 설치 확인
- [x] CUDA 사용 가능 확인
- [x] GTX 1650 Ti GPU 인식 확인
- [x] CUDA tensor 연산 확인
- [x] MediaPipe / OpenCV / MLflow 등 주요 패키지 확인
- [x] `.python-version` 생성
- [x] `requirements-lock.txt` 생성
- [x] MLflow DB를 `runtime/mlflow/mlflow.db`로 정리
- [x] Raw / Working SSD 경로 확정
- [x] GPU / CUDA 현재 상태 재검증

### 문서 / Repository 정리

- [x] `docs/00_Pilot_Design_Baseline.md` 배치
- [x] `docs/01_개발환경_구축_기록.md` 배치
- [x] AI-Hub 구조 / EDA reference 배치
- [x] ETRI 구조 / EDA reference 배치
- [x] 개발환경 read-only Audit 완료
- [x] `configs/paths.example.yaml` 작성
- [x] `configs/paths.local.yaml` 실제 PC 경로 설정 및 검증
- [x] `configs/` 디렉토리 준비
- [x] `docs/` 소유권 정리
- [x] `*:Zone.Identifier` 잔재 파일 정리
- [x] `.gitignore`에 `*:Zone.Identifier` 규칙 반영
- [x] `AGENTS.md` 역할 정리
- [x] `README.md` 역할 정리
- [x] `STATUS.md` 역할 정리
- [x] Data Readiness Audit 완료

### Git Repository Baseline

- [x] Git repository 초기화
- [x] branch `main` 확인
- [x] `git status` 확인
- [x] `.gitignore` 실제 동작 검증
- [x] `.venv/` ignore 확인
- [x] `runtime/` / `runtime/mlflow/mlflow.db` ignore 확인
- [x] `configs/paths.local.yaml` ignore 확인
- [x] `git add -n .` staging dry-run 검증
- [x] 대용량 데이터 / runtime 산출물 staging 제외 확인
- [x] baseline commit 완료: `d7a6101 chore: establish pilot project baseline`
- [x] Phase 0 — Repository Baseline 완료

---

## 3. 최근 검증 결과

개발환경 Audit 결과:

```text
FAIL = 0
환경 기준 = PASS
```

확인된 항목:

- Project Root 일치
- WSL / Ubuntu 버전 일치
- `.venv` / Interpreter 일치
- Python 3.12.14 일치
- 주요 패키지 버전 일치
- CUDA available = True
- GPU = NVIDIA GeForce GTX 1650 Ti
- CUDA tensor 연산 성공
- `.python-version` 정상
- `requirements-lock.txt`와 현재 환경 일치
- AI-Hub Raw read 가능
- ETRI Raw read 가능
- Working Data read/write 가능
- MLflow DB 경로 일치
- 기준 문서 존재

이전 Audit에서 지적된 Repository 정리 항목은 현재 문서 기록 기준으로 모두 처리된 상태다.

Data Readiness Audit 결과:

```text
AI-Hub: READY WITH WARNING
ETRI: READY WITH WARNING
Working: READY
전체: DATA READY WITH WARNINGS
```

확인된 후속 검증 항목:

- AI-Hub 18,420 JSON / 55,260 JPG 전체 count는 Phase 2 Full Candidate Inventory에서 최종 검증한다.
- ETRI `P205/A053_P205_G011_H120.mp4` 비정상 RGB는 Raw를 변경하지 않고 Phase 2 inventory validation에서 invalid / exclusion flag로 처리한다.
- Working Root는 read 및 소형 임시 파일 write/delete test를 통과했다.
- AI-Hub Raw / ETRI Raw / Working Root는 canonical path 기준으로 서로 분리되며 symlink가 아니다.

Git Repository Baseline 검증 결과:

```text
repository: initialized
branch: main
ignore rules: PASS
staging dry-run: PASS
baseline commit: d7a6101 chore: establish pilot project baseline
Phase 0: COMPLETE
```

Phase 5 Stage A 최종 결과:

```text
Final Audit: PHASE 5 COMPLETE — READY FOR PHASE 6
GPU smoke: PASS
Batch size: 8
Epochs: 15
Fine-tuning: last_n_blocks (last_n_blocks=2)
Frozen BatchNorm: freeze_running_stats
Run ID: stage-a-20260829T170113531051Z-adf86795
Best epoch: 12
Best validation frame Macro-F1: 0.536788810523163
Best validation video Macro-F1: 0.652049
Video Recall: 복약 0.55 / 음수 0.60 / 기타 0.80
Embedding dimension: 1024
Encoder A/B interface: PASS ([B,1024] / [B,1024])
Best checkpoint reload: PASS
Reload evaluation reproducibility: PASS (동일 조건 2회 완전 동일)
Checkpoint provenance: PASS
Phase 3/4 frozen artifact: PASS
```

Video confusion matrix:

```text
[[11, 6, 3],
 [ 2,12, 6],
 [ 4, 4,32]]
```

Encoder B checkpoint:

```text
<work_root>/checkpoints/stage_a/stage-a-20260829T170113531051Z-adf86795/best.pt
```

Git / MLflow provenance:

```text
git_commit_hash: 7c3f1c34f1def0dfbf0800557f1e596321506be4
git_dirty: false
MLflow experiment: stage_a_visual_encoder
MLflow run_id: ffb970c506c843e89b6613788c61a2c9
MLflow per-epoch / best / final metrics 및 artifacts: PASS
```

Phase 6 Embedding Preflight 결과:

```text
Independent audit: PASS_WITH_WARNINGS
Full Cache Generation: READY_WITH_WARNINGS
BLOCKER: 0 / MAJOR: 0
A/B fairness: PASS
1-clip GPU smoke: PASS
source_clip_key: P201:A003:G006:H070
participant/fold/class: P201 / 3 / 복약
original/sample frames: 136 / 64
ROI: success 0 / partial 64 / fallback 0
Encoder A/B: [64,1024] / [64,1024]
NaN/Inf: 0
A/B frozen: true
shared indices: PASS
cache reload: PASS
selected valid ETRI clips: 239
manifest SHA-256: 0c641e3301196afa92c4cf7b7cad28dfd9e21c5f88a5ebbe02b694110b3b4b93
```

Phase 6 ETRI Embedding Cache 최종 결과:

```text
Full run: scripts.run_etri_embedding_cache --resume
status: PASS
validation_mode: full_239
ETRI selected valid: 239
cache: success 239 / failed 0 / created 227 / resumed 12
T: 64
D: 1024
sampling: fixed_uniform
inference_batch_size: 8
Encoder A: torchvision MobileNet_V3_Small_Weights.DEFAULT / ImageNet-only
Encoder B: stage-a-20260829T170113531051Z-adf86795/best.pt / best epoch 12
Encoder A/B shape: [64,1024] / [64,1024]
Encoder A/B NaN/Inf: 0
A/B clip-key parity: PASS
A/B frame-index parity: PASS
Manifest/cache key-set parity: PASS
ROI: success 0 / partial 15178 / fallback 118 / total 15296 = 239 × 64
class: 기타 120 / 복약 59 / 음수 60
fold: 48 / 48 / 48 / 48 / 47
manifest SHA-256: 0c641e3301196afa92c4cf7b7cad28dfd9e21c5f88a5ebbe02b694110b3b4b93
ROI config SHA-256: 7b692e3fa0260f745664869a58a958cf276d428cf693abda4449fcbadc71e381
Stage A config SHA-256: 0987566c7175a408f4d72504e77c16ec835eba722a52a98a03fe62e204e4e54f
Independent Final Audit: PASS_WITH_WARNINGS
Phase 3/4/5 frozen artifact: unchanged
```

Phase 7 최종 결과:

```text
Phase 7: COMPLETE
Phase 7 implementation: COMPLETE
Full CV: A/B/C/D × participant-disjoint 5-fold
Production runs: 20/20 FINISHED / failed 0 / duplicate production run name 0
Implementation commit: 60c72bc125b0cc2fe94bd9500858192e3edaf521
Git provenance: 20/20 동일 commit / git_dirty=false / git_diff_sha256=None
Manifest SHA-256: 0c641e3301196afa92c4cf7b7cad28dfd9e21c5f88a5ebbe02b694110b3b4b93
Selected valid clips: 239
Fold validation clips: 48 / 48 / 48 / 48 / 47
Participants: 30 / train 24 per fold / validation 6 per fold
Participant leakage: 0
Clip leakage: 0
Class: 복약 59 / 음수 60 / 기타 120
T: 64
D: 1024
OOF: A/B/C/D 각각 239 rows / exact-once PASS / duplicate 0 / missing 0
A/B/C/D/Manifest clip-key set parity: PASS
MLflow experiment: phase7_etri_2x2_ablation
MLflow production runs: 20 / FINISHED 20 / failed 0
Metric/aggregation independent recomputation: EXACT MATCH
Maximum deviation: 0.00e+00
Standard deviation: population ddof=0
Post-run Independent Final Audit: PASS_WITH_WARNINGS
Audit findings: BLOCKER 0 / MAJOR 0 / MINOR 1 / INFO 3
Phase 8: COMPLETE
Selected structure: Experiment D / AI-Hub fine-tuned Encoder / GRU + Linear
Selection artifact: configs/phase8_selected_model.yaml
Result document: docs/05_Phase8_Structure_Selection_Result.md
Phase 9: READY TO START (implementation/training NOT STARTED)
```

Phase 7 정식 5-fold 결과:

| Exp | Encoder | Stage B | Macro-F1 mean ± std | 복약 Recall | 음수 Recall | 기타 Recall |
|---|---|---|---:|---:|---:|---:|
| A | ImageNet-only | Mean | 0.4507 ± 0.0794 | 0.3394 | 0.2667 | 0.7833 |
| B | AI-Hub fine-tuned | Mean | 0.4491 ± 0.0874 | 0.2212 | 0.3000 | 0.8667 |
| C | ImageNet-only | GRU | 0.5190 ± 0.0420 | 0.3742 | 0.3000 | 0.8833 |
| D | AI-Hub fine-tuned | GRU | 0.5322 ± 0.0546 | 0.3576 | 0.3333 | 0.9000 |

위 표의 std는 population standard deviation(`ddof=0`)이다. Phase 8 독립 재계산은 exact match였고,
primary metric 5-fold mean Macro-F1 기준 Experiment D를 선택했다. exact tie가 없어 secondary
tie-break는 사용하지 않았다.

---

## 4. 현재 해야 할 작업

Phase 1 완료 항목:

- [x] `configs/`, `manifests/`, `scripts/`, `src/`, `tests/` 최소 골격 확정
- [x] `configs/paths.local.yaml` 공통 loader 및 필수 root 검증 구현
- [x] canonical path 기준 Raw input / Working output 양방향 분리 안전장치 구현
- [x] YAML / 필수 key / root 존재 / 경로 분리 unit test 구현 및 통과

Phase 2 AI-Hub 구현 완료 항목:

- [x] Phase 2 AI-Hub Full Candidate Inventory 구현 범위 확인
- [x] AI-Hub inventory 생성 및 validation 코드 작성
- [x] metadata xlsx intersection join 구현
- [x] 사용자 실행용 전체 scan 명령과 PASS 기준 정의
- [x] mock unit test 및 실제 AI-Hub 3 JSON smoke test 통과
- [x] AI-Hub Full Candidate Inventory 전체 scan 완료
- [x] 전체 count 및 metadata intersection join PASS
- [x] Raw JSON annotation 이상 1건 탐지 및 exclusion 확정
- [x] AI-Hub Inventory 결과 `PASS_WITH_WARNINGS`
- [x] 사용 가능 AI-Hub candidate 18,419건 확정
- [x] Phase 3용 `pilot_selected=true && valid=false` validation FAIL 정책 및 공통 검사 함수 정의

Phase 2 ETRI 구현 완료 항목:

- [x] Batch B `P201~P230/RGB Videos` inventory scan 코드 작성
- [x] ETRI filename / participant / action / take / height validation 구현
- [x] A003 / A004 class mapping 및 target participant coverage 검증 구현
- [x] ETRI hard-negative 13개 행동 mapping 구현
- [x] A045~A048 multi-person flag 및 Pilot exclusion 초기 상태 구현
- [x] 소형·비정상 RGB validation 및 3단계 상태 정책 구현
- [x] mock unit test 및 실제 ETRI 3 MP4 metadata smoke test 통과
- [x] 사용자 실행용 Full Inventory 명령과 PASS 기준 정의
- [x] ETRI Batch B Full Candidate Inventory 전체 scan 완료
- [x] 전체 6,589 clips / 30 participants / A003·A004 count 및 coverage PASS
- [x] 비정상 RGB 1건을 `valid=false`로 격리하고 Raw 보존
- [x] ETRI Inventory 결과 `PASS_WITH_WARNINGS`
- [x] 사용 가능 ETRI candidate 6,588건 확정

Phase 3 Fixed Pilot Manifest 완료 항목:

- [x] selection/split/fold config 및 seed 외부화
- [x] AI-Hub 202 actor의 train 162 / val 40 actor-disjoint split 확정
- [x] AI-Hub Pilot 400 video / 1,200 frame 선택 (복약 100, 음수 100, 기타 200)
- [x] AI-Hub Eat_food hard negative 50 video 포함
- [x] ETRI 30 participant의 participant-disjoint 5-fold 확정 (fold별 6명)
- [x] ETRI Pilot 239 clip 선택 (복약 59, 음수 60, 기타 120)
- [x] ETRI hard negative 60 / general other 60 구성
- [x] invalid 및 A045~A048 선택 제외 validation
- [x] 전체 candidate row 보존 및 `roi_status=pending` 유지
- [x] Fixed Pilot Manifest SHA-256 summary 생성
- [x] synthetic unit test 및 실제 inventory manifest 생성/validation 통과

현재 해야 할 작업:

- [x] Phase 4 ROI Preflight 대표 sample/config 범위 확정
- [x] MediaPipe face/hand landmark ROI 및 full-frame fallback 구현
- [x] AI-Hub image / ETRI deterministic frame loader 구현
- [x] overlay/crop/preview 및 JSON/CSV report 구현
- [x] synthetic/mock unit validation 완료
- [x] 공식 MediaPipe face/hand task 모델 Working cache 준비
- [x] WSL 시스템 prerequisite `libgles2` 사용자 설치 및 MediaPipe 초기화 확인
- [x] 실제 AI-Hub 1-frame smoke 및 hand-only contextual ROI 재검증
- [x] 사용자 최초 64-frame Preflight 실행 및 ETRI 고 fallback 문제 확인 (PASS 보류)
- [x] no-face/no-hand용 Pose contextual ROI 및 pose-only partial 구현
- [x] 실제 ETRI 기존 fallback 3-frame Pose smoke 통과
- [x] 사용자가 Pose 보완 정책으로 동일 64-frame representative set 재실행
- [x] dataset/class별 fallback rate 및 visual output 검토
- [x] 최종 64-frame 결과: success 4 / partial 60 / fallback 0
- [x] AI-Hub: success 4 / partial 12 / fallback 0
- [x] ETRI: success 0 / partial 48 / fallback 0
- [x] reason: face_and_hand_landmarks 4 / hand_only 26 / pose_only 34
- [x] 사용자 visual review 완료 및 Phase 4 `PASS_WITH_WARNINGS` 판정
- [x] Pilot 동안 ROI policy 고정

현재 해야 할 작업:

- [x] Phase 5 — Stage A Dataset / ROI preprocessing / MobileNetV3-Small 구현
- [x] Stage A frame/video 평가, checkpoint provenance, MLflow logging 경로 구현
- [x] Stage A unit/integration 자동 테스트 구현
- [x] Phase 5 read-only audit 및 본 학습 전 필수 code fix 반영
- [x] 수정 후 final implementation audit
- [x] 사용자 실제 GPU smoke test PASS 및 batch size 8 확정
- [x] Stage A 15 epoch 본 학습 및 frame/video evaluation 완료
- [x] Encoder B best checkpoint 생성/reload 및 Encoder A/B interface 검증 PASS
- [x] Phase 5 독립 Final Audit 완료

---

## 5. Phase 0 완료 결과

다음 기준을 모두 만족하여 Phase 0을 완료했다.

- [x] Git repository 초기화
- [x] `.gitignore` 검증 PASS
- [x] `.venv/` / `runtime/` / local path config ignore 확인
- [x] staging dry-run PASS
- [x] 대용량 데이터 / runtime 산출물 staging 없음
- [x] baseline commit 완료

---

## 6. 다음 작업

다음 작업:

```text
Phase 9 — Pilot deployment/check model
```

Phase 7 최종 상태:

```text
COMPLETE — Full 20-run / OOF aggregation / Post-run Independent Final Audit 완료
```

Phase 6 현재 작업:

- [x] T=64 fixed-uniform sampling 및 shared-frame contract 구현
- [x] Phase 4 ROI 재사용과 frozen Encoder A/B embedding contract 구현
- [x] temporary smoke cache/provenance/save-reload validation 구현
- [x] Phase 6 unit test 구현
- [x] 사용자 실제 ETRI Pilot 1-clip Encoder A/B smoke test PASS
- [x] 독립 audit `PASS_WITH_WARNINGS` 및 full-cache readiness `READY_WITH_WARNINGS`
- [x] audit minor finding provenance/frame-count/tests 보완
- [x] full-cache entrypoint/resume/summary/validation gate 구현
- [x] 사용자 multi-clip preflight 실행 및 결과 검토
- [x] 사용자 전체 ETRI Pilot 239개 embedding cache 생성
- [x] full-cache validation gate PASS
- [x] Phase 6 independent Final Audit 완료
- [x] Final audit findings P-1~P-4 closed

Phase 7 현재 작업:

- [x] immutable Phase 6 cache dataset/validation 구현
- [x] Mean Pooling + Linear 및 GRU + Linear 구현
- [x] A/B/C/D canonical experiment/fairness validation 구현
- [x] participant-disjoint 5-fold runtime guard 구현
- [x] 공통 training/validation, best checkpoint 및 reload 구현
- [x] fold metric, OOF prediction/completeness, 5-fold aggregation 구현
- [x] MLflow provenance 및 output overwrite safety 구현
- [x] validation-only / fold / experiment / all-experiment / smoke CLI 구현
- [x] synthetic Phase 7 tests 및 실제 239-cache validation-only PASS
- [x] Phase 7 pre-run independent audit (`PASS_WITH_WARNINGS`)
- [x] Mean smoke 완료 (A / fold 0 / 1 epoch / no MLflow)
- [x] GRU smoke 완료 (C / fold 0 / 1 epoch / no MLflow)
- [x] MLflow smoke 완료 (A / fold 1 / 1 epoch)
- [x] smoke 이후 final regression 85 tests / diff check / validation-only PASS
- [x] Phase 7 implementation commit (`60c72bc125b0cc2fe94bd9500858192e3edaf521`)
- [x] A/B/C/D × 5-fold 총 20 production run
- [x] MLflow production run 20/20 FINISHED
- [x] OOF exact-once / A/B/C/D 각각 239 rows 검증
- [x] 실제 OOF / 5-fold aggregation 독립 재계산 exact match
- [x] Post-run Independent Final Audit (`PASS_WITH_WARNINGS`)
- [x] Phase 7 COMPLETE

Phase 8 현재 작업:

- [x] A/B/C/D 각각 intended 5-fold evidence completeness 검증
- [x] participant-disjoint 및 OOF 239개 exact-once 검증
- [x] fold metrics/OOF 기반 aggregation 독립 재계산 및 Phase 7 exact match
- [x] manifest/fold/T/D/ROI/sampling/normalization/training provenance fairness 검증
- [x] 고정 primary metric 기준 deterministic ranking 및 Experiment D 선택
- [x] 2×2 effect analysis 및 human-readable 결과 문서 생성
- [x] machine-readable Phase 9 handoff artifact 생성
- [x] Phase 8 tests 및 전체 regression PASS
- [x] Phase 8 COMPLETE
- [ ] Phase 9 구현/학습 (NOT STARTED)

---

## 7. Pilot 전체 진행 순서

```text
Phase 0  Repository Baseline          완료
↓
Phase 1  프로젝트 골격            완료
↓
Phase 2  Full Candidate Inventory  완료
↓
Phase 3  Fixed Pilot Manifest       완료
↓
Phase 4  ROI Preflight              완료 (PASS_WITH_WARNINGS)
↓
Phase 5  Stage A                    완료
↓
Phase 6  ETRI Embedding             완료 (PASS_WITH_WARNINGS)
↓
Phase 7  2×2 Ablation               완료 (PASS_WITH_WARNINGS)
↓
Phase 8  구조 선택                  완료
↓
Phase 9  Pilot deployment/check model  다음 (READY TO START)
↓
Phase 10 Pipeline Integration
```

---

## 8. Blocker / Issue

현재 blocker는 없다. Phase 4 warning은 다음과 같다.

```text
1. ETRI 48 frame은 face+hand success 없이 모두 partial이다.
2. ETRI의 hand_only / pose_only reason 분포는 class마다 다를 수 있다.
3. pose_only ROI는 비교적 넓은 contextual crop이다.
4. Stage B / End-to-End Error Analysis에서 ROI reason별 성능 편차를 확인한다.
5. 현재 Pilot에서는 ROI policy를 고정하고 Stage A/B 결과를 보고 즉시 재튜닝하지 않는다.
6. 필요하면 Pilot 완료 후 Full Experiment에서 ROI scale/policy ablation을 검토한다.
```

Phase 5 warning / limitation:

```text
1. Train loss는 지속 감소했지만 validation loss는 후반 증가하여 overfitting 신호가 확인됐다.
   Pilot 목적은 최고 성능 확보가 아니라 Stage A Visual Encoder 생성과 전체 pipeline 검증이다.
   Validation video Macro-F1 기반 best checkpoint 정책에 따라 epoch 12 Encoder B를 고정한다.
   이 결과를 이유로 Stage A를 재튜닝하거나 재학습하지 않는다.
   Encoder A/B의 실질적 가치는 이후 ETRI Stage B, 2×2 ablation,
   participant-disjoint 5-fold CV에서 평가한다.
2. Stage A 음수는 AI-Hub Drink_bever / Drink_alcohol 기반 auxiliary visual class이며,
   최종 ETRI 물 마시기 성능이 아니다.
3. CUDA strict deterministic 미설정은 비차단 limitation이다.
4. Encoder A는 별도 checkpoint 없이 torchvision ImageNet pretrained
   MobileNetV3-Small로 재구성하는 설계를 유지한다.
```

Phase 6 warning / limitation:

```text
1. ETRI Phase 6 전체 15,296 sampled frame에서 ROI는 success 0, partial 15,178,
   fallback 118이었다. fallback rate는 약 0.77%다.
2. 이는 Phase 4 frozen ROI policy에서 허용된 fallback이며 sample/frame은 삭제되지 않았다.
3. Encoder A/B 모두 동일한 ROI 결과를 사용하므로 ImageNet-vs-AI-Hub 비교 조건을 오염시키지 않는다.
4. Independent Final Audit 판정은 PASS_WITH_WARNINGS이다.
5. Final audit findings P-1~P-4는 deterministic cache naming, full-run gate,
   resume rejection, output-root/manifest mutation 회귀 테스트로 모두 closed 상태다.
```

Phase 7 warning / limitation:

```text
1. Post-run Independent Final Audit 판정은 PASS_WITH_WARNINGS이며 BLOCKER 0 / MAJOR 0이다.
2. 유일한 MINOR PF-1은 STATUS/README가 pre-run 상태였던 문서 sync 문제이며 이번 갱신으로 해소했다.
3. fold aggregate의 std는 population standard deviation(ddof=0)이다.
4. CUDA strict deterministic algorithms는 활성화하지 않았다. Phase 5/6과 동일한 비차단 정책이며
   20개 run의 공정성 조건에는 차이가 없다.
5. 본 결과는 축소 Pilot 결과이며 최종 제품 성능이나 광범위한 일반화 성능으로 해석하지 않는다.
6. 음수 Recall은 네 구성 모두에서 상대적으로 낮게 관측되어 Phase 8 판단 시 함께 검토한다.
7. smoke directory 3개와 MLflow smoke run 1개는 production/aggregate에서 분리됐다.
8. Phase 7에서는 winner를 결정하지 않았으며, Phase 8에서 고정 정책으로 Experiment D를 선택했다.
```

새로운 blocker, 검증 실패, 설계 변경 필요성이 발생하면 이 섹션을 즉시 갱신한다.

---

## 9. Reference

상위 설계 기준:

```text
docs/00_Pilot_Design_Baseline.md
```

개발환경 기준:

```text
docs/01_개발환경_구축_기록.md
```

모델 구현 Reference:

```text
docs/03_Model_Implementation_References.md
```

Coding Agent 작업 규칙:

```text
AGENTS.md
```

프로젝트 개요:

```text
README.md
```

---

## 10. STATUS 갱신 원칙

이 문서는 작업 상태가 실제로 변경될 때 갱신한다.

반드시 갱신:

- Phase 변경
- 체크리스트 완료
- 중요한 구현 작업 완료
- blocker / issue 발생 또는 해소
- 다음 작업 / 우선순위 변경

`README.md`와 `AGENTS.md`에는 일상적인 진행 상태를 중복 기록하지 않는다.
