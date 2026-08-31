# 1차 Pilot Development Record

- 프로젝트: `medication-drinking-action-recognition`
- 작성 기준일: 2026-08-31
- 문서 상태: **FINAL — 1st Pilot Development Record**
- 기록 대상: Phase 0~10, `1st Pilot COMPLETE`
- 작성 방식: Repository, Git, SSD Raw/Working, MLflow DB, checkpoint와 평가 artifact를 직접 읽은 read-only evidence audit
- 경로 표기: `<project_root>` = `/home/user/projects/medication-drinking-action-recognition`, `<work_root>` = `/mnt/d/AI 도약과정/데이터/data_workspace/medication_drinking_action_data`

# 1. 문서 목적 및 범위

## 1.1 문서 목적

이 문서는 복약·물 마시기·기타 3-class 행동 인식 1차 Pilot이 설계, 구현, 검증된 과정을 실제 evidence로 복원한다. 단순 상태 요약이 아니라 데이터가 inventory와 manifest를 거쳐 ROI, Stage A, embedding cache, Stage B CV, 구조 선택, deployment/check model, raw-video 평가로 이어진 lifecycle과 provenance를 기록한다.

## 1.2 기록 범위

Phase 0의 repository baseline부터 Phase 10의 participant-disjoint raw-video OOF와 self-recorded pipeline check까지를 다룬다. Full Experiment는 시작되지 않았으며, Pilot 결과를 production 성능이나 외부 일반화의 증거로 확장하지 않는다.

## 1.3 작성 근거 및 Evidence 범위

직접 조사한 근거는 다음과 같다.

- 공식 문서: Design Baseline, 환경·Data Readiness, 데이터 구조·EDA, 구현 Reference, Phase 5/8/10 기록
- 구현: `configs/`, `src/`, `scripts/`, `tests/` 전체와 CLI 정의
- 실행 artifact: inventory/manifest summary와 JSONL/CSV, ROI report와 preview, Stage A/7/9 checkpoint 및 history, 239개 embedding cache, OOF/evaluation JSON·CSV, self-recorded 결과
- MLflow: `runtime/mlflow/mlflow.db`를 SQLite read-only URI로 조회
- Git: working tree, commit graph와 `git log --stat`; 현재 HEAD `793c4ed`, tag `pilot-v1.0`은 `08ebfd6`
- Raw: path config로 확정한 AI-Hub/ETRI root의 존재와 상위 구조. Raw 전체 재스캔은 하지 않고 기존 full inventory를 실행 수량 근거로 사용했다.

## 1.4 정보 충돌 시 판단 기준

실행 artifact → checkpoint/training/evaluation → 코드/config/manifest → Git history → final evaluation/STATUS → baseline/guide → README 순으로 실제 실행 상태를 판별했다. 예를 들어 AI-Hub actor는 candidate split population 202명과 selected Pilot 192명이 모두 맞지만 모집단이 다르므로 구분해 기록했다. Phase 10 consistency artifact의 verdict `DIFFERENCE_OBSERVED`는 prediction/metric 불일치가 아니라 확률 최대차 `1.2516975e-6`이 tolerance `1e-6`을 넘은 결과로 기록했다.

---

# 2. 프로젝트 목표

## 2.1 최종 문제 정의

짧은 RGB 영상에서 `복약(0)`, `음수(1)`, `기타(2)`를 분류한다. 최종 음수 target은 ETRI `A004` 물 마시기다. AI-Hub `Drink_bever`/`Drink_alcohol`은 Stage A의 auxiliary visual supervision이지 최종 class 의미를 모든 음료로 확장하지 않는다.

## 2.2 1차 Pilot의 목적

4GB GPU와 축소 데이터로 데이터 계약, leakage-safe split, ROI fallback, 2-stage interface, 2×2 비교, CV 평가, raw-video inference를 한 차례 완주해 구현 가능성과 주요 실패 양상을 확인하는 것이 목적이었다.

## 2.3 Pilot과 Full Experiment의 구분

Pilot은 AI-Hub 400 videos와 ETRI 239 clips로 workflow를 검증했다. Full Experiment는 전체 target/other pool 확대, sampling·loss·augmentation·hyperparameter 실험, 별도 일반화 검증을 포함하는 후속 범위이며 2026-08-31 현재 `NOT STARTED`다.

## 2.4 주요 제약 조건

- NVIDIA GTX 1650 Ti 4GB, WSL 2 환경
- 서로 다른 도메인의 frame dataset(AI-Hub)과 video dataset(ETRI)
- 복약·음수의 짧고 작은 물체/손-입 상호작용과 hard negative
- actor/participant leakage 방지 필요
- ETRI encoder joint fine-tuning을 제외하고 frozen embedding을 사용
- untouched independent final test 부재

## 2.5 Pilot에서 검증하려 한 핵심 질문

MobileNetV3-Small의 ImageNet 특징과 AI-Hub fine-tuned 특징 중 무엇이 ETRI에 유리한가, Mean Pooling과 GRU 중 temporal modeling이 이득인가, ROI 실패를 sample 삭제 없이 처리할 수 있는가, cached CV 결과가 실제 raw MP4 path에서 재현되는가를 검증했다.

---

# 3. 데이터 확보 및 이해

## 3.1 데이터셋 선정 배경

AI-Hub 일상생활 영상은 3-frame JPG와 actor metadata를 제공해 경량 visual encoder 학습에 사용했다. ETRI 리빙랩 RGB video는 participant와 행동 ID가 명확하고 물 마시기 `A004`를 포함해 최종 temporal task와 participant-disjoint 평가에 사용했다.

## 3.2 AI-Hub 데이터 구조

Raw root 아래 `1.Training`, `2.Validation`, 라벨링/원천/metadata 및 별도 EDA 기록이 존재한다. 실제 inventory master는 JSON/video candidate 18,420개이며 각 candidate가 JPG 3장을 참조한다. 관측 55,260 JPG/reference가 expected count와 일치했다. metadata 20,517 rows 중 local 18,420개가 모두 match했고 metadata-only 2,097 rows가 있었다.

## 3.3 ETRI 데이터 구조

Raw root에는 `RGB Videos`, `3D Skeletal Data`, `Body Index Frames`, docs/EDA가 있으며, 공식 데이터셋의 4번째 구성요소인 Depth Map은 로컬에 포함돼 있지 않다. Pilot target은 Batch B participant `P201`~`P230`의 RGB 영상이다. inventory는 30 participants, 6,589 clips, A003 복약 119개, A004 물 마시기 120개를 확인했다.

## 3.4 데이터별 실제 보유 범위

| 데이터 | 전체 inventory | 유효/usable | Warning |
|---|---:|---:|---|
| AI-Hub | 18,420 candidates / 55,260 JPG | 18,419 | `duplicate_frame_reference` 1 |
| ETRI Batch B | 6,589 RGB clips / 30 participants | 6,588 | `rgb_file_too_small` 1 (`P205/A053...`) |

두 inventory 모두 count mismatch, fatal/unisolated invalid, duplicate candidate/clip은 0이며 최종 `PASS_WITH_WARNINGS`였다.

실제 보유/미보유 범위는 다음과 같다. AI-Hub는 `1.Training`/`2.Validation` 라벨링 JSON, 영상당 sampled JPG 3장, metadata xlsx(20,517 rows), 행동분류 번호 대응표를 보유하며 원본 MP4와 행위 텍스트 CSV는 미보유다. ETRI Batch B는 RGB MP4, 3D Skeleton CSV, Body Index PNG를 보유하며 Depth Map은 미보유다. 모델 학습·평가의 직접 입력 modality는 AI-Hub JPG frame과 ETRI RGB MP4였다. ETRI Skeleton·Body Index는 모델 입력에 사용하지 않았으며, AI-Hub metadata는 모델 입력이 아니라 inventory 보강·데이터 식별 및 검증을 위한 보조 metadata source로 활용했다.

## 3.5 실제 Pilot 사용 범위

| Stage | 복약 | 음수 | 기타 | 합계 |
|---|---:|---:|---:|---:|
| AI-Hub selected videos | 100 | 100 | 200 | 400 videos / 1,200 frames |
| ETRI selected clips | 59 | 60 | 120 | 239 clips |

AI-Hub selected actor는 192명(train 152, val 40)이며 교집합은 없다. split 자체는 전체 candidate actor 202명(train side 162, val side 40)에 먼저 부여됐다. ETRI는 fold별 participant 6명, clip 수 `48/48/48/48/47`; P227의 유효 복약이 1개라 총 복약이 59개다. 기타는 hard negative 60 + general other 60이다.

## 3.6 데이터 품질 / 제약 사항

AI-Hub는 3장의 frame-level proxy이고 음수 의미가 ETRI 물 마시기와 완전히 같지 않다. ETRI는 target 수가 적고 기타가 두 배이며 A045~A048 multi-person 행동은 Pilot에서 제외했다. invalid는 삭제하지 않고 inventory에 `valid=false`, `pilot_selected=false`로 남겼다.

---

# 4. EDA와 문제 정의

## 4.1 AI-Hub 주요 EDA 결과

구조·EDA 문서는 복약 `Take_pills`, 음수 auxiliary `Drink_bever`/`Drink_alcohol`, 다양한 기타 행동을 확인했다. 로컬 AI-Hub 라벨링데이터는 `viewpoint_1`(1인칭)과 `viewpoint_3`(3인칭) 두 시점만 존재하고(`viewpoint_2` 없음), 각 영상이 sampled JPG 3장과 JSON 1건을 참조한다. Pilot은 `viewpoint_3`만 사용했으므로 "3-view"가 아니라 "단일 시점의 영상당 3-frame proxy"로 이해해야 한다. 손·얼굴·작은 객체가 핵심이지만 단일 frame만으로는 행동 전후 맥락이 약하며, `Eat_food`처럼 손-입 motion이 유사한 hard negative가 존재했다.

## 4.2 ETRI 주요 EDA 결과

복약 A003과 물 마시기 A004가 참가자별 반복 촬영되고, 다수의 일상행동이 기타 후보를 이룬다. participant identity와 take가 강한 상관을 가지므로 clip random split은 leakage 위험이 있었다. A045~A048 multi-person 및 한 개 손상 RGB라는 품질 제약도 확인됐다.

## 4.3 클래스 정의와 불균형 문제

최종 class mapping은 `복약:0, 음수:1, 기타:2`다. ETRI Batch B 원본은 복약 119, 음수 120, 기타 6,350으로 극심한 클래스 불균형을 가진다. Pilot에서는 전체 workflow 검증과 target sensitivity 확인을 위해 participant 다양성을 유지하면서 59:60:120으로 기타 비중을 의도적으로 완화했다. 그럼에도 class별 성능 차이를 감추지 않기 위해 accuracy보다 Macro-F1과 per-class Recall을 중심으로 평가했다.

## 4.4 주요 혼동 / Hard Negative

섭취·얼굴 만지기·양치·전화 등 손이 얼굴 주변으로 이동하는 행동, 컵/약처럼 작은 물체, 행동 순간이 짧은 clip이 주요 혼동원이었다. 실제 최종 OOF에서도 복약 30/59, 음수 31/60이 기타로 분류되어 이 문제 정의가 확인됐다.

## 4.5 EDA 결과가 Pilot 설계에 미친 영향

발견은 actor/participant 단위 split, 기타의 hard-negative/general-other 균형, 얼굴·손·상체 contextual ROI, `T=64` uniform sampling, frame encoder와 temporal classifier 분리, class-balanced 관점의 Macro-F1/Recall 채택으로 연결됐다.

---

# 5. 모델 구조 리서치

## 5.1 요구 조건

4GB GPU에서 학습·추론 가능하고, `[B,3,H,W]→[B,D]`와 `[B,T,D]→[B,3]`의 명확한 계약, frozen cache 재사용, 짧은 temporal context 지원이 필요했다.

## 5.2 검토한 모델 / 접근법

공식 PyTorch/torchvision의 MobileNetV3-Small, Mean Pooling+Linear, `nn.GRU`+Linear를 주 후보로 검토했다. 구현 Reference에는 3D CNN(R3D/MC3), SlowFast, optical-flow Two-Stream, 섭취 행동용 2D CNN+temporal 접근도 비교 대상으로 기록돼 있다.

## 5.3 후보별 장단점

MobileNetV3-Small은 경량·ImageNet warm start와 1024-D feature를 제공한다. Mean은 단순하고 빠르나 순서를 버린다. GRU는 순서를 보존하면서 cache 학습이 가능하지만 parameter와 과적합 위험이 늘어난다. 3D/Two-Stream은 강한 temporal modeling 가능성이 있으나 GPU·I/O·구현 복잡도가 크다.

## 5.4 제외한 구조와 이유

3D CNN, SlowFast, optical flow, TensorFlow pipeline 이식, 외부 taxonomy/split/threshold는 Pilot 자원과 고정 설계 범위를 벗어나 제외했다. 외부 repository 코드는 직접 재사용하지 않고 orchestration/방법론만 참고했다.

## 5.5 최종 후보 구조 도출

Stage A를 ImageNet pretrained MobileNetV3-Small로 고정하고, Encoder A(ImageNet-only)와 Encoder B(AI-Hub fine-tuned), Stage B Mean/GRU를 직교 조합한 A/B/C/D가 Pilot 후보가 됐다.

---

# 6. Pilot Design 결정

## 6.1 전체 Pilot 전략

`Raw → full inventory → split/fold → fixed subset → ROI gate → Stage A → frozen cache → 2×2 CV → selection → full-Pilot deployment/check → raw-video OOF` 순서로 결정해 데이터 선택이 결과에 따라 바뀌지 않도록 했다.

## 6.2 AI-Hub / ETRI 역할 분리

AI-Hub는 `viewpoint_3` visual representation 학습, ETRI Batch B는 최종 3-class temporal 학습·평가를 담당한다. 서로 다른 음수 의미와 split 단위를 섞지 않았다.

## 6.3 Stage A / Stage B 설계

Stage A 입력 `[B,3,224,224]`, embedding `[B,1024]`; Stage B 입력 `[B,64,1024]`, logits `[B,3]`이다. 최종 GRU는 hidden 128, 1 layer, unidirectional, final hidden, dropout 0.0이다.

## 6.4 Split / Leakage 방지 정책

AI-Hub actor-disjoint train/val, 동일 video 3 frames 동일 split; ETRI participant-disjoint 5-fold, 동일 participant의 모든 take 동일 fold를 강제했다. validation 실패 시 자동 보정하지 않고 중단한다.

## 6.5 Fixed Pilot Manifest

seed 42로 split/fold를 먼저 고정하고 subset을 선택했다. AI-Hub manifest SHA-256은 `9dbe8e3100c47a6e81be999309a11e202b6c6a7e5c5f20c924816a4e448e6e9e`, ETRI는 `0c641e3301196afa92c4cf7b7cad28dfd9e21c5f88a5ebbe02b694110b3b4b93`이다.

## 6.6 ROI Preflight

MediaPipe face/hand/pose로 contextual crop을 만들고 상태를 `success/partial/fallback`으로 기록했다. 대표 64-frame 최종 report는 success 4, partial 60, fallback 0으로 `PASS_WITH_WARNINGS`; 실패 sample 삭제 대신 full-frame fallback을 유지했다.

## 6.7 Frozen Encoder / Embedding Cache

두 encoder를 ETRI에서 frozen하고 동일 239 clips, 동일 64 frame indices에 `[64,1024]` 두 embedding을 저장했다. cache는 재생성 가능한 derived artifact이며 manifest/config/checkpoint hash를 보유한다.

## 6.8 2×2 Ablation

A=ImageNet+Mean, B=AI-Hub+Mean, C=ImageNet+GRU, D=AI-Hub+GRU를 seed 42, CE, AdamW, lr `1e-4`, weight decay `1e-4`, batch 16, 15 epochs로 5 folds 비교했다.

## 6.9 평가 지표와 Model Selection 정책

Primary는 fold Macro-F1의 5-fold mean, 표준편차는 population `ddof=0`이다. 동률이면 복약 recall, 음수 recall, std, 단순성, latency/memory 순이다. 실제로 D가 단독 1위여서 tie-break는 쓰지 않았다.

## 6.10 Self-recorded 사용 정책

학습·metric·ROI/threshold/model 선택에는 사용하지 않고 decode부터 JSON 출력까지 기능·정성 확인에만 사용했다.

## 6.11 End-to-End의 정의

Raw video decode → fixed-uniform sampling → ROI → frozen encoder → Stage B → softmax를 뜻한다. CNN과 GRU를 joint optimization했다는 뜻이 아니다.

---

# 7. 개발환경 및 Repository 구축

## 7.1 개발환경

Windows Host, WSL 2 Ubuntu 26.04.1 LTS, Windows VS Code+WSL, `uv`, `.venv`, Python 3.12.14 환경이다. snapshot에는 PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA runtime 12.6, MediaPipe 1.0.1, OpenCV 5.0.0, MLflow 3.15.2가 기록됐다. GTX 1650 Ti 4GB에서 CUDA tensor 및 Phase 5/7/9, self-recorded GPU 실행이 확인됐다. WSL MediaPipe 실행 전 `libGLESv2.so.2` prerequisite 문제가 해결됐다.

## 7.2 Repository 구조 (WSL)

`configs`는 공유/로컬 경로 및 Phase config, `docs`는 기준·EDA·결과, `src`는 데이터/model/evaluation core, `scripts`는 CLI, `tests`는 unit/smoke contract, `manifests`는 placeholder(`.gitkeep`만 tracked), `runtime`은 Git 제외 MLflow를 맡는다. 현재 `src` 모듈 14개(`__init__.py` 제외), `scripts` CLI 12개(`run_etri_embedding_smoke.py` 포함), `tests` module 12개가 확인됐다.

## 7.3 Raw / Working Data 구조 (SSD)

- AI-Hub Raw: `/mnt/d/AI 도약과정/데이터/data_raw/057.일상생활_영상_데이터/01.데이터`
- ETRI Raw: `/mnt/d/AI 도약과정/데이터/data_raw/고령자 일상행동인식 3차원 영상 데이터셋(리빙랩)`
- Working: `<work_root>` 아래 `manifests`, `roi_preflight`, `embeddings`, `checkpoints`, `evaluations`, `cache`
- Self-recorded: `<work_root>/self_recorded/pipeline_check`

Raw는 immutable이며 derived output은 Working에만 저장한다.

## 7.4 Git / MLflow 관리

Git은 `d7a6101` baseline부터 Phase 1 `4f28e00`, inventory `aff8719/d6f336e`, manifest `b8dfdb6`, ROI `e05f074`, Stage A `7c3f1c3`, Phase 6 `1936928`, Phase 7 `60c72bc/3a5ad4a`, Phase 8 `f79401b`, Phase 9 `dacd931`, Phase 10 `f0b5916` 순이다. `pilot-v1.0` tag는 문서 freeze `08ebfd6`, 현재 HEAD는 README provenance 보강 `793c4ed`다.

MLflow DB는 `<project_root>/runtime/mlflow/mlflow.db`(1,400,832 bytes)다. 실험은 `stage_a_visual_encoder`, `phase7_etri_2x2_ablation`, `phase9_pilot_deployment_check`; 총 23 runs(Stage A 1, Phase 7 smoke 1 + production 20, Phase 9 1)이 모두 `FINISHED`다. Phase 7 smoke run은 DB에 `phase7_A_fold1_smoke` 1개만 기록됐다. 533 params, 1,240 metric rows, 161 tags가 존재한다.

## 7.5 Artifact 관리 구조

코드·공유 config·문서·소형 selection config는 Git 관리한다. machine path, runtime/MLflow, checkpoint, embedding, evaluation, ROI image는 제외한다. Raw는 재생성 불가능한 원본, inventory 이후 산출물은 원본+config+code provenance로 재생성 가능한 derived artifact다.

## 7.6 주요 파일 및 디렉토리 역할

| 위치 | 역할 | 관리 |
|---|---|---|
| `configs/phase8_selected_model.yaml` | D 선택과 hash handoff | Git |
| `<work_root>/manifests` | full inventory와 fixed Pilot | SSD derived |
| `<work_root>/roi_preflight` | 64-frame gate/report/visuals | SSD derived |
| `<work_root>/embeddings/.../clips` | 239개 Encoder A/B cache | SSD derived |
| `<work_root>/checkpoints/stage_a` | Encoder B best | SSD derived |
| `<work_root>/checkpoints/phase7_ablation` | 20 fold model+OOF | SSD derived |
| `<work_root>/checkpoints/phase9_deployment` | 단일 기능 확인 모델 | SSD derived |
| `<work_root>/evaluations` | Phase 10 quantitative/qualitative evidence | SSD derived |
| `runtime/mlflow` | DB와 run artifacts | runtime/Git 제외 |

---

# 8. Phase 0~10 작업 과정

## 8.1 Phase 0 — Repository Baseline

1. **목적과 시작 조건**: 설계·환경 문서와 Raw/Working 경로가 준비된 상태에서 재현 가능한 Git 기준선을 만든다.
2. **입력 데이터 / 이전 Phase Artifact**: Baseline, 환경·EDA·구조 문서, `.python-version`, lock file.
3. **수행 작업 및 주요 구현**: `main` repository, `.gitignore`, AGENTS/README/STATUS 역할과 기본 tree 확정.
4. **생성 Artifact**: commit `d7a6101`, 후속 상태 commit `70fd237`.
5. **Artifact의 이후 활용**: 모든 구현 commit의 provenance 기준.
6. **검증 방법 및 실제 실행 결과**: branch/status, ignore와 staging dry-run PASS; raw/runtime/local config 제외 확인.
7. **발견된 문제 / 대응 / 남은 Warning**: Windows Zone.Identifier 잔재와 ownership/경로를 정리; 환경은 고정.
8. **완료 기준 및 최종 판정**: clean baseline과 안전 규칙 확정, COMPLETE.

## 8.2 Phase 1 — Path Infrastructure

1. **목적과 시작 조건**: PC 절대경로를 코드에서 분리한다.
2. **입력 데이터 / 이전 Phase Artifact**: `configs/paths.local.yaml`의 3 roots와 runtime path.
3. **수행 작업 및 주요 구현**: `src/path_config.py`, example/local config 계약, required path validation.
4. **생성 Artifact**: commit `4f28e00`, `manifests/scripts/src/tests` 골격.
5. **Artifact의 이후 활용**: 모든 CLI가 동일 path resolution을 사용.
6. **검증 방법 및 실제 실행 결과**: `tests/test_path_config.py`; Raw readable, Working/runtime 분리.
7. **발견된 문제 / 대응 / 남은 Warning**: 공백·한글 경로는 YAML/quoted shell로 처리.
8. **완료 기준 및 최종 판정**: local path Git 제외와 roots validation PASS.

## 8.3 Phase 2 — Full Candidate Inventory

1. **목적과 시작 조건**: 삭제 없는 master inventory와 품질 flag를 만든다.
2. **입력 데이터 / 이전 Phase Artifact**: AI-Hub JSON/JPG/metadata, ETRI Batch B RGB.
3. **수행 작업 및 주요 구현**: `ai_hub_inventory.py`, `etri_inventory.py`, build CLI, schema/count/duplicate/validity 검사.
4. **생성 Artifact**: `<work_root>/manifests/ai_hub`, `.../etri`; summary와 inventory JSONL.
5. **Artifact의 이후 활용**: Phase 3 fixed selection의 유일 입력.
6. **검증 방법 및 실제 실행 결과**: AI-Hub 18,420/55,260, ETRI 6,589/30 expected=observed; invalid 각 1개 격리.
7. **발견된 문제 / 대응 / 남은 Warning**: duplicate frame reference 1, too-small RGB 1; Raw 수정 없이 flag.
8. **완료 기준 및 최종 판정**: 두 dataset `PASS_WITH_WARNINGS`, fatal 0.

## 8.4 Phase 3 — Fixed Pilot Manifest

1. **목적과 시작 조건**: 결과를 보기 전에 leakage-safe split/fold와 subset을 동결한다.
2. **입력 데이터 / 이전 Phase Artifact**: Phase 2 full inventories, seed/cap config.
3. **수행 작업 및 주요 구현**: actor split, participant folds, deterministic stratified/capped selection, leakage validator.
4. **생성 Artifact**: AI-Hub/ETRI manifest JSONL, selected CSV, summary, 두 SHA-256.
5. **Artifact의 이후 활용**: Phase 4~10의 population과 fold 기준.
6. **검증 방법 및 실제 실행 결과**: AI-Hub 400/1,200, ETRI 239/30/5 folds; participant·actor overlap 0; 재생성 결정성 PASS.
7. **발견된 문제 / 대응 / 남은 Warning**: P227 복약 1개로 fold 4가 47 clips; 자동 보충 없이 실제 availability 유지.
8. **완료 기준 및 최종 판정**: 모든 selected row valid, split/fold guard PASS.

## 8.5 Phase 4 — ROI Preflight

1. **목적과 시작 조건**: 전체 preprocessing 전에 얼굴·손·행동 context 보존과 fallback을 확인한다.
2. **입력 데이터 / 이전 Phase Artifact**: 두 fixed manifest, MediaPipe face/hand/pose task, 대표 seed 42.
3. **수행 작업 및 주요 구현**: representative selector, frame/video loader, contextual ROI, overlay/crop/preview, class별 report.
4. **생성 Artifact**: report JSON/CSV와 64 samples×3 visual images.
5. **Artifact의 이후 활용**: 동일 ROI config hash가 embedding/raw inference에 전달.
6. **검증 방법 및 실제 실행 결과**: 최초 ETRI fallback 68.75% 이후 pose contextual path 추가; 최종 4 success/60 partial/0 fallback, visual review.
7. **발견된 문제 / 대응 / 남은 Warning**: hand-only crop이 작아 원본 비율 기반 context로 확장. partial 비율은 높아 warning 유지.
8. **완료 기준 및 최종 판정**: fallback 0과 context 보존, `PASS_WITH_WARNINGS`.

## 8.6 Phase 5 — Stage A Visual Encoder

1. **목적과 시작 조건**: Encoder A/B의 동일 1024-D interface와 AI-Hub transfer checkpoint를 만든다.
2. **입력 데이터 / 이전 Phase Artifact**: AI-Hub 400-video manifest, ROI/preprocessing, ImageNet MobileNetV3-Small.
3. **수행 작업 및 주요 구현**: 마지막 2 blocks fine-tune, BN running stats freeze, frame/video evaluation, MLflow, reload verifier.
4. **생성 Artifact**: run `stage-a-20260829T170113531051Z-adf86795`, `best.pt` SHA `263936...fd6`, effective config/history; MLflow run `ffb970...a2c9`.
5. **Artifact의 이후 활용**: Encoder B cache와 Phase 10 raw inference.
6. **검증 방법 및 실제 실행 결과**: batch 8, 15 epochs; best epoch 12 frame Macro-F1 0.536789, video Macro-F1 0.652049, video recalls 0.55/0.60/0.80, video CM `[[11,6,3],[2,12,6],[4,4,32]]`; reload deterministic.
7. **발견된 문제 / 대응 / 남은 Warning**: history상 train loss는 0.95→0.09로 계속 내려가지만 validation loss는 약 0.92(epoch 2)→1.39(best epoch 12)→1.51(epoch 13)로 상승하는 과적합 신호가 기록됐다. best checkpoint는 이 구간에서도 policy대로 validation video Macro-F1(epoch 12) 기준으로 선택했고 별도 retuning은 하지 않았다. frame보다 video aggregate가 높지만 어느 값도 ETRI 일반화 metric은 아니다.
8. **완료 기준 및 최종 판정**: A/B `[B,1024]`, checkpoint provenance/reload PASS; COMPLETE.

## 8.7 Phase 6 — ETRI Embedding Cache

1. **목적과 시작 조건**: frozen A/B를 동일 ETRI frames에 적용해 CV 반복 비용을 제거한다.
2. **입력 데이터 / 이전 Phase Artifact**: ETRI manifest SHA `0c641e33...`, ROI SHA `7b692e...`, Encoder B checkpoint.
3. **수행 작업 및 주요 구현**: fixed-uniform 64 sampling, batched encoding 8, resumable validated cache, parity/NaN/Inf gate.
4. **생성 Artifact**: `embeddings/etri_pilot_t64_ab/manifest-0c641e33`, 239 `.pt`; 각 파일은 A/B `[64,1024]`와 metadata, 약 528KB.
5. **Artifact의 이후 활용**: Phase 7 CV 및 Phase 9 full-Pilot Stage B training.
6. **검증 방법 및 실제 실행 결과**: 227 created+12 resumed=239 success, failures/NaN/Inf 0, clip/frame-index parity true; ROI 15,178 partial+118 fallback.
7. **발견된 문제 / 대응 / 남은 Warning**: ETRI 15,296 frames 중 success 0, fallback 0.7714%; sample은 제외하지 않음.
8. **완료 기준 및 최종 판정**: full-239 validation `PASS`, Phase 7 ready with ROI warning.

## 8.8 Phase 7 — 2×2 Ablation

1. **목적과 시작 조건**: encoder source와 temporal head 효과를 공정한 5-fold 조건에서 비교한다.
2. **입력 데이터 / 이전 Phase Artifact**: immutable 239 cache, fixed folds/config.
3. **수행 작업 및 주요 구현**: Mean/GRU model, dataset guard, trainer, per-fold best/history/OOF, aggregate, MLflow.
4. **생성 Artifact**: A/B/C/D×5 production checkpoint 20개, fold metrics/OOF, 네 aggregate summaries; MLflow production 20 runs.
5. **Artifact의 이후 활용**: Phase 8 selection; D fold checkpoints는 Phase 10 공식 OOF에 재사용.
6. **검증 방법 및 실제 실행 결과**: A `0.450650±0.079355`, B `0.449102±0.087356`, C `0.518992±0.042044`, D `0.532176±0.054575`; leakage/239 OOF completeness PASS.
7. **발견된 문제 / 대응 / 남은 Warning**: target recall이 낮고 fold variation 존재. 비교 조건은 고정.
8. **완료 기준 및 최종 판정**: clean implementation commit `60c72bc`, 20 runs/OOF complete, `PASS_WITH_WARNINGS`.

## 8.9 Phase 8 — Structure Selection

1. **목적과 시작 조건**: 사전 고정 policy로 구조만 선택한다.
2. **입력 데이터 / 이전 Phase Artifact**: 네 aggregate summary와 artifact hashes.
3. **수행 작업 및 주요 구현**: completeness/fairness/leakage audit, ranking, immutable handoff YAML.
4. **생성 Artifact**: `configs/phase8_selected_model.yaml`, SHA `8e9966...638`; result document.
5. **Artifact의 이후 활용**: Phase 9/10 encoder·GRU config와 checkpoint path 결정.
6. **검증 방법 및 실제 실행 결과**: D 0.532176이 C보다 0.013184 높아 rank 1; tie 없음.
7. **발견된 문제 / 대응 / 남은 Warning**: D 복약 recall 0.357576은 C보다 0.016667 낮지만 primary metric을 사후 변경하지 않음.
8. **완료 기준 및 최종 판정**: Experiment D selected, COMPLETE.

## 8.10 Phase 9 — Deployment/Check Model

1. **목적과 시작 조건**: 선택된 구조로 single-video 기능 확인에 쓸 단일 Stage B를 만든다.
2. **입력 데이터 / 이전 Phase Artifact**: Encoder B cache, selected D config, selected-valid 239 전체.
3. **수행 작업 및 주요 구현**: GRU 재초기화, fold filter 없이 15 epochs full-Pilot training, reload/provenance/MLflow.
4. **생성 Artifact**: `deployment_check.pt` SHA `8065c1...736`, diagnostics; MLflow `fe14b2...5659` FINISHED.
5. **Artifact의 이후 활용**: representative ETRI와 self-recorded single-video inference. 공식 OOF에는 사용하지 않음.
6. **검증 방법 및 실제 실행 결과**: dataset 239/30, duplicate/invalid 0; epoch 15 training diagnostic Macro-F1 0.838143; reload shape `[1,3]` PASS.
7. **발견된 문제 / 대응 / 남은 Warning**: training-set diagnostic은 generalization metric이 아니며 best model selection 근거로 사용하지 않음.
8. **완료 기준 및 최종 판정**: full scope, fixed encoder, checkpoint restore/provenance PASS.

## 8.11 Phase 10 — Pipeline Integration and Final Evaluation

1. **목적과 시작 조건**: cache 밖 raw-video path와 선택 구성의 기능을 검증한다.
2. **입력 데이터 / 이전 Phase Artifact**: Encoder B, D fold0~4, Phase 9 single model, ETRI manifest, self-recorded 3 videos.
3. **수행 작업 및 주요 구현**: shared preprocessing single-video inference, fold-routed raw OOF, cache consistency, JSON output.
4. **생성 Artifact**: raw predictions/metrics/fold CMs/aggregate CM/consistency/failure JSON; self-recorded 3 JSON; final evaluation 문서.
5. **Artifact의 이후 활용**: Pilot의 공식 quantitative evidence와 integration limitation.
6. **검증 방법 및 실제 실행 결과**: raw 239/239, failure/duplicate/missing/leakage 0; aggregate Macro-F1 0.538337, accuracy 0.623431; cached prediction 239/239 일치. self-recorded 3/3 crash 없이 CUDA 완료.
7. **발견된 문제 / 대응 / 남은 Warning**: 확률 max diff가 tolerance를 0.2517e-6 초과; class/metric/CM은 동일. self target 2건은 기타 오분류.
8. **완료 기준 및 최종 판정**: required documentation fix 반영, regression 113 PASS, `Phase 10 COMPLETE / 1st Pilot COMPLETE`.

---

# 9. 최종 모델 구조

## 9.1 전체 Raw-video Inference Pipeline

```text
RGB MP4 → frame count → 64 uniform indices → decode
→ MediaPipe contextual ROI/full-frame fallback → resize/normalize
→ Encoder B frame embedding → GRU final hidden → Linear(3) → softmax
```

## 9.2 Stage A Visual Encoder

ImageNet pretrained MobileNetV3-Small을 AI-Hub actor-disjoint Pilot로 last 2 blocks fine-tune한 Encoder B다. classifier 이전 1024-D feature를 사용하며 ETRI에서는 완전히 frozen이다.

## 9.3 Stage B Temporal Classifier

`GRU(input_size=1024, hidden_size=128, num_layers=1, bidirectional=false, batch_first=true)`의 final hidden을 `Linear(128,3)`에 입력한다.

## 9.4 Tensor / Interface 구조

`[64,3,224,224] → [64,1024] → [1,64,1024] → [1,3]`. normalization mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`다.

## 9.5 Training 구조와 Inference 구조의 관계

Stage A는 AI-Hub frame/video proxy에서 학습했다. ETRI Stage B는 cached frozen embedding으로 학습했지만 Phase 10 inference는 raw frames에서 같은 ROI/normalization/encoder를 실시간 실행한다. 양 path의 239 predictions와 metric은 일치했다.

## 9.6 Fold Checkpoint와 Deployment Checkpoint의 역할 차이

D fold checkpoint는 해당 participant fold를 보지 않고 학습돼 공식 OOF 평가에 사용한다. Phase 9 checkpoint는 239개 전체를 보았으므로 성능 평가에 사용할 수 없고 single-video/deployment 기능 확인에만 사용한다.

---

# 10. 최종 평가 결과

## 10.1 Model Selection

| Exp | Encoder | Stage B | 5-fold Macro-F1 mean±std | 복약 R | 음수 R | 기타 R |
|---|---|---|---:|---:|---:|---:|
| A | ImageNet | Mean | 0.450650±0.079355 | 0.339394 | 0.266667 | 0.783333 |
| B | AI-Hub | Mean | 0.449102±0.087356 | 0.221212 | 0.300000 | 0.866667 |
| C | ImageNet | GRU | 0.518992±0.042044 | 0.374242 | 0.300000 | 0.883333 |
| D | AI-Hub | GRU | **0.532176±0.054575** | 0.357576 | 0.333333 | 0.900000 |

## 10.2 Raw-video OOF Evaluation

30 participants/5 folds의 239 selected-valid clips를 각자의 held-out fold model로 정확히 한 번 평가했다. aggregate Macro-F1 `0.538337`, accuracy `0.623431`, macro precision `0.604227`, macro recall `0.529755`; 239/239 성공이다. fold mean과 aggregate는 계산 방식과 fold 크기 차이로 동일하지 않다.

## 10.3 Class별 평가 결과

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 복약 | 0.567568 | 0.355932 | 0.437500 | 59 |
| 음수 | 0.606061 | 0.333333 | 0.430108 | 60 |
| 기타 | 0.639053 | 0.900000 | 0.747405 | 120 |

## 10.4 Confusion Matrix / Error Analysis

행=true, 열=prediction, 순서 복약/음수/기타다.

```text
[[21, 8, 30],
 [ 9,20, 31],
 [ 7, 5,108]]
```

가장 큰 오류는 target을 기타로 놓치는 방향이다. 기타→target false positive는 12건인데 target→기타 false negative는 61건이다. 작은 물체·핵심 순간·domain/participant variation·불균형·frozen encoder는 가능한 설명이지만 artifact만으로 원인을 확정할 수는 없다.

## 10.5 Self-recorded Pipeline Check

세 영상 모두 CUDA에서 decode/ROI/Encoder B/Phase 9 GRU/JSON 생성이 PASS했다. 그러나 medication(0.6392), drinking(0.8251)은 모두 기타로, other(0.6825)는 기타로 예측됐다. 이는 기능 성공이지만 3개로 metric을 계산하거나 tuning/model selection에 사용할 수 없다.

## 10.6 Limitations 및 결과 해석 범위

239 clips/30명 Pilot, untouched test 부재, CV의 선택·추정 이중 사용, target recall 저조, 99.23% partial ROI와 0.77% fallback, ETRI encoder freeze, AI-Hub/ETRI/self-recorded domain shift, self-recorded 3건, 기타 120 대 복약 59·음수 60의 class imbalance, ETRI Batch A 및 cross-batch·external dataset 일반화 미평가, raw OOF(CPU)와 self-recorded(CUDA) 사이의 체계적 CPU/CUDA parity benchmark 부재라는 한계가 있다. 이 항목들은 공식 `docs/06_Pilot_Final_Evaluation.md` §16과 일치한다. Phase 9 training diagnostic과 self-recorded PASS는 성능 근거가 아니다.

---

# 11. 프로젝트에서 얻은 기술적 판단

## 11.1 데이터 관련 판단

full inventory를 삭제 없는 master로 유지하고 split 후 subset을 고정하면 invalid와 selection provenance를 보존할 수 있다(Phase 2~3). 목표 행동보다 기타 pool 설계가 결과에 크게 관여하며 실제 OOF는 기타 편향을 보였다.

## 11.2 모델 구조 관련 판단

이 Pilot에서는 GRU(C/D)가 Mean(A/B)보다 각각 +0.068342/+0.083074 높아 temporal order 사용이 유리했다. AI-Hub transfer 효과는 Mean에서 -0.001548, GRU에서 +0.013184로 작고 구조 의존적이어서 transfer 자체의 보편적 우위를 뜻하지 않는다.

## 11.3 평가 / Leakage 관련 판단

participant-disjoint fold와 fold-routed raw OOF가 identity leakage를 막는 필수 계약이었다. aggregate Macro-F1과 fold mean은 구분해야 하며, deployment full-data checkpoint를 OOF에 쓰면 평가가 무효가 된다.

## 11.4 실험 관리 / 재현성 관련 판단

manifest/config/checkpoint SHA와 MLflow/Git commit을 함께 남긴 덕분에 cache와 raw path를 sample 단위로 대조할 수 있었다. 엄격한 probability tolerance는 prediction이 같은 미세 numerical drift도 드러냈다.

## 11.5 개발 프로세스 관련 판단

ROI preflight에서 68.75% fallback을 발견한 뒤 pose path를 보완한 과정은 작은 gate가 대량 derived artifact 이전에 필요함을 보여준다. unit/smoke→full user execution→independent audit 순서가 4GB 환경에서 위험과 비용을 제한했다.

---

# 12. 재현 방법 및 주요 Artifact

## 12.1 재현 절차 및 주요 실행 명령

아래는 실제 CLI에서 확인한 흐름이다. full scan/train/evaluation은 장시간 작업이므로 명령 기록이며 본 audit에서는 실행하지 않았다.

```bash
# Environment/path
cp configs/paths.example.yaml configs/paths.local.yaml
.venv/bin/python -m unittest discover -s tests -q

# Inventory → fixed manifests → ROI gate
.venv/bin/python scripts/build_ai_hub_inventory.py
.venv/bin/python scripts/build_etri_inventory.py
.venv/bin/python scripts/build_pilot_manifests.py
.venv/bin/python scripts/run_roi_preflight.py

# Stage A → embeddings
.venv/bin/python scripts/run_stage_a.py smoke
.venv/bin/python scripts/run_stage_a.py train
.venv/bin/python scripts/run_stage_a.py evaluate --checkpoint <work_root>/checkpoints/stage_a/<run>/best.pt
.venv/bin/python scripts/run_stage_a.py verify-encoders --checkpoint <work_root>/checkpoints/stage_a/<run>/best.pt
.venv/bin/python scripts/run_etri_embedding_cache.py --resume

# 2×2 → selection → full-Pilot check model
.venv/bin/python scripts/run_phase7_ablation.py --validate-only
.venv/bin/python scripts/run_phase7_ablation.py --all-experiments
.venv/bin/python scripts/run_phase8_selection.py
.venv/bin/python scripts/run_phase9_deployment.py --dry-run
.venv/bin/python scripts/run_phase9_deployment.py --train

# Official raw OOF → single video
# 공식 artifact는 아래 명령을 default output(<work_root>/evaluations/phase10_raw_video_oof)으로 실행해 생성됐다.
# run_phase10_raw_video_oof.py는 기존 output directory를 덮어쓰지 않고 중단하므로, 재검증 시에는
# <work_root> 내부의 새 --output-root를 지정한다(공식 artifact는 그대로 둔다).
.venv/bin/python scripts/run_phase10_raw_video_oof.py --device cpu \
  --output-root "<work_root>/evaluations/phase10_raw_video_oof_recheck"
.venv/bin/python scripts/run_inference.py <video.mp4> --device auto --output-json <result.json>
```

Phase 7/9 CLI의 `--overwrite`는 명시적 재생성 옵션이며 기존 artifact를 보호하기 위해 기본 사용하지 않는다. `run_phase10_raw_video_oof.py`(output directory)와 `run_inference.py`(`--output-json` 파일)는 `--overwrite` 없이 기존 산출물을 거부한다. `--limit`, `--smoke` 결과는 final metric을 생성하지 않는다.

## 12.2 Artifact Map

```text
Repository
├─ configs/                 phase/path/selection contracts
├─ docs/                    baseline, EDA, guides, evaluation, this record
├─ src/                     inventory → model → inference core
├─ scripts/                 12 CLIs
├─ tests/                   12 test modules
└─ runtime/mlflow/          DB + run artifacts (Git excluded)

SSD Raw (immutable)
├─ AI-Hub Training/Validation JSON+JPG+metadata
└─ ETRI RGB Videos + skeletal/body-index references

SSD Working
├─ manifests/{ai_hub,etri,pilot}
├─ cache/mediapipe
├─ roi_preflight/{ai_hub,etri,reports}
├─ embeddings/etri_pilot_t64_ab/manifest-0c641e33
├─ checkpoints/{stage_a,phase7_ablation,phase9_deployment}
├─ evaluations/{phase10_raw_video_oof,phase10_self_recorded}
└─ self_recorded/pipeline_check
```

대표 provenance는 AI-Hub manifest `9dbe8e...e9e`, ETRI manifest `0c641e...b93`, Encoder B `263936...fd6`, Phase 9 `8065c1...736`, raw metrics `09b794...3bf`다.

## 12.3 Artifact Dependency

```text
AI-Hub Raw ─→ AI-Hub Full Inventory ─→ actor split + Fixed Pilot(400)
                                      └→ ROI policy ─→ Stage A best Encoder B

ETRI Raw ───→ ETRI Full Inventory ───→ participant folds + Fixed Pilot(239)
                                         │
ROI policy + Encoder A/B + manifest ─────┴→ A/B Embedding Cache [239,64,1024]
                                              │
                                              └→ A/B/C/D × 5-fold
                                                  ├→ aggregate → D selection
                                                  ├→ D fold checkpoints → raw-video OOF
                                                  └→ selected D + all 239 retrain
                                                       → deployment_check.pt
                                                       → ETRI/self-recorded single-video check

raw-video OOF ─→ cached OOF consistency ─→ Pilot Final Evaluation
```

---

## Evidence audit 주석

- 현재 working tree는 이 문서 생성 전 clean이었고 branch는 `main...origin/main`이었다.
- Raw 전체 파일 수를 본 작업에서 다시 세지 않았다. `full_run=true` inventory summary의 expected/observed count를 실행 evidence로 사용했다.
- checkpoint binary는 경로·크기·연결된 diagnostics/evaluation SHA와 reload 기록으로 확인했다. 새 checkpoint를 생성하지 않았다.
- 최종 회귀 기록은 113 PASS이며, 본 문서 작성 후 별도 정적/테스트 검증 결과는 완료 보고에 기록한다.
