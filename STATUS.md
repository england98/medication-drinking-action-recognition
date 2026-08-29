# STATUS.md

# Project Status

- 프로젝트: `medication-drinking-action-recognition`
- 현재 Phase: **Phase 2 완료 — Phase 3 Fixed Pilot Manifest 진입 준비**
- 최종 업데이트: 2026-08-29
- 문서 성격: **현재 작업 상태의 Single Source of Truth**

---

## 1. 현재 상태

개발환경, Data Readiness, Git Repository Baseline 구축이 완료되었다.
Phase 1 프로젝트 최소 골격과 안전한 path infrastructure 구축을 완료했다.
Phase 2의 AI-Hub Full Candidate Inventory 생성·validation 및 전체 scan을 완료했다.
AI-Hub Inventory 결과는 **PASS_WITH_WARNINGS**다.
ETRI Batch B Full Candidate Inventory 생성·validation 코드와 소수 MP4 smoke test를 완료했으며,
전체 scan 결과는 **PASS_WITH_WARNINGS**다. AI-Hub와 ETRI Full Candidate Inventory가 모두 확정되어
현재는 **Phase 3 Fixed Pilot Manifest 진입 준비 상태**다.

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
Fixed Pilot Manifest        ← 다음
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

현재 해야 할 작업:

- [ ] Phase 3 Fixed Pilot Manifest 구현 시 공통 selection validation 연결

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
Phase 2 — Full Candidate Inventory
```

Full Candidate Inventory 우선순위:

```text
1. AI-Hub Full Candidate Inventory
2. ETRI Batch B Full Candidate Inventory
3. inventory validation
4. split / fold
5. Fixed Pilot Manifest
```

모델 구현은 Manifest 단계 이후에 진행한다.

---

## 7. Pilot 전체 진행 순서

```text
Phase 0  Repository Baseline          완료
↓
Phase 1  프로젝트 골격            완료
↓
Phase 2  Full Candidate Inventory  완료
↓
Phase 3  Fixed Pilot Manifest       ← 다음
↓
Phase 4  ROI Preflight
↓
Phase 5  Stage A
↓
Phase 6  ETRI Embedding
↓
Phase 7  2×2 Ablation
↓
Phase 8  구조 선택
↓
Phase 9  Pilot deployment/check model
↓
Phase 10 Pipeline Integration
```

---

## 8. Blocker / Issue

현재 문서에 기록된 미해결 blocker는 없음.

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
