# 데이터 경로 및 Data Readiness 검증 기록

- 프로젝트: `medication-drinking-action-recognition`
- 검증일: 2026-08-29
- 검증 성격: Phase 0 Git Repository Baseline 전 read-only Raw Data / Working Root readiness audit
- 상위 기준: `docs/00_Pilot_Design_Baseline.md`

---

## 1. 검증 결과 요약

```text
AI-Hub: READY WITH WARNING
ETRI: READY WITH WARNING
Working: READY
전체: DATA READY WITH WARNINGS
```

Pilot에 필요한 Raw Data와 Working Root는 존재하고 서로 안전하게 분리되어 있다. Raw Data는 수정하지 않았으며, Working Root에서만 소형 임시 파일 write/delete test를 수행했다.

---

## 2. 확정 경로

```text
WSL Project Root
/home/user/projects/medication-drinking-action-recognition

AI-Hub Raw Root
/mnt/d/AI 도약과정/데이터/data_raw/057.일상생활_영상_데이터/01.데이터

ETRI Raw Root
/mnt/d/AI 도약과정/데이터/data_raw/고령자 일상행동인식 3차원 영상 데이터셋(리빙랩)

SSD Working Root
/mnt/d/AI 도약과정/데이터/data_workspace/medication_drinking_action_data
```

Design Baseline, 개발환경 기록, AGENTS.md, README.md, STATUS.md 및 path config 사이에 충돌하는 실제 경로는 발견되지 않았다.

---

## 3. AI-Hub Data Readiness

확인 항목:

- `1.Training` / `2.Validation` 존재
- 라벨링데이터 `viewpoint_1` / `viewpoint_3` 존재
- `viewpoint_3` Training / Validation 각 105개 class directory
- metadata XLSX 존재
- 행동분류 번호 대응 XLSX 존재
- JSON / JPG 존재 및 대표 JSON parsing 성공
- `Drink_bever`, `Take_pills`, `Drink_alcohol` class 확인

핵심 class 실측:

| Class | Training JSON | Validation JSON | 합계 | JPG |
|---|---:|---:|---:|---:|
| `Drink_bever` | 107 | 11 | 118 | 354 |
| `Take_pills` | 99 | 12 | 111 | 333 |
| `Drink_alcohol` | 95 | 12 | 107 | 321 |

Warning:

- 18,420 JSON / 55,260 JPG 전체 count는 대규모 scan을 피하기 위해 이번 audit에서 재실행하지 않았다.
- Phase 2 Full Candidate Inventory에서 전체 count와 JSON/file pairing을 최종 검증한다.

---

## 4. ETRI Data Readiness

Batch B 실측:

| 항목 | 수량 |
|---|---:|
| Participant directory `P201`~`P230` | 30 |
| RGB MP4 | 6,589 |
| A003 복약 | 119 |
| A004 물 마시기 | 120 |

A003과 A004는 모두 30명 participant 전원에게 존재한다.

대표 A003 RGB metadata:

```text
codec: H.264
resolution: 1920x1080
duration: 6.65 seconds
```

Skeleton과 Body Index는 문서 기준대로 ZIP 상태로 존재한다. Audit에서 압축을 해제하지 않았다.

Warning:

```text
RGB Videos/P205/A053_P205_G011_H120.mp4
size: 232,133 bytes
```

상기 A053 비정상 RGB는 Raw에서 삭제하지 않는다. Phase 2 inventory validation에서 `valid=false` 및 exclusion reason을 기록하고 `pilot_selected=false`로 처리한다.

---

## 5. Working Root Readiness

Audit 시점에 Working Root는 비어 있었다.

```text
file: 0
symlink: 0
예상치 못한 Raw 복사본: 없음
```

26-byte 임시 파일을 생성하고 즉시 삭제하여 write/delete를 검증했다. 임시 파일 잔재는 없다.

향후 저장 대상:

```text
manifests
preprocessing
ROI results
frame cache
embeddings
checkpoints
self_recorded/pipeline_check
```

---

## 6. Raw Preservation 검증

Canonical/resolved path 기준:

- AI-Hub Raw != Working
- ETRI Raw != Working
- Working은 Raw Root의 하위 경로가 아님
- Raw Root는 Working의 하위 경로가 아님
- 세 root는 symlink가 아님
- Raw를 output으로 사용하는 기존 code/config는 없음

현재 경로 구조는 다음 흐름을 지원한다.

```text
Raw (read only)
→ Full Candidate Inventory
→ Fixed Pilot Manifest
→ preprocessing
→ SSD Working Root
```

---

## 7. 후속 구현 안전장치

Phase 1/2 path loader와 preprocessing code에 다음을 반영한다.

- input/output root 분리
- resolved path 기준 동일성과 포함 관계 검증
- symlink resolve 후 재검증
- output이 Raw 하위인 경우 명시적 실패
- overwrite 기본 금지
- 기존 파일 충돌 시 명시적 오류
- manifest에 `root_key + relative_path` 저장
- source-to-derived mapping 기록
- invalid sample은 삭제 대신 validation/exclusion flag로 보존

---

## 8. 최종 판정

```text
DATA READY WITH WARNINGS
```

경고 항목은 Phase 2 Full Candidate Inventory validation에서 처리할 수 있으며, Phase 0 Git Repository Baseline 진행을 차단하는 blocker는 아니다.
