# 057.일상생활 영상 데이터 - 데이터셋 구조 안내

AI-Hub **「일상생활 영상 데이터」**(영문명 *Personal Informative Visual Lifelogging Data For AI Learning*) 중
로컬에 내려받은 **라벨링데이터(프레임 이미지 + JSON 어노테이션) + 메타/원천 보조파일**의 구성을 정리한 문서입니다.

- 데이터 출처: <https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=648>
- 이 문서 = 폴더 구조 · 파일명 규칙 · 각 데이터 설명 · 라벨 저장 방식 · 행동 클래스 목록 · 사용 시 주의사항
- 원본(전체 데이터셋) 사양은 `Guide/데이터 설명서 및 활용가이드/데이터설명서.pdf`, 클래스 번호 대응표는 `1.Training/원천데이터/일상행동_행동분류_번호_비교.xlsx` 참고

---

## 1. 데이터셋 개요 (출처 페이지 · 데이터설명서 기준)

| 항목 | 내용 |
|---|---|
| 국문명 / 영문명 | 일상생활 영상 데이터 / Personal Informative Visual Lifelogging Data For AI Learning |
| 구축 목적 | 개인맞춤형 VR/AR/MR 서비스 제공, 궁극적으로 실효적 생활 개선에 기여하는 AI 학습용 데이터 구축 |
| 활용 분야 | 메타버스 3D 맵 구현, 맞춤형 실감(쇼핑) 서비스, 일상행동 인식(Action Recognition) |
| 데이터 성격 | 한국인의 일상생활을 **시간유형(필수·여가·의무)** 및 **인구통계 속성(성별·연령 등)** 기준으로 분류한 1인칭/3인칭 영상 |
| 행위 체계 근거 | ① 통계청 생활시간조사 177개 행위 ② 유사 선행연구 행위체계 ③ 전문가 자문 ④ 연령구간별 파일럿 조사 |
| 참고 모델 | UCF101, CNN R(2+1D) + IG65M |
| 구축년도 / 갱신년월 | 2021년 / 2023-05 |
| 수행기관 | 총괄·수집정제 ㈜메트릭스, 수집정제 ㈜더바이럴, 가공·검수 ㈜딥네츄럴, 모델링 울산과학기술원 |
| 촬영 장비 | GoPro 및 스마트폰(Android/iOS) — JSON `info.camera_info` 값은 `gopro`, `cellular phone-AOS`, `cellular phone-IOS` 3종 |
| 영상 사양 | Full HD, 프레임 이미지 1920×1080 |

### 1-1. 원본(전체 데이터셋) 구축 규모 — 출처 페이지 표

| 데이터 종류 | 규모 | 크기 | 형식 |
|---|---|---|---|
| 일상생활 행위 영상 | 20,517건(7,000시간) | 55 TB | mp4 |
| 주요 객체 이미지 | 61,551건 | 184.7 GB | jpg |
| 일상생활 행위 텍스트 | 20,517건 | 120 MB | CSV |
| 학습용(어노테이션) 데이터 | 20,517세트 | 20.5 GB | json |

- 피험자: 일반인 **200명**
- 연령구간: age1(10~19), age2(20~39), age3(40~65), age4(65세 이상) / 성별 male·female / 지역 17개 시·도 / 촬영 시점 1인칭·3인칭
- **로컬에 받은 것은 이 원본의 부분집합**입니다(아래 6장). 원본 mp4·CSV 텍스트는 로컬에 없습니다.

### 1-2. 갱신 이력

| 버전 | 일자 | 변경 내용 |
|---|---|---|
| 1.0 | 2022-07-28 | 데이터 최초 개방 |
| 1.1 | 2022-09-22 | 원천데이터 수정 |
| 1.2 | 2022-11-04 | 원천데이터 카테고리 비교군 파일 추가 (`일상행동_행동분류_번호_비교.xlsx`) |
| 1.3 | 2023-05-26 | 메타데이터 추가 (`메타데이터_230525_add`) |

> **라이선스**: 출처 페이지에 별도 라이선스 문구는 명시돼 있지 않으며, AI-Hub 공통 이용약관(비영리·연구 목적 이용, 재배포 제한 등)이 적용됩니다. 실제 이용 조건은 다운로드 시 동의한 AI-Hub 약관을 확인하세요.

---

## 2. 로컬 최상위 디렉토리 구조

```
01.데이터/
├── 1.Training/
│   ├── 라벨링데이터/
│   │   ├── viewpoint_1/                 (1인칭 시점, 행동 카테고리 폴더 76개)
│   │   │   └── TLCategoryNNN/
│   │   │       └── age{1..4}/
│   │   │           └── {male|female}/
│   │   │               └── {video_id 5자리}/
│   │   │                   ├── C{NNN}_A{a}_{M|F}{video_id}_V1_image_{f}.jpg   (프레임 3장)
│   │   │                   └── C{NNN}_A{a}_{M|F}{video_id}_V1.json            (어노테이션 1건)
│   │   └── viewpoint_3/                 (3인칭 고정 시점, 행동 카테고리 폴더 105개, 파일명 _V3_)
│   ├── 메타데이터_230525_add/
│   │   └── metadata/
│   │       └── (2021-1-35-57) metadata.csv.xlsx    (시트 3개: Issues / video_meta / video_meta_수정전)
│   └── 원천데이터/
│       └── 일상행동_행동분류_번호_비교.xlsx        (폴더번호 ↔ AI허브 업로드번호 대응표, 122행)
├── 2.Validation/
│   └── 라벨링데이터/
│       ├── viewpoint_1/                 (VLCategoryNNN 76개)
│       └── viewpoint_3/                 (VLCategoryNNN 105개)
└── 데이터_구조_안내.md                   (이 문서)
```

- Training / Validation은 **완전히 동일한 폴더 규칙**을 따르며, 카테고리 폴더 접두어만 `TL`(Training) / `VL`(Validation)로 다릅니다. 파일명 접두어는 양쪽 모두 `C`(Category)입니다.
- `메타데이터_230525_add`, `원천데이터`는 **Training 하위에만** 존재하며, 내용은 Training/Validation 구분 없이 데이터셋 전체를 다룹니다.
- 원본 mp4 영상, 행위 텍스트 CSV는 로컬에 **포함되어 있지 않습니다**(위 1-1의 20,517건 mp4/CSV는 원본 규모 설명일 뿐).

---

## 3. 폴더별 상세 설명

### 3-1. 라벨링데이터 (`1.Training/라벨링데이터`, `2.Validation/라벨링데이터`)

학습에 실제로 사용하는 본체 데이터. 영상에서 샘플링한 **정지 프레임(jpg) 3장**과, 해당 영상 1건의 **행동·피험자 정보 + 객체 bbox를 담은 어노테이션(json) 1개**가 최소 단위입니다.

#### (a) viewpoint_1 vs viewpoint_3 — 촬영 시점

각 JSON의 `video.meta.viewpoint` 값(1 또는 3)과 파일명의 `_V1_` / `_V3_` 표기로 구분되며, **카메라의 촬영 시점(피험자 기준 위치)** 차이입니다. `viewpoint_2`는 데이터셋에 존재하지 않습니다(시점 값은 1, 3만 사용).

| 구분 | 시점 | 특징 |
|---|---|---|
| `viewpoint_1` | 1인칭 | 피험자가 착용/휴대한 카메라로 촬영. 자신의 손·발 등 신체 동작이 화면 중앙에 클로즈업 |
| `viewpoint_3` | 3인칭 고정 | 벽면·모서리 등에 고정 설치한 카메라로 촬영. 방 전체와 피험자 전신이 넓게 보임 |

- **촬영 장비 ≠ 시점**: 두 시점 모두 GoPro와 스마트폰(AOS/iOS)이 섞여 사용되었습니다. 메타 xlsx의 `record_device`는 대부분 `Gopro`로 기록되어 JSON `camera_info`와 완전히 일치하지는 않습니다.
- **카테고리 커버리지 차이**:
  - `viewpoint_3`: 105개 행동 클래스 전부 보유
  - `viewpoint_1`: 76개 클래스만 보유하며, **이 76개는 viewpoint_3의 105개에 대한 완전한 부분집합**
  - 즉 모든 행동은 3인칭으로 촬영되었고, 그중 76개 행동만 1인칭 촬영이 추가로 존재. **1인칭 폴더가 없는 29개 클래스**(폴더명 번호): 1~8, 17~30, 33, 34, 97, 98, 111, 112, 114 (먹기/씻기/걷기·달리기·점프 등 이동 동작, 화장, 헬스 동작 일부)

#### (b) TLCategory / VLCategory 폴더 = 행동 클래스

`TLCategory + 3자리` (예: `TLCategory003`, Validation은 `VLCategory003`)는 **하나의 일상 행동**을 뜻합니다. 이 3자리 번호는 **폴더명 행동분류 번호**(= 파일명 `C###`의 번호, `일상행동_행동분류_번호_비교.xlsx`의 A열)이며, AI허브 다운로드 화면의 클래스 번호(B열)와는 다릅니다(→ 5-2 참조).

- 로컬에 존재하는 폴더 번호: `001~122` 중 **105개**(003=약을 먹는다, 010=발을 씻는다 등). 46·72·92·96·100·102~107·109·110·113·116·117·120번은 로컬에 없음(원본에서도 AI허브 미업로드 클래스).

#### (c) 카테고리 폴더 내부 구조

```
TLCategory003/
├── age1/                         ← 연령대 그룹 (age1~age4)
│   ├── male/
│   │   └── 14303/                ← video_id (촬영 세션, 5자리 zero-pad)
│   │       ├── C003_A1_M14303_V3_image_16.jpg
│   │       ├── C003_A1_M14303_V3_image_72.jpg
│   │       ├── C003_A1_M14303_V3_image_84.jpg
│   │       └── C003_A1_M14303_V3.json
│   └── female/
│       └── 10785/ ...
├── age2/  ├── age3/  └── age4/    (동일 구조; 해당 조합이 없으면 폴더 자체가 없음)
```

계층: **행동 클래스 → 촬영 시점(상위 viewpoint 폴더에서 이미 분리) → 연령대(age1~4) → 성별(male/female) → video_id**.

#### (d) 파일명 규칙

```
C{NNN}_A{a}_{G}{video_id}_V{v}[_image_{f}].{ext}
```

| 토큰 | 의미 | 예 |
|---|---|---|
| `C{NNN}` | 행동 클래스 번호(폴더명 번호, 3자리) | `C003` |
| `A{a}` | 연령대 1~4 | `A1` |
| `{G}` | 성별 `M`(male) / `F`(female) | `M` |
| `{video_id}` | 촬영 세션 ID (5자리, 폴더명과 동일) | `14303` |
| `V{v}` | 촬영 시점 1 / 3 | `V3` |
| `_image_{f}` | (jpg만) 원본 영상 내 샘플 프레임 인덱스 | `_image_72` |
| `{ext}` | `jpg`(프레임) / `json`(어노테이션) | |

- 영상(=json) 1건 = **jpg 3장 + json 1개**가 기본 저장 단위입니다.
- `_image_{f}`의 `f`는 연속 번호가 아니라 하이라이트 구간에서 뽑은 프레임의 위치 인덱스(관측 범위 대략 1~400).

### 3-2. 메타데이터_230525_add

2023-05-25 추가 배포된 보정 메타데이터. `(2021-1-35-57) metadata.csv.xlsx` 한 파일이며 **시트 3개**로 구성됩니다(파일명은 csv지만 실제로는 xlsx).

| 시트명 | 행 수 | 내용 |
|---|---|---|
| `Issues` | 30여 건 | 일부 영상의 `video_date` / `video_time` / `video_length` 값 오류에 대한 **수정 전·후 비교 로그**. `0000-00-00` 같은 기본값이 정상값으로 교정된 케이스 등 |
| `video_meta` | 20,517 | **정정 반영 후** 전체 영상 메타 테이블 |
| `video_meta_수정전` | 20,517 | 정정 반영 전(원본) 메타 테이블. `video_meta`와 `Issues`에 열거된 행에서만 값이 다름 |

`video_meta` 컬럼(19개):
`Filename, video_id, category_id, actor_id, viewpoint, gender, height, age, family_number, job, region, place, interact_person, interact_ICT, explan, video_date, video_time, video_length, record_device`

- `category_id`는 파일명의 `C###`와 동일한 **폴더명 번호 체계**(예: `C063…` → `category_id 63`).
- 이 테이블 하나로 개별 json을 열지 않고도 전체 영상 목록·피험자 속성·행동 설명을 조회할 수 있습니다.
- 단, **원본 전체(20,517건) 기준**이므로 로컬에 실제 존재하는 영상보다 큰 목록이며, `record_device`·CSV 텍스트 등 로컬에 없는 정보도 포함합니다.

### 3-3. 원천데이터

`일상행동_행동분류_번호_비교.xlsx` 한 파일(Sheet1, 122행 + 헤더). 컬럼 3개:

| 컬럼 | 의미 |
|---|---|
| A. `폴더명 행동분류` | 로컬 폴더/파일명이 쓰는 번호 (1~122) |
| B. `AI허브 업로드 행동분류` | AI-Hub 다운로드 화면·데이터설명서가 쓰는 번호 (1~105, 미업로드 클래스는 공란) |
| C. `행동분류` | 행동 이름(국문) |

- 122개 행동 중 **17개는 B열이 공란** = AI-Hub에 업로드되지 않은 클래스(46 공부하다, 72 발표를 하다, 92 파충류 사육, 96 클라이밍, 100 볼링, 102~107·109·110 구기·투척, 113 골프, 116·117·120 만들기 등). 나머지 **105개가 실제 배포·로컬 보유 클래스**.
- 1~45번은 A=B로 일치하지만 **46번부터 어긋납니다**(공란 클래스만큼 B가 밀림). 카테고리를 찾을 때는 반드시 A열(폴더명 번호)을 기준으로 하세요.
- 폴더명은 "원천데이터"지만 **원본 mp4는 없고 이 대응표만** 들어 있습니다.

---

## 4. JSON 어노테이션 파일 구조

`TLCategoryNNN` 폴더 최하단의 `C{NNN}_A{a}_{G}{video_id}_V{v}.json` 1개가 영상 1건에 대응합니다.

### 4-1. 실제 예시

```json
{
  "info": {
    "description": "daily Action Recognition dataset",
    "year": 2021,
    "video_resolution": "full HD",
    "camera_info": "cellular phone-AOS"
  },
  "categories": { "id": 9, "name": "Wash_hand", "SuperCategory": "daily Action Recognition" },
  "images": [
    { "id": "C009_A3_M12344_V1_image_122", "filename": "C009_A3_M12344_V1_image_122.jpg", "width": "1920", "height": "1080" },
    { "id": "C009_A3_M12344_V1_image_358", "filename": "C009_A3_M12344_V1_image_358.jpg", "width": "1920", "height": "1080" },
    { "id": "C009_A3_M12344_V1_image_4",   "filename": "C009_A3_M12344_V1_image_4.jpg",   "width": "1920", "height": "1080" }
  ],
  "video.meta": {
    "video_id": "12344", "category_id": 9, "actor": 131, "viewpoint": 1,
    "gender": "male", "height": 174, "age": 3, "family_number": 3,
    "job": "Self-employed", "region": "Gyeongnam", "place": "in",
    "interact_person": "alone", "interact_ict": "unuse",
    "explan": "손을 씻는 모습을 촬영함",
    "video_date": "2021-11-09", "video_time": "18:14:00", "video_length": 1277
  },
  "timeline": { "id": "618e4884a80cf96d5fff2a49", "start": 23, "end": 143 },
  "annotation": [
    { "id": 1, "image_id": "C009_A3_M12344_V1_image_122.jpg",
      "bbox": [[908,606],[1295,606],[1295,879],[908,879]], "obj_name": "손" }
  ]
}
```

### 4-2. 필드 정의 (데이터설명서 + 파일 확인)

| 경로 | 타입 | 설명 / 관측 값 |
|---|---|---|
| `info.description` | str | `"daily Action Recognition dataset"` |
| `info.year` | int | `2021` |
| `info.video_resolution` | str | `"full HD"` |
| `info.camera_info` | str | `gopro` / `cellular phone-AOS` / `cellular phone-IOS` |
| `categories.id` | int | 행동 클래스 번호 = 소속 폴더 번호와 일치 |
| `categories.name` | str | 클래스 영문명. 대체로 폴더당 1개지만 **일부 폴더에 2개 혼재**(63번: `Read_book`/`Study`, 101번: `Juggle_ball`/`Pitch_baseball`). 신뢰 기준은 `id`/폴더번호 |
| `categories.SuperCategory` | str | `"daily Action Recognition"` |
| `images[]` | list, 길이 3 | `id`, `filename`, `width`("1920"), `height`("1080") — 폭/높이는 **문자열**. 해상도는 모두 1920×1080(Full HD) |
| `video.meta.video_id` | str | 촬영 세션 ID(폴더명과 동일) |
| `video.meta.category_id` | int | `categories.id`와 동일 |
| `video.meta.actor` | int | **피험자 식별자**(데이터설명서 필드명 `actor_id`, 범위 1~200). `video_id`(촬영 세션)와 별개로 한 사람에게 부여 |
| `video.meta.viewpoint` | int | `1` / `3` — 소속 viewpoint 폴더와 일치 |
| `video.meta.gender` | str | `male` / `female` |
| `video.meta.height` | int | 신장(cm). 관측 범위 약 140~187. `150` 등 기본값으로 의심되는 값도 있음 |
| `video.meta.age` | int | **정수 1~4**(데이터설명서 표기는 `age1~age4`) |
| `video.meta.family_number` | int | 동거인 수. 관측 범위 1~7 |
| `video.meta.job` | str | 직업. `Office worker`, `Student`, `none` 등 코드값 + **`강사`·`작가`·`기타`·`6급공무원합격후발령대기중` 같은 한글 자유기입 혼재** → 정제 필요 |
| `video.meta.region` | str | 17개 시·도 영문(`Seoul`, `Gyeonggi`, `Busan`, `Jeju` …) |
| `video.meta.place` | str | `in`(실내) / `out`(실외) |
| `video.meta.interact_person` | str | `alone` / `partner` (동반자 유무) |
| `video.meta.interact_ict` | str | `use` / `unuse` (촬영 중 ICT 기기 사용 여부) |
| `video.meta.explan` | str | 영상 내용 한글 자유 서술. 품질 편차 큼("손을씻었습니다" ~ 여러 문장) |
| `video.meta.video_date` | str | 촬영일 `YYYY-MM-DD`. 대부분 2021-06~2021-11. **`0021-…`, `2121-…` 등 오타 이상치 소수 존재** |
| `video.meta.video_time` | str | 촬영 시각. 실제로는 `HH:MM:SS`(데이터설명서 표기 `MM:SS`) |
| `video.meta.video_length` | int | 영상 총 길이(**초**). 영상마다 다름(관측 범위 대략 240~2,320초) |
| `timeline.id` | str | 하이라이트 구간 식별자(MongoDB ObjectId 형태) |
| `timeline.start` / `timeline.end` | int | **행동 하이라이트 구간의 시작/종료 시각(초)**. 전체 영상 중 해당 행동이 일어난 구간 |
| `annotation[]` | list, 보통 길이 3 (가끔 4) | 샘플 프레임별 주요 객체 bbox |
| `annotation[].id` | int | 라벨 식별자(1~3) |
| `annotation[].image_id` | str | 대상 `images[].filename` |
| `annotation[].bbox` | list | **4개 꼭짓점 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`** (좌상→우상→우하→좌하, 축정렬 사각형). 일부는 프레임 전체(0,0)~(1919,1079)를 덮음 |
| `annotation[].obj_name` | str | 객체 한글 이름. **자유 텍스트**(수백 종), 쉼표로 복수 객체 표기(예: `"알약, 약봉지"`, `"창문, 닦는 도구/걸레"`) |

> **데이터설명서 대비 실제 차이**: ① `images[].filename`(설명서는 `file_name`) ② `video.meta.actor`(설명서는 `actor_id`) ③ `age`는 정수(설명서는 `age1~4`) ④ `video_time`은 `HH:MM:SS` ⑤ 폴더번호는 최대 122까지(설명서 범위 표기는 `1~112` 또는 `1~105`).

---

## 5. 행동 라벨이 기록·저장되는 방식

### 5-1. 라벨이 저장되는 위치 (같은 정보가 여러 곳에 중복 기록됨)

| 라벨 종류 | 저장 위치 |
|---|---|
| **행동 클래스** | ① 폴더 경로 `viewpoint_*/TLCategoryNNN` ② 파일명 `C{NNN}` ③ JSON `categories.id` ④ JSON `categories.name`(영문) ⑤ JSON `video.meta.category_id` ⑥ 메타 xlsx `category_id` — ①③⑤⑥은 서로 일치, ④만 일부 폴더에서 2종 혼재 |
| **촬영 시점** | 폴더 `viewpoint_1/3` = 파일명 `_V1/_V3` = JSON `video.meta.viewpoint` |
| **행동 발생 구간(하이라이트)** | JSON `timeline.start` / `timeline.end` (초 단위). 전체 영상 중 실제 행동이 일어난 구간 |
| **주요 객체 위치** | JSON `annotation[].bbox` + `obj_name` (샘플 프레임 3장 각각에 대해) |
| **피험자 속성 라벨** | JSON `video.meta.*` (성별·연령·지역·직업·동거인·실내외·동반자·ICT) + 메타 xlsx |

- 하이라이트 구간은 데이터설명서 기준 **"1개 영상에 3명이 각각 하이라이트를 표시 → 2인 교차검수 → 합산점수가 높은 구간 선정"** 방식으로 라벨링되었습니다.
- bbox는 작업 가이드라인에 따라 **행위별 주요 객체**에 대해 부여하며, 한 프레임에 2개 이상 객체를 동시에 표기할 수 있습니다(일부 프레임은 객체 2개).
- **프레임 단위 행동 클래스 라벨은 별도로 없습니다.** 클래스는 영상(=폴더/json) 단위이고, 프레임 라벨은 객체 bbox뿐입니다.

### 5-2. 행동 클래스 번호 체계 3종 (혼동 주의)

| 체계 | 사용처 | 범위 |
|---|---|---|
| **폴더명 번호** (A) | 로컬 폴더 `TLCategoryNNN`, 파일명 `C###`, JSON `categories.id`/`category_id`, 메타 xlsx `category_id` | 1~122 중 105개 |
| **AI허브 업로드 번호** (B) | AI-Hub 다운로드 화면, 데이터설명서 클래스표의 `N` | 1~105 |
| `categories.name` | JSON 내 영문 라벨 | 문자열(폴더당 1~2종) |

→ **로컬 데이터 안에서는 전부 A(폴더명 번호)로 통일**되어 있으므로, 외부 문서(설명서)의 N번호와 대조할 때만 `일상행동_행동분류_번호_비교.xlsx`로 변환하면 됩니다.

### 5-3. 전체 행동 클래스 목록 (로컬 보유 105종, 폴더명 번호 순)

`categories.name`은 각 클래스 JSON에서 관측되는 영문 라벨, `AI허브#`는 `일상행동_행동분류_번호_비교.xlsx` B열입니다.

| 폴더# | 행위(국문) | categories.name | AI허브# |
|--:|---|---|--:|
| 1 | 음식을 먹는다 | Eat_food | 1 |
| 2 | 음료를 마신다 | Drink_bever | 2 |
| 3 | 약을 먹는다 | Take_pills | 3 |
| 4 | 담배를 피다 | Smoke | 4 |
| 5 | 술을 마신다 | Drink_alcohol | 5 |
| 6 | 얼굴을 씻는다 | Wash_face | 6 |
| 7 | 이를 닦는다 | Brush_teeth | 7 |
| 8 | 면도를 하다 | Shave_beard | 8 |
| 9 | 손을 씻는다 | Wash_hand | 9 |
| 10 | 발을 씻는다 | Wash_feet | 10 |
| 11 | 손빨래를 하다 | Handwash | 11 |
| 12 | 자동차를 닦는다(세차한다) | Wash_car | 12 |
| 13 | 식기를 닦는다 | wash_dish | 13 |
| 14 | 거울을 닦는다 | Wip_mirror | 14 |
| 15 | 창문을 닦는다 | Clean_window | 15 |
| 16 | 승용차를 탄다 | take_car | 16 |
| 17 | 걷는다 | Walk | 17 |
| 18 | 바닥을 기다 | Crawl | 18 |
| 19 | 위아래로 점프하다 | Jump | 19 |
| 20 | 달리다 | Run | 20 |
| 21 | 물건 등을 던지다 | Throw | 21 |
| 22 | (움직이는)물체 등을 받다 | Catch | 22 |
| 23 | 화장하다 | Makeup | 23 |
| 24 | 눈화장을 하다 | eyemakeup | 24 |
| 25 | 립스틱을 바르다 | lipstick | 25 |
| 26 | 마사지하다(머리 또는 얼굴) | Massage | 26 |
| 27 | 팩하다 | Sheet_mask | 27 |
| 28 | 머리를 손질하다 | hairstyling | 28 |
| 29 | 머리를 말리다 | hairdry | 29 |
| 30 | 머리를 자르다 | haircut | 30 |
| 31 | 매니큐어를 바른다 | manicure | 31 |
| 32 | 옷을 입는다 | cloth | 32 |
| 33 | 모자를 쓴다 | Wear_cap | 33 |
| 34 | 장신구를 착용하다 | Wear_acc | 34 |
| 35 | 신발을 신는다 | shoes | 35 |
| 36 | 재활용 쓰레기를 분리한다 | Separate_trash | 36 |
| 37 | 의류를 버린다 | Throw_clothes | 37 |
| 38 | 쓰레기를 버린다 | Throw_garbage | 38 |
| 39 | 재료를 씻는다 | wash_ingredient | 39 |
| 40 | 재료를 자른다 | Cut_ingredient | 40 |
| 41 | 조리한다 | Cook | 41 |
| 42 | 빵(쿠키)를 굽는다 | Bake_bread | 42 |
| 43 | 반죽하다 | Mix_batter | 43 |
| 44 | 반죽을 위로 던지다 | Toss_pizza | 44 |
| 45 | 식탁에 상을 차린다 | Set_table | 45 |
| 47 | 그림을 그린다 | Draw | 46 |
| 48 | 글씨를 쓴다 | Write | 47 |
| 49 | 배달음식을 전해받는다 | Receive_food | 48 |
| 50 | 카드게임을 하다 | Play_card | 49 |
| 51 | 택배를 전해받는다 | Receive_package | 50 |
| 52 | 아이와 논다 | Play_child | 51 |
| 53 | 아이를 씻긴다 | wash_baby | 52 |
| 54 | 아이의 머리를 묶는다 | Tie_hair | 53 |
| 55 | 아이와 산책을 하다 | Walk_child | 54 |
| 56 | 아이에게 밥을 먹인다 | Feed_child | 55 |
| 57 | 아이에게 옷을 입힌다 | Dress_child | 56 |
| 58 | 장난감을 정리하다 | Clean_toy | 57 |
| 59 | 책이나 서류를 정리하다 | Clean_book | 58 |
| 60 | 옷을 정리하다 | Organize_clothes | 59 |
| 61 | 그릇을 정리하다 | Arrange_bowl | 60 |
| 62 | TV를 본다 | Watch_tv | 61 |
| 63 | 책(서류)를 본다 | Read_book / Study | 62 |
| 64 | 신문을 본다 | Read_newsp | 63 |
| 65 | 그림을 본다 | See_painting | 64 |
| 66 | 동물을 관찰하다 | Observe_animal | 65 |
| 67 | 식물을 관찰하다 | Observe_plant | 66 |
| 68 | 곤충을 관찰하다 | Observe_insect | 67 |
| 69 | 전화통화를 하다 | Call | 68 |
| 70 | 사람과 대화를 하다 | conversation | 69 |
| 71 | 노래를 부른다 | Sing | 70 |
| 73 | 컴퓨터를 한다 | Play_computer | 71 |
| 74 | 휴대폰을 조작한다 | Operate_phone | 72 |
| 75 | 노트북을 조작한다 | Operate_laptop | 73 |
| 76 | 태블릿pc를 조작하다 | Operate_tablet | 74 |
| 77 | 게임기를 조작하다 | Operate_game | 75 |
| 78 | 세탁을 하다 | Laundry | 76 |
| 79 | 청소를 하다 | Clean_room | 77 |
| 80 | 다림질 하다 | Iron | 78 |
| 81 | 바느질 하다 | Sew | 79 |
| 82 | 운전하다 | Drive | 80 |
| 83 | 피아노를 친다 | Play_piano | 81 |
| 84 | 기타를 친다 | Play_guitar | 82 |
| 85 | 바이올린을 켜다 | Play_violin | 83 |
| 86 | 드럼을 치다 | Play_drum | 84 |
| 87 | 반려동물 목욕을 시킨다 | Bath_dog | 85 |
| 88 | 반려동물에게 밥을 준다 | Feed_dog | 86 |
| 89 | 반려동물 미용하다 | Groom_pet | 87 |
| 90 | 반려동물과 논다 | Play_dog | 88 |
| 91 | 반려동물과 산책하다 | Walk_dog | 89 |
| 93 | 식물에 물을 준다 | Water_plants | 90 |
| 94 | 줄넘기를 한다 | Jump_rope | 91 |
| 95 | 훌라후프를 하다 | Hulahoop | 92 |
| 97 | 푸쉬업을 하다 | Pushup | 93 |
| 98 | 스쿼트를 하다 | Squats | 94 |
| 99 | 복싱하다 | Punch | 95 |
| 101 | 공놀이를 하다 | Juggle_ball / Pitch_baseball | 96 |
| 108 | 테니스를 하다 | Swing_tennis | 97 |
| 111 | 요가를 하다 | yoga | 98 |
| 112 | 춤을 춘다 | Dance | 99 |
| 114 | 풍선을 불다 | Blow_ballon | 100 |
| 115 | 촛불을끄다 | Blow_candle | 101 |
| 118 | 피규어(프라모델)을 조립하다 | Assemble_pmodel | 102 |
| 119 | 액세서리를 만든다 | Make_acc | 103 |
| 121 | 망치질하다 | Hammering | 104 |
| 122 | 뜨개질을 하다 | Knit | 105 |

> 폴더명 번호 20번(`달리다`) 등 일부 국문명은 원본 xlsx에 전각 공백이 섞여 있어 표기를 정리했습니다.

---

## 6. 로컬 저장 규모

라벨링데이터는 아래 네 갈래로 나뉘어 저장돼 있습니다(폴더별 JSON 파일 수 = 영상 수).

| 구분 | viewpoint_1 | viewpoint_3 | 합계 |
|---|--:|--:|--:|
| Training | 6,617 | 9,786 | 16,403 |
| Validation | 815 | 1,202 | 2,017 |
| **합계** | **7,432** | **10,988** | **18,420** |

- 영상(JSON) 1건당 프레임 JPG 3장 → JPG 약 55,260장
- 행동 클래스: viewpoint_3 = 105종(전체), viewpoint_1 = 76종(viewpoint_3의 부분집합)
- 로컬 18,420건은 원본(출처 페이지 기준 20,517건)의 일부입니다. 원본에만 있고 로컬에 없는 영상·클래스가 존재합니다.

---

## 7. 데이터 사용 시 참고

1. **viewpoint 커버리지**: 29개 클래스는 `viewpoint_1` 폴더가 없습니다(3-1-(a) 목록). 1인칭 데이터만 필요한 경우 해당 클래스는 확보 불가.
2. **라벨 품질 편차**:
   - `categories.name`(영문)이 폴더당 2종 혼재하는 케이스(63·101번) → `id`/폴더번호를 기준으로.
   - `bbox` 일부가 프레임 전체를 덮음(사실상 위치 정보 없음에 가까움) → 객체 위치를 쓸 때 필터링 필요.
   - `obj_name`은 표준 클래스가 아닌 자유 텍스트(쉼표 복수표기).
   - `job`은 코드값 + 한글 자유기입 혼재, `height`는 기본값 의심 값 존재, `explan`은 서술 품질 편차 큼.
   - `video_date`에 `0021-…`, `2121-…` 등 오타 이상치 소수 존재 → 시계열로 쓸 경우 범위 필터.
3. **메타 xlsx는 원본 전체(20,517건) 기준**: 로컬보다 큰 목록이며 로컬에 없는 영상·`record_device`·CSV 텍스트 정보도 포함. 로컬 데이터와 조인할 때는 `Filename`/`video_id`로 교집합만.
4. **번호 체계 혼동 금지**: 로컬은 전부 "폴더명 번호(A)". 외부 문서(데이터설명서의 N)는 "AI허브 업로드 번호(B)". 변환은 `일상행동_행동분류_번호_비교.xlsx`.
5. **원본 mp4·행위 CSV 미포함**: 프레임 3장 외 원본 프레임이 필요하면 AI-Hub에서 원천 영상(mp4)을 별도 요청해야 함.
6. **메타 xlsx 시트명**: `Issues` / `video_meta` / `video_meta_수정전`이며, `video_meta`와 `video_meta_수정전`은 **동일본이 아니라 정정 전/후본**입니다.

---

## 8. 함께 제공되는 가이드 문서 (`057.일상생활_영상_데이터/Guide/`)

| 경로 | 내용 |
|---|---|
| `데이터 설명서 및 활용가이드/데이터설명서.pdf` | 데이터셋 개요·행위 105종 건수표·JSON 스키마 정의·라벨링 예시 |
| `데이터 설명서 및 활용가이드/데이터구축가이드라인.pdf` | 수집·가공·검수 기준 |
| `저작도구 설명서/`, `저작도구 소스코드.zip` | 어노테이션 저작도구 설치·사용 매뉴얼 및 소스 |
| `AI 모델 상세 설명서/`, `AI 모델 소스코드.zip` | 참조 모델(R(2+1D) 등) 환경 설치·사용 매뉴얼 및 소스 |
