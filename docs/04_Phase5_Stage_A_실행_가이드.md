# Phase 5 — Stage A 실행 가이드

상태: **IMPLEMENTED / AWAITING USER SMOKE TEST AND TRAINING**

Stage A는 Fixed Pilot Manifest의 `AI-Hub`, `pilot_selected=true`, `valid=true`,
`viewpoint_3`만 읽습니다. 각 video의 3개 frame에 Phase 4 MediaPipe ROI와 full-frame
fallback을 적용한 뒤 224×224 resize, ImageNet normalization을 수행합니다. Validation에는
random augmentation이 없습니다.

```bash
.venv/bin/python -m scripts.run_stage_a smoke --batch-size 8
.venv/bin/python -m scripts.run_stage_a train --batch-size 8
.venv/bin/python -m scripts.run_stage_a evaluate --checkpoint "<TRAIN 출력의 best checkpoint 경로>" --batch-size 8
.venv/bin/python -m scripts.run_stage_a verify-encoders --checkpoint "<TRAIN 출력의 best checkpoint 경로>"
```

OOM이면 architecture나 데이터를 바꾸지 않고 `--batch-size 4`, 이후 `--batch-size 2`로만
낮춥니다. Best checkpoint 기준은 validation video-level Macro-F1이며, 동률이면 낮은 val loss,
그마저 동률이면 먼저 나온 epoch를 유지합니다. Stage A의 음수는 AI-Hub auxiliary visual
class이므로 최종 ETRI 물 마시기 성능으로 해석하지 않습니다.

각 본 학습은 Working Root의 `checkpoints/stage_a/<run_id>/`에 `best.pt`,
`history.json`, `effective_config.json`을 생성합니다. CLI override를 병합한 effective 값이
세 파일과 MLflow에 동일하게 기록됩니다. MediaPipe detector 안전성 때문에 `num_workers=0`만
지원합니다. Frozen feature block은 BatchNorm running statistics까지 고정합니다.

MLflow experiment는 `stage_a_visual_encoder`, artifact root는
`runtime/mlflow/artifacts/stage_a`입니다. Epoch별 metric과 best/final metric을 구분하고,
best confusion matrix, history, effective config, checkpoint를 artifact로 기록합니다.
