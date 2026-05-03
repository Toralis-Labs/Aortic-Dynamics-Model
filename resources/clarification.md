# Clarification And Locked Decisions

## Locked Decisions

- Focus is STEP2 through STEP5 while preserving STEP1 behavior.
- There is no STEP6 or STEP7 in the active implementation scope.
- Each step has core outputs for downstream execution and optional debug outputs for
  diagnosis.
- Each step writes exactly one main JSON contract.
- Status values are limited to `success`, `requires_review`, and `failed`.

## Script Names

- STEP2: `step2_geometry_contract.py`
- STEP3: `step3_naming_orientation.py`
- STEP4: `step4_infrarenal_neck.py`
- STEP5: `step5_pipeline_manifest.py`

## STEP2

- Aortic inlet is the start.
- Abdominal aorta end just before bifurcation is the end.
- STEP2 must fail if either cannot be resolved trustworthily.
- STEP2 core outputs are `segmentscolored.vtp`, `aorta_centerline.vtp`, and
  `step2_geometry_contract.json`.
- STEP2 optional outputs are limited to `boundary_debug.vtp` and `step2_debug.json`.
- STEP2 uses the face map as its anatomy registry, follows named outlet tunnels through
  the STEP1 graph, and preserves topology-plus-terminal segments.
- Fallback policy: `requires_review` above 5% fallback, `failed` above 15% fallback, and
  `requires_review` when fallback affects priority regions.

## STEP3

- STEP3 consumes STEP2 geometry as truth and adds semantic names/landmarks.
- Priority vessels are abdominal aorta trunk, left/right common iliac, left/right
  internal iliac, left/right external iliac, and left/right renal arteries.
- Non-priority unmatched vessels should warn rather than fail.
- Low-confidence priority vessel starts should return `requires_review`.

## STEP4

- STEP4 writes grouped machine-readable measurements.
- Unmeasurable individual values use `{ "status": "unmeasurable" }`.
- Missing required iliac diameter series is a hard failure once STEP4 is implemented.

## Demo Face Map

The active demo face map is `0044_H_ABAO_AAA/face_id_to_name.json`.
Known required entry: face `2` maps to `abdominal_aorta_inlet`.
