# STEP2 Clarification And Locked Decisions

## Locked Decisions

- This branch focuses only on STEP2 geometry-contract authoring.
- STEP1 implementation code is out of scope; only STEP1 output artifacts required by
  STEP2 are preserved.
- STEP2 writes exactly one main JSON contract.
- STEP2 status values are limited to `success`, `requires_review`, and `failed`.

## Script Names

- STEP2: `step2_geometry_contract.py`

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

## Demo Face Map

The active demo face map is `0044_H_ABAO_AAA/face_id_to_name.json`.
Known required entry: face `2` maps to `abdominal_aorta_inlet`.
