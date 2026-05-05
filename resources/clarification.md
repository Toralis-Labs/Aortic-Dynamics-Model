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
- STEP4 future preferred root wrapper/name: `step4_evar_geometry_measurements.py`
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

- Conceptual STEP4 scope is `STEP4_EVAR_GEOMETRY_MEASUREMENTS`.
- STEP4 measures aortic neck, iliac, common iliac, external iliac, internal iliac,
  renal-to-internal-iliac path length, and access-vessel lumen geometry from STEP3 named
  anatomy.
- STEP4 writes grouped machine-readable EVAR lumen-geometry measurements from STEP3
  named anatomy.
- Existing `step4_infrarenal_neck.py` is only a compatibility-wrapper name until the
  implementation is updated; it is not the conceptual scope of STEP4.
- Future implementation should live under `src/step4/evar_geometry_measurements.py`.
- Future preferred root wrapper/name is `step4_evar_geometry_measurements.py`.
- STEP4 core outputs are `step4_measurements.json` and
  `step4_evar_geometry_regions.vtp`.
- Unmeasurable individual values use object status, not missing raw numbers.
- Missing required anatomy for a measurement group marks that group as
  `unmeasurable` or `requires_review`.
- Missing internal iliac branches do not globally fail STEP4.
- Standard AAA/Conformable aortic neck and iliac measurements can still succeed without
  internal iliac branches.
- IBE-specific internal iliac and renal-to-internal-iliac fields become
  `unmeasurable` or `requires_review` if internal iliac branches are missing.
- STEP4 must not measure tissue variables, clinical intake variables, IFU/device
  matching, catalogue matching, component selection, or final suitability.
- Global failure applies only if STEP3 inputs are missing/unusable or required
  aortic/iliac anatomy is so incomplete that STEP4 cannot produce a trustworthy
  contract.

## Demo Face Map

The active demo face map is `0044_H_ABAO_AAA/face_id_to_name.json`.
Known required entry: face `2` maps to `abdominal_aorta_inlet`.
