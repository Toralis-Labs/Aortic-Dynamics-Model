# Implementation Plan

## Summary

Clean the repository into a small, contract-driven pipeline before expanding STEP2/STEP4
behavior. The active target is `STEP1 -> STEP2 -> STEP3 -> STEP4 -> STEP5`, with STEP2
as the strict geometry-authoring contract.

## Current Work

- Preserve `step1segment.py` behavior.
- Keep root scripts as thin CLI wrappers for the package modules.
- Keep active implementation under `src/common`, `src/step2`, `src/step3`, and future
  `src/step4` / `src/step5`.
- Keep only the demo input `0044_H_ABAO_AAA/0156_0001.vtp` and
  `0044_H_ABAO_AAA/face_id_to_name.json`.
- Ignore generated outputs and local tool environments.

## STEP2 Stabilization

- Emit canonical contract fields only: `step_name`, `step_status`, `input_paths`,
  `output_paths`, `upstream_references`, `coordinate_system`, `units`, `aorta_start`,
  `aorta_end`, `aorta_trunk`, `segment_summary`, `boundary_summary`,
  `fallback_details`, `qa`, and `warnings`.
- Keep default outputs minimal: `segmentscolored.vtp`, `aorta_centerline.vtp`, and
  `step2_geometry_contract.json`.
- Write `boundary_debug.vtp` and `step2_debug.json` only when `--write-debug` is passed.
- Preserve the current surface-seeded assignment and ostium boundary approach unless a
  concrete STEP2 issue is found.

## STEP3 / STEP4 / STEP5

- STEP3 must consume canonical STEP2 contracts without legacy-mode dependence.
- STEP4 remains a scaffold until the measurement logic is ported deliberately.
- STEP5 validates required outputs and reports `success`, `requires_review`, or `failed`.

## Verification

- Static import checks for all wrappers and active packages.
- Smoke run STEP1, STEP2, and STEP3 on the checked-in demo input when VTK/VMTK are
  available.
- Confirm generated outputs do not appear in `git status`.
