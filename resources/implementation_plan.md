# STEP2–STEP5 Implementation Plan

## Summary

This implementation plan refactors the current vascular prototype into a stricter step-based architecture while preserving STEP1 behavior.

The active goal is to build a minimal, contract-driven pipeline for:

- geometry authoring
- vessel naming and landmark registration
- infrarenal/iliac measurement
- orchestration and QA reporting

This refactor must reduce default outputs to only the artifacts fundamentally needed for downstream use.

## Scope Boundaries

### In scope

- preserve STEP1
- implement STEP2 through STEP5 using new script names
- create one main JSON contract per step
- reduce default outputs
- move old reference logic into the new architecture safely

### Out of scope

- DICOM segmentation implementation
- final IFU matching logic
- stent recommendation
- virtual device deployment
- CFD pipeline
- GNN vessel classification
- STEP6 and STEP7

---

## Phase 1 — Preserve and wrap STEP1

### Goal

Do not redesign STEP1. Preserve it and make it consumable by the refactored pipeline.

### Required work

- keep current STEP1 algorithmic behavior unchanged
- preserve current STEP1 output contract
- expose STEP1 through a callable wrapper if useful
- define explicit path/config handoff into STEP2
- ensure STEP2 never guesses STEP1 output paths

### STEP1 preserved outputs

- `surface_cleaned.vtp`
- `centerlines_raw_debug.vtp`
- `centerline_network.vtp`
- `junction_nodes_debug.vtp`
- `centerline_network_metadata.json`

---

## Phase 2 — Build STEP2 as the strict geometry-authoring contract

### New script

`step2_geometry_contract.py`

### Inputs

- STEP1 outputs
- input lumen surface
- `face_id_to_name.json`

### Core outputs

- `Output files\STEP2\segmentscolored.vtp`
- `Output files\STEP2\aorta_centerline.vtp`
- `Output files\STEP2\step2_geometry_contract.json`

### Optional outputs

- `Output files\STEP2\boundary_debug.vtp`
- `Output files\STEP2\step2_debug.json`

These optional outputs must not be required by downstream steps.

### Main implementation tasks

1. consume STEP1 outputs explicitly
2. use face map early to identify `abdominal_aorta_inlet`
3. use STEP1 topology only as routing/search skeleton
4. identify the abdominal aorta end just before bifurcation
5. author a single aorta trunk centerline from inlet to pre-bifurcation end
6. build bounded surface segments with one label per cell
7. write minimal `segmentscolored.vtp`
8. write `aorta_centerline.vtp` containing only the final trunk polyline
9. write unified `step2_geometry_contract.json`
10. write optional debug artifacts only if configured

### Hard requirements

- STEP2 must fail if the aortic end before bifurcation cannot be measured properly
- STEP2 must fail if a single trustworthy aorta trunk cannot be authored
- STEP2 must record fallback and QA status in the JSON contract
- STEP2 must not scatter critical truth across multiple JSON contracts

### STEP2 JSON contract must include

- source inputs
- upstream STEP1 references
- coordinate system and units
- aorta start metadata
- aorta end metadata
- aorta trunk length
- reference to `aorta_centerline.vtp`
- segment summary
- boundary summary
- fallback summary
- QA summary
- warnings
- final status

### STEP2 migration note

Current logic from `segment2.py` should be reused selectively, but the new file must obey the reduced output contract and single-JSON rule.

---

## Phase 3 — Build STEP3 as the naming and landmark contract

### New script

`step3_naming_orientation.py`

### Inputs

- `STEP2\segmentscolored.vtp`
- `STEP2\aorta_centerline.vtp`
- `STEP2\step2_geometry_contract.json`
- `face_id_to_name.json`

### Core outputs

- `Output files\STEP3\named_segmentscolored.vtp`
- `Output files\STEP3\named_centerlines.vtp`
- `Output files\STEP3\step3_naming_orientation_contract.json`

### Optional outputs

- none by default

### Main implementation tasks

1. consume STEP2 geometry as the only geometric authority
2. name face-map-connected vessels
3. build anatomically meaningful named vessel objects
4. emit one surface file with named segments
5. emit one centerline file with one polyline per vessel/segment object
6. record start/end and landmark metadata in one JSON contract
7. warn on missing non-priority vessels
8. return `requires_review` if priority vessel starts are low-confidence

### Important STEP3 rule

Topology may guide search, but final vessel starts should be surface-authored whenever possible.

### Priority vessel set

- abdominal aorta trunk
- left common iliac
- right common iliac
- left internal iliac
- right internal iliac
- left external iliac
- right external iliac
- left renal artery
- right renal artery

### Celiac simplification rule

The proximal celiac beginning is the important part for the current implementation.
Downstream celiac branching can remain simplified after the proximal origin.

### STEP3 migration note

Current logic from `orientation.py` should be split conceptually:
- consume geometry from STEP2
- emit naming/landmark contract
- avoid emitting multiple redundant VTP outputs

---

## Phase 4 — Build STEP4 as the strict measurement contract

### New script

`step4_infrarenal_neck.py`

### Inputs

- `STEP3\named_segmentscolored.vtp`
- `STEP3\named_centerlines.vtp`
- `STEP3\step3_naming_orientation_contract.json`

### Core outputs

- `Output files\STEP4\step4_measurements.json`
- `Output files\STEP4\infrarenal_neck_colored.vtp`

### Optional outputs

- `Output files\STEP4\infrarenal_neck_report.txt`

### Main implementation tasks

1. resolve trunk and renal anchors from STEP3 contract
2. locate D0, D5, D10, D15
3. determine proximal neck length
4. determine right and left common iliac diameter series and lengths
5. determine lowest renal to aortic bifurcation length
6. determine lowest renal to right/left iliac bifurcation lengths
7. determine right and left external iliac access diameters
8. determine maximum aneurysm diameter
9. write grouped section-based JSON
10. color proximal neck and aneurysm maximum-diameter region in VTP

### Hard failure rules

- fail if a required iliac diameter series is missing
- use structured `{ "status": "unmeasurable" }` for fields that cannot be measured individually
- write optional debug explanation if the step fails

### STEP4 migration note

Current logic from `measure_infrarenal_neck.py` should be reused only after adapting it to the new path contracts and the machine-readable JSON-first design.

---

## Phase 5 — Build STEP5 as pipeline validation and manifest

### New script

`step5_pipeline_manifest.py`

### Inputs

- STEP1 through STEP4 core outputs

### Core outputs

- `Output files\STEP5\pipeline_manifest.json`
- `Output files\STEP5\pipeline_summary.txt`

### Optional outputs

- none

### Main implementation tasks

1. validate required step outputs exist
2. validate each step contract at minimal contract level
3. aggregate warnings from STEP1 through STEP4
4. write per-step status
5. write final pipeline status
6. write summary text for human review

### Rules

- do not automatically rerun missing steps in the first implementation
- report missing dependencies instead
- only use statuses:
  - `success`
  - `failed`
  - `requires_review`

---

## Shared Refactor Work

### Required shared modules

- path/config helpers
- VTP read/write helpers
- JSON read/write helpers
- VMTK runtime setup helpers
- shared geometry utility helpers
- contract/status helper functions

### Refactor rule

Shared helpers may be extracted from existing scripts, but Codex must not alter STEP1 behavior while doing so.

---

## Deletion / replacement policy

### Keep

- STEP1 outputs and behavior
- STEP1 reference script as preserved source until replacement is verified

### Replace

- old STEP2/STEP3/STEP4 script outputs with the new reduced core contracts
- old redundant VTPs that are no longer part of the active default contract

### Delete later, only after replacement works

- redundant debug outputs that are no longer part of the approved architecture
- duplicate JSON artifacts replaced by the single-contract-per-step model

---

## Acceptance Criteria

### STEP2 accepted when

- core outputs exist
- aorta trunk is authored correctly
- aortic start and pre-bifurcation end are trustworthy
- segment assignment is complete enough under the locked fallback policy
- JSON contract is complete
- optional debug artifacts can explain failure when needed

### STEP3 accepted when

- named surface exists
- named per-vessel centerlines exist
- priority vessels are matched with acceptable confidence
- JSON contract records vessel starts/ends and landmarks

### STEP4 accepted when

- grouped machine-readable JSON exists
- colored VTP exists
- all required core measurements are present or explicitly structured as unmeasurable where allowed
- missing required iliac series triggers fail

### STEP5 accepted when

- manifest exists
- summary exists
- statuses are correct
- warnings are aggregated
- missing dependencies are clearly reported