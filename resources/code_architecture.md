# Toralis Vascular Pipeline Code Architecture

## Purpose

This software is a contract-driven vascular anatomy pipeline whose near-term purpose is to convert one vascular lumen surface model into a reusable anatomical object that can support:

- stable aorta trunk extraction
- surface-true vessel boundary definition
- named vessel segmentation
- landmark registration
- infrarenal neck and iliac measurement
- later IFU matching logic

This software is **not** yet trying to solve final stent recommendation, virtual device deployment, CFD, or ML-based vessel classification. It is building the anatomical and geometric backbone that those later systems will depend on.

The current demonstration input is a Stanford-style vascular lumen `.vtp` model and a manually prepared `face_id_to_name.json`. The long-term architecture must remain compatible with a future upstream segmentation step from DICOM, but that upstream step is **not implemented in this phase**.

## Overall Design Principles

1. **STEP1 is preserved**
   - STEP1 is already complete and should not be redesigned unless a specific blocker is found.
   - STEP1 remains the topology and centerline routing layer.

2. **The graph is the search skeleton, not the final authority**
   - STEP1 centerline graph is used to route search and identify likely vessel relationships.
   - True vessel starts, ostia, bifurcations, and segment boundaries must be authored from the lumen surface whenever possible.

3. **STEP2 is the geometry-authoring layer**
   - STEP2 is the single source of truth for the abdominal aorta trunk and bounded topology segments.
   - STEP2 must emit a strict geometry contract that later steps consume instead of re-solving geometry.

4. **STEP3 consumes geometry and adds semantics**
   - STEP3 names vessels and registers landmarks.
   - STEP3 must not silently redefine aortic start/end or vessel boundary geometry authored by STEP2.

5. **Each step has only two output classes**
   - **Core outputs**: required for downstream execution
   - **Optional outputs**: debug/reference artifacts used only when diagnosing failures

6. **Each step writes exactly one main machine-readable JSON contract**
   - That JSON is the authoritative contract for the step.
   - VTP outputs are primarily for geometry reuse and visual confirmation.

7. **Core outputs must be minimal**
   - The default pipeline must emit only the few outputs required for downstream work.
   - Extra debug outputs must not be part of the default public contract.

8. **Failure should be explicit**
   - Important geometric failures must stop the step rather than silently producing weak outputs.
   - `requires_review` is reserved for outputs that were produced but contain flagged uncertainty.

## Pipeline Scope

The active pipeline scope for this phase is:

- STEP1 — preserved topology centerline network
- STEP2 — geometry contract and aorta trunk authoring
- STEP3 — vessel naming and landmark contract
- STEP4 — infrarenal neck and iliac measurement contract
- STEP5 — orchestration, QA manifest, final summary

There is **no STEP6 or STEP7 in this phase**.

---

## STEP1 — Topology Centerline Network

### Role

STEP1 produces the topology model for the vascular tree from the input lumen surface.

### Status

- Preserved
- Existing behavior should remain stable
- Only safe packaging/refactor changes are allowed

### What STEP1 does

- sanitizes the raw lumen surface
- detects terminal regions
- computes VMTK centerlines
- reconstructs a shared centerline network
- writes root / terminal / junction metadata

### What STEP1 does not do

- it does not solve final aorta trunk boundary definition
- it does not solve true surface-derived vessel starts
- it does not assign final vessel names
- it does not perform EVAR measurements

### STEP1 core outputs (preserved)

- `surface_cleaned.vtp`
- `centerlines_raw_debug.vtp`
- `centerline_network.vtp`
- `junction_nodes_debug.vtp`
- `centerline_network_metadata.json`

### STEP1 rule

Later steps must consume STEP1 outputs through explicit paths/config and must not guess paths.

---

## STEP2 — Geometry Contract and Aorta Trunk Authoring

### Script name

`step2_geometry_contract.py`

### Role

STEP2 is the most important stage in the active architecture.

It must:

- identify the abdominal aorta start
- identify the abdominal aorta end just before bifurcation
- collapse the multi-part STEP1 aorta representation into one explicit trunk
- author bounded topology segments from the surface
- assign every surface cell to exactly one segment
- emit the strict geometry contract that all later steps consume

### Inputs

Required:

- STEP1 outputs
- input lumen surface
- `face_id_to_name.json`

Current required face-map fact:

- face `2` is `abdominal_aorta_inlet`

### Core outputs

- `segmentscolored.vtp`
- `aorta_centerline.vtp`
- `step2_geometry_contract.json`

### Optional outputs

- one boundary-debug VTP
- one QA/debug JSON

No other optional outputs should be created by default.

### STEP2 output rules

#### `segmentscolored.vtp`
This is the only default surface file written by STEP2.

It should contain only the minimum needed for downstream execution and visual inspection:

- `SegmentId`
- color array

No extra optional diagnostic arrays should be embedded in the core VTP by default.

#### `aorta_centerline.vtp`
This is a visual and geometric output for confirming the aorta trunk.

Default core visualization choice:
- the file contains **only the final aorta trunk polyline**

The full model is not duplicated inside this file by default.

Any extra geometric details such as cumulative arclength, tangents, radii, or landmark IDs belong in the JSON contract, not in the core VTP.

#### `step2_geometry_contract.json`
This is the single authoritative machine-readable output for STEP2.

It must include at minimum:

- input source paths
- upstream STEP1 references
- coordinate system and units
- aorta start metadata
- aorta end metadata
- aorta trunk length
- reference to `aorta_centerline.vtp`
- segment summary table
- boundary summary table
- warnings
- fallback details
- QA status

### Required aorta start and end definition

#### Aorta start
The start must be the true abdominal aorta inlet identified from the face map and confirmed geometrically.

The contract must record:

- source type
- centroid
- normal
- area
- equivalent diameter
- major/minor diameter if available
- confidence

#### Aorta end
The end must be the abdominal aorta end just **before** bifurcation.

The contract must record both:

- a centerline landmark
- a surface-derived boundary profile when successful

The end contract must include:

- bifurcation point
- boundary centroid if measurable
- local tangent or boundary normal
- area when measurable
- equivalent diameter
- major/minor diameter if available
- confidence
- extraction method
- whether the end was surface-derived, centerline-derived, or both

### Hard failure conditions for STEP2

STEP2 must fail if any of the following occur:

- aortic inlet cannot be resolved
- aortic end before bifurcation cannot be measured properly
- `aorta_centerline.vtp` cannot be written
- a single valid aorta trunk cannot be authored
- core segment assignment fails badly enough that the step is no longer trustworthy

### Fallback assignment policy

Fallback cell assignment must be recorded in the contract.

Initial threshold policy:

- `requires_review` if fallback-assigned cells > 5%
- `failed` if fallback-assigned cells > 15%
- `requires_review` regardless of overall percentage if fallback affects:
  - aortic inlet
  - aortic bifurcation
  - renal origins
  - iliac split boundaries

### STEP2 QA requirements

The contract must record at minimum:

- total surface cells
- assigned cells
- unassigned cells
- fallback-assigned cells
- fallback percentage
- whether any priority region was fallback-authored
- non-empty segment count
- aorta start confidence
- aorta end confidence
- step status

---

## STEP3 — Vessel Naming and Landmark Contract

### Script name

`step3_naming_orientation.py`

### Role

STEP3 consumes STEP2 geometry and assigns semantic vessel names and landmark meaning.

STEP3 must not redefine the geometry authored by STEP2.

### Inputs

Required:

- `segmentscolored.vtp`
- `aorta_centerline.vtp`
- `step2_geometry_contract.json`
- `face_id_to_name.json`

### Core outputs

- `named_segmentscolored.vtp`
- `named_centerlines.vtp`
- `step3_naming_orientation_contract.json`

### Optional outputs

- none by default

### STEP3 output rules

#### `named_segmentscolored.vtp`
This is the anatomical vessel surface output.

It should contain only:

- `SegmentId`
- `SegmentName`
- color array

No extra optional diagnostic arrays are required in the core file.

#### `named_centerlines.vtp`
This file is a per-vessel centerline representation.

Rules:

- one polyline for each segment/vessel object
- the centerline for each vessel must start at the anatomically meaningful proximal beginning of that vessel
- the centerline for each vessel must end at its distal end
- centerlines may be disconnected from one another in the final file
- this is intentional and useful

This file should contain only:

- `SegmentId`
- `SegmentName`

No extra optional arrays are required in the core file.

### Vessel naming priorities

The following vessels are high priority and should be matched accurately:

- abdominal aorta trunk
- left common iliac
- right common iliac
- left internal iliac
- right internal iliac
- left external iliac
- right external iliac
- left renal artery
- right renal artery

If any of these are matched but their surface-authored proximal start is low-confidence, STEP3 should return:

- `requires_review`

### Non-priority vessel behavior

If a non-priority vessel is not matched, STEP3 should warn instead of fail.

### Celiac handling rule

The proximal ostium / beginning of the celiac vessel is the most important part.

After the proximal celiac beginning, downstream branches can remain grouped more simply in the current implementation.

### Common iliac rule

Common iliac definition must not be treated as merely topology-based in the final sense.

Implementation rule:

- topology may be used to route the search
- the actual vessel beginning should be surface-authored whenever possible
- topology-only inference is not the preferred final authority

### STEP3 contract requirements

`step3_naming_orientation_contract.json` must include:

- input references to STEP2 outputs
- segment-to-name mapping
- vessel priority classification
- matched face-map entries
- unmatched face-map entries
- inferred vessels
- landmark registry
- proximal start metadata for each named vessel
- distal end metadata for each named vessel
- confidence flags
- warnings
- step status

---

## STEP4 — Infrarenal Neck and Iliac Measurement Contract

### Script name

`step4_infrarenal_neck.py`

### Role

STEP4 performs the first EVAR-relevant measurement contract using STEP3 named and landmarked anatomy.

### Inputs

Required:

- `named_segmentscolored.vtp`
- `named_centerlines.vtp`
- `step3_naming_orientation_contract.json`

### Core outputs

- `step4_measurements.json`
- `infrarenal_neck_colored.vtp`

### Optional outputs

- optional text report for debugging/reference only

### Visualization requirements for `infrarenal_neck_colored.vtp`

The core colored VTP must visualize at least:

- proximal neck region
- maximum aneurysm diameter region

This may be expanded later, but only these are required now.

### STEP4 JSON format

The core JSON must be a **section-based grouped object**, not a flat list.

### Required measurement scope

The step must support the following summarized outputs when measurable:

- reference artery
- proximal neck D0
- proximal neck D5
- proximal neck D10
- proximal neck D15
- proximal neck length
- right common iliac D0
- right common iliac D-10
- right common iliac D-15
- right common iliac D-20
- right common iliac length
- left common iliac D0
- left common iliac D-10
- left common iliac D-15
- left common iliac D-20
- left common iliac length
- lowest renal to aortic bifurcation length
- lowest renal to right iliac bifurcation length
- lowest renal to left iliac bifurcation length
- right external iliac access diameter
- left external iliac access diameter
- maximum external aneurysm diameter

### Unmeasurable fields

If a value cannot be measured, it must use a structured status object:

```json
{ "status": "unmeasurable" }