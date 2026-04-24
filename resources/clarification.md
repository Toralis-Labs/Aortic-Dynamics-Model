# Clarification And Locked Decisions

## Locked Decisions

### General

- The current discussion is focused only on STEP2 through STEP5.
- There is no STEP6 or STEP7 in the active implementation scope.
- STEP1 is already complete and should be preserved.
- STEP1 behavior should not be redesigned unless a specific blocker is found.
- Each step must have:
  - core outputs required for downstream execution
  - optional review/debug outputs not required for downstream execution
- Each step must write exactly one main machine-readable JSON contract.

### Script names

- STEP2 script name: `step2_geometry_contract.py`
- STEP3 script name: `step3_naming_orientation.py`
- STEP4 script name: `step4_infrarenal_neck.py`
- STEP5 script name: `step5_pipeline_manifest.py`

### STEP2

- STEP2 core outputs are:
  - `segmentscolored.vtp`
  - `aorta_centerline.vtp`
  - `step2_geometry_contract.json`
- STEP2 optional outputs are limited to:
  - one boundary-debug VTP
  - one QA/debug JSON
- `segmentscolored.vtp` is the only default surface file written by STEP2.
- `aorta_dimensions.json` and `segments_metadata.json` should be merged into one STEP2 geometry contract JSON.
- STEP2 must include both:
  - a centerline landmark for the aortic end/bifurcation region
  - a surface-derived boundary profile when successful
- The aortic inlet is the start.
- The abdominal aorta end before bifurcation is the end.
- Accuracy of the aortic start and aortic end is critical and will be used later in multiple applications.
- STEP2 must fail if the aortic end before bifurcation cannot be measured properly.
- STEP2 JSON should store summary + references.
- Full geometry should remain in VTP outputs.
- STEP2 should include major/minor diameter when available, but these are not required for step success.

### STEP2 fallback policy

Initial locked recommendation:

- `requires_review` if fallback-assigned cells > 5%
- `failed` if fallback-assigned cells > 15%
- `requires_review` regardless of percentage if fallback affects:
  - aortic inlet
  - aortic bifurcation
  - renal origins
  - iliac split boundaries

### STEP3

- STEP3 core outputs are:
  - `named_segmentscolored.vtp`
  - `named_centerlines.vtp`
  - `step3_naming_orientation_contract.json`
- STEP3 should focus on naming and landmarking accurately.
- STEP3 should not be responsible for redefining geometry.
- `named_segmentscolored.vtp` is the abdominal aortic model where each vessel segment is recognized and labeled.
- `named_centerlines.vtp` should contain one polyline for each segment/vessel object.
- The centerline for each vessel must accurately represent where that vessel starts in space.
- Centerlines may be disconnected from one another.
- This is intentional and useful.
- Coordinates for each vessel start and end must be as accurate as possible.
- The start/ostium of each vessel is especially important for later information extraction.
- Priority vessels are:
  - abdominal aorta trunk
  - left common iliac
  - right common iliac
  - left internal iliac
  - right internal iliac
  - left external iliac
  - right external iliac
  - left renal artery
  - right renal artery
- If a non-priority vessel is not matched, STEP3 should warn.
- If a priority vessel is matched but its surface-authored proximal start is low-confidence, STEP3 should return `requires_review`.
- For celiac handling:
  - the proximal celiac branch/ostium is the important part
  - from the start of celiac and later, downstream branch coloring can remain simplified for now
- Common iliac handling should not be final-authority topology-based.
- Search routing may use topology.
- Final vessel starts should be surface-authored whenever possible.

### VTP array minimization

- STEP2 `segmentscolored.vtp` should contain only:
  - `SegmentId`
  - color array
- STEP3 `named_segmentscolored.vtp` should contain only:
  - `SegmentId`
  - `SegmentName`
  - color array
- STEP3 `named_centerlines.vtp` should contain only:
  - `SegmentId`
  - `SegmentName`
- Extra confidence, parent ID, fallback, and optional diagnosis fields belong in optional outputs or JSON contracts, not in core VTP files.

### Aorta centerline visualization

- `aorta_centerline.vtp` should contain only the final aorta trunk polyline by default.
- Extra geometric information such as arclength, tangents, radii, and landmark point IDs should live in the JSON contract, not in the core VTP.

### STEP4

- STEP4 must write a machine-readable JSON as a core output.
- STEP4 core outputs are:
  - `step4_measurements.json`
  - `infrarenal_neck_colored.vtp`
- STEP4 may optionally write a text report for debug/reference.
- The STEP4 JSON must be a section-based grouped object.
- If a value cannot be measured, it must use:
  - `{ "status": "unmeasurable" }`
- STEP4 must fail if one iliac diameter series is missing.
- STEP4 JSON should contain summarized values only in the first implementation.
- The colored VTP must at minimum visualize:
  - proximal neck
  - maximum aneurysm diameter

### STEP5

- STEP5 core outputs are:
  - `pipeline_manifest.json`
  - `pipeline_summary.txt`
- STEP5 should validate and report.
- STEP5 should not automatically rerun prior steps in the first implementation.
- Allowed status values are only:
  - `success`
  - `failed`
  - `requires_review`
- STEP5 should be the only step that aggregates warnings globally.
- STEP5 does not need hashes, timestamps, git commit fields, or software version fields in the first implementation.

## Current face map facts

Active development face-map path:
`C:\Users\ibrah\OneDrive\Desktop\Fluids Project\Vascular specific\0044_H_ABAO_AAA\face_id_to_name.json`

Current file:

`face_id_to_name.json`

Known entries:

- `2` -> `abdominal_aorta_inlet`
- `3` -> `celiac_branch`
- `4` -> `celiac_artery`
- `5` -> `left_external_iliac`
- `6` -> `right_external_iliac`
- `7` -> `inferior_mesenteric_artery`
- `8` -> `left_internal_iliac`
- `9` -> `right_internal_iliac`
- `10` -> `left_renal_artery`
- `11` -> `right_renal_artery`
- `12` -> `superior_mesenteric_artery`

Not present in the face map:

- `left_common_iliac`
- `right_common_iliac`

## Remaining Open Questions

At the current state, the major architectural decisions are mostly locked.

Resolved now for first implementation:

- STEP2 contract canonical top-level fields:
  - `schema_version`
  - `step_name`
  - `step_status`
  - `warnings`
  - `input_paths`
  - `output_paths`
  - `upstream_references`
  - `coordinate_system`
  - `units`
  - `aorta_start`
  - `aorta_end`
  - `aorta_trunk`
  - `segment_summary`
  - `boundary_summary`
  - `fallback_details`
  - `qa`
- STEP2 field naming rules:
  - use `snake_case` everywhere
  - use `step_status` as the canonical status field name
  - use `input_paths.face_map`
  - use `input_paths.input_vtp`
  - use `input_paths.surface_cleaned`
  - use `input_paths.centerline_network`
  - use `input_paths.step1_metadata`
  - use `output_paths.segments_vtp`
  - use `output_paths.aorta_centerline`
  - use `output_paths.contract_json`
- STEP3 contract canonical top-level fields:
  - `schema_version`
  - `step_name`
  - `step_status`
  - `warnings`
  - `input_paths`
  - `output_paths`
  - `step2_references`
  - `segment_name_map`
  - `vessel_priority_classification`
  - `matched_face_map_entries`
  - `unmatched_face_map_entries`
  - `inferred_vessels`
  - `landmark_registry`
  - `proximal_start_metadata`
  - `distal_end_metadata`
  - `confidence_flags`
  - `qa`
- STEP3 field naming rules:
  - use `snake_case` everywhere
  - use `segment_name_map` as the canonical `SegmentId` -> vessel-name mapping
  - use `matched_face_map_entries` and `unmatched_face_map_entries` exactly with those names
  - use `proximal_start_metadata` and `distal_end_metadata` exactly with those names
  - use `step_status` as the canonical status field name
  - use `input_paths.face_map`
  - use `input_paths.segments_vtp`
  - use `input_paths.aorta_centerline`
  - use `input_paths.step2_contract_json`
  - use `output_paths.named_segments_vtp`
  - use `output_paths.named_centerlines_vtp`
  - use `output_paths.contract_json`
- The active development face-map path for STEP2 and STEP3 remains:
  - `C:\Users\ibrah\OneDrive\Desktop\Fluids Project\Vascular specific\0044_H_ABAO_AAA\face_id_to_name.json`
- This face-map path is treated as a required STEP3 input and not as a file generated by STEP2.

The following may still need refinement later, but are not blockers for the first implementation:

- exact section names for STEP4 grouped measurement JSON
- whether STEP3 later distinguishes more celiac downstream branches beyond the current simplified rule
- whether later versions should add richer QA severity categories beyond `success`, `failed`, `requires_review`
