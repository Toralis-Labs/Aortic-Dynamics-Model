# Implementation TODO

## Global Architecture

- [ ] Preserve STEP1 behavior exactly unless a concrete blocker is found.
- [ ] Replace current STEP2–STEP5 architecture docs with the new reduced-output design.
- [ ] Enforce one main machine-readable JSON contract per step.
- [ ] Enforce core vs optional output separation for each step.
- [ ] Create shared path/config object for step-specific input and output paths.
- [ ] Ensure each step writes only into its own output folder.

## Script Naming

- [ ] Create `step2_geometry_contract.py`
- [ ] Create `step3_naming_orientation.py`
- [ ] Create `step4_infrarenal_neck.py`
- [ ] Create `step5_pipeline_manifest.py`

## STEP1 Preservation

- [ ] Keep current STEP1 algorithmic behavior unchanged.
- [ ] Preserve current STEP1 output contract.
- [ ] Add safe callable wrapper if helpful.
- [ ] Ensure STEP2 can consume STEP1 outputs without path guessing.

## STEP2 Core Contract

- [ ] Implement `Output files\STEP2`
- [ ] Read STEP1 outputs explicitly
- [ ] Read `face_id_to_name.json` explicitly
- [ ] Use face `2` / `abdominal_aorta_inlet` as aorta start
- [ ] Identify the abdominal aorta end just before bifurcation
- [ ] Author one explicit aorta trunk centerline
- [ ] Write `segmentscolored.vtp`
- [ ] Write `aorta_centerline.vtp`
- [ ] Write `step2_geometry_contract.json`
- [ ] Merge old aorta and segment metadata into the single STEP2 contract
- [ ] Record start centroid, normal, area, equivalent diameter, confidence
- [ ] Record end landmark and surface-derived boundary when successful
- [ ] Record major/minor diameters when available
- [ ] Record aorta trunk length
- [ ] Record fallback usage
- [ ] Record QA fields
- [ ] Record warnings and status
- [ ] Fail if aortic end before bifurcation cannot be measured properly
- [ ] Keep core VTP arrays minimal

## STEP2 Proximal Boundary Refinement

- [x] Clean duplicate/corrupt attempted proximal-boundary patch.
- [x] Implement one active proximal-boundary control flow in `_refine_branch_boundaries`.
- [x] Use `_extract_branch_stable_section_boundary` to find the internal daughter-section reference.
- [x] Use `_refine_branch_ostium_backward_from_stable_section` to search backward toward the parent ostium.
- [x] Use `backward_refined_from_stable_section_v1` for successful refined proximal boundaries.
- [x] Add `stable_section_reference` metadata to every selected branch proximal boundary.
- [x] Add `backward_refinement` metadata with step size, max search, accepted count, stop reason, arclengths, search distance, and attempts.
- [x] Preserve STEP3 compatibility by keeping existing `segment_summary` proximal-boundary fields.
- [x] Mark fallback or low-confidence proximal boundaries with `requires_review` and warning metadata.
- [x] Enforce code hygiene: one `SegmentBoundaryProfile`, one `_project_points_to_polyline`, one `_refine_branch_boundaries`, no duplicate definitions, no contradictory constants, and no code after `return warnings`.
- [x] Run acceptance checks: `py_compile`, AST duplicate-definition check, control-flow text check, and resources documentation check.

## STEP2 Optional Outputs

- [ ] Implement one boundary-debug VTP only
- [ ] Implement one QA/debug JSON only
- [ ] Make optional outputs available only for diagnosis/reference
- [ ] Ensure downstream steps do not require optional outputs

## STEP2 Fallback / QA Policy

- [ ] Implement fallback percentage calculation
- [ ] Mark `requires_review` if fallback-assigned cells > 5%
- [ ] Fail if fallback-assigned cells > 15%
- [ ] Mark `requires_review` if fallback touches:
  - [ ] aortic inlet
  - [ ] aortic bifurcation
  - [ ] renal origins
  - [ ] iliac split boundaries

## STEP3 Core Contract

- [ ] Implement `Output files\STEP3`
- [ ] Consume STEP2 outputs only as geometric truth
- [ ] Write `named_segmentscolored.vtp`
- [ ] Write `named_centerlines.vtp`
- [ ] Write `step3_naming_orientation_contract.json`
- [ ] Keep `named_segmentscolored.vtp` arrays minimal:
  - [ ] `SegmentId`
  - [ ] `SegmentName`
  - [ ] color array
- [ ] Keep `named_centerlines.vtp` arrays minimal:
  - [ ] `SegmentId`
  - [ ] `SegmentName`
- [ ] Emit one polyline per vessel/segment object
- [ ] Ensure vessel centerlines can remain disconnected
- [ ] Record precise proximal start/end metadata in JSON
- [ ] Prioritize:
  - [ ] abdominal aorta trunk
  - [ ] left/right common iliac
  - [ ] left/right internal iliac
  - [ ] left/right external iliac
  - [ ] left/right renal
- [ ] Warn on non-priority vessel mismatch
- [ ] Return `requires_review` for low-confidence priority vessel starts

## STEP3 Surface-Based Starts

- [ ] Use topology only as routing aid
- [ ] Attempt surface-authored proximal starts whenever possible
- [ ] Do not treat topology-only common iliac inference as final truth
- [ ] Preserve celiac proximal origin as the important celiac landmark
- [ ] Allow simplified downstream celiac grouping after the proximal origin

## STEP4 Core Contract

- [ ] Implement `Output files\STEP4`
- [ ] Write `step4_geometry_measurements.json`
- [ ] Write `step4_infrarenal_neck_labeled.vtp`
- [ ] Keep JSON as a geometry measurement contract only
- [ ] Measure landmarks: lowest renal artery and aortic bifurcation
- [ ] Measure aortic neck diameters at D0, D10, and D15
- [ ] Measure aortic neck reference diameter, length, end point, and proximal angulation
- [ ] Measure left and right iliac landing-segment treatment diameters and distal seal-zone geometry
- [ ] Record measurement statuses using `measured`, `not_available`, `requires_review`, or `failed_to_measure`
- [ ] Record discovered Step 3 VTP arrays in metadata
- [ ] Preserve the full named surface in the labeled VTP
- [ ] Add `Step4RegionId`, `InfrarenalNeckMask`, and `Step4ColorRGB` cell-data arrays
- [ ] Add `Step4RegionName` cell-data array if practical
- [ ] Label only the infrarenal neck region in the VTP
- [ ] Do not add device sizing, clinical pass/fail, access-risk scoring, hemodynamics, or Step 5 reporting

## STEP5 Manifest Layer

- [ ] Implement `Output files\STEP5`
- [ ] Write `pipeline_manifest.json`
- [ ] Write `pipeline_summary.txt`
- [ ] Validate required core outputs from STEP1–STEP4
- [ ] Aggregate warnings globally
- [ ] Record only:
  - [ ] `success`
  - [ ] `failed`
  - [ ] `requires_review`
- [ ] Do not rerun prior steps automatically
- [ ] Report missing dependencies explicitly

## Shared Refactor Work

- [ ] Extract shared VTP read/write helpers
- [ ] Extract shared JSON helpers
- [ ] Extract shared path/config helpers
- [ ] Extract shared VMTK runtime helpers
- [ ] Extract shared geometry utility helpers
- [ ] Keep STEP1 behavior stable during helper extraction

## Verification

- [ ] Verify STEP2 `aorta_centerline.vtp` visually
- [ ] Verify STEP2 `segmentscolored.vtp` visually
- [ ] Verify STEP2 aortic start and end correctness visually
- [ ] Verify STEP3 named vessel surface visually
- [ ] Verify STEP3 per-vessel centerlines visually
- [ ] Verify STEP4 proximal neck coloring visually
- [ ] Verify STEP4 aneurysm diameter coloring visually
- [ ] Verify STEP5 manifest correctness

## Cleanup After Replacement Works

- [ ] Remove obsolete redundant STEP2 outputs
- [ ] Remove obsolete redundant STEP3 outputs
- [ ] Remove obsolete redundant STEP4 outputs
- [ ] Keep only approved core and optional outputs in the default architecture
