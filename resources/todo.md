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
- [ ] Write `step4_measurements.json`
- [ ] Write `infrarenal_neck_colored.vtp`
- [ ] Keep JSON as section-based grouped object
- [ ] Support structured unmeasurable field:
  - [ ] `{ "status": "unmeasurable" }`
- [ ] Measure proximal neck:
  - [ ] D0
  - [ ] D5
  - [ ] D10
  - [ ] D15
  - [ ] neck length
- [ ] Measure right common iliac:
  - [ ] D0
  - [ ] D-10
  - [ ] D-15
  - [ ] D-20
  - [ ] length
- [ ] Measure left common iliac:
  - [ ] D0
  - [ ] D-10
  - [ ] D-15
  - [ ] D-20
  - [ ] length
- [ ] Measure lowest renal to aortic bifurcation length
- [ ] Measure lowest renal to right iliac bifurcation length
- [ ] Measure lowest renal to left iliac bifurcation length
- [ ] Measure right external iliac access diameter
- [ ] Measure left external iliac access diameter
- [ ] Measure maximum aneurysm diameter
- [ ] Color proximal neck in the VTP
- [ ] Color maximum aneurysm diameter region in the VTP
- [ ] Fail if one iliac diameter series is missing

## STEP4 Optional Output

- [ ] Add optional `infrarenal_neck_report.txt` only for human debug/reference

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