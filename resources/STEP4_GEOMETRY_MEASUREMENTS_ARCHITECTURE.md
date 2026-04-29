# STEP4_GEOMETRY_MEASUREMENTS Architecture

## Step 4 Purpose

Step 4 converts Step 3 named anatomy into structured 3D geometry measurements.

Step 4 produces:

- one JSON measurement file
- one labeled VTP file showing the infrarenal neck region

Step 4 does not choose a device.
Step 4 does not perform device sizing.
Step 4 does not perform clinical suitability assessment.
Step 4 only measures anatomy from the current 3D model.

Step 4 has exactly two required outputs: JSON and labeled VTP.
Step 4 is a geometry-measurement stage and a measurement contract only.

## Step 4 Inputs

Required:

- `Output files/STEP3/named_segmentscolored.vtp`
- `Output files/STEP3/named_centerlines.vtp`
- `Output files/STEP3/step3_naming_orientation_contract.json`

## Step 4 Outputs

Required:

- `Output files/STEP4/step4_geometry_measurements.json`
- `Output files/STEP4/step4_infrarenal_neck_labeled.vtp`

No other required outputs.

## Step 4 Measurements

Step 4 measures only the following groups.

### A. landmarks

- `lowest_renal_artery_name`
- `lowest_renal_ostium_center_xyz`
- `lowest_renal_aortic_centerline_s_mm`
- `aortic_bifurcation_center_xyz`
- `aortic_bifurcation_s_mm`

### B. aortic_neck

- `neck_diameter_D0_major_mm`
- `neck_diameter_D0_minor_mm`
- `neck_diameter_D0_equivalent_mm`
- `neck_diameter_D10_major_mm`
- `neck_diameter_D10_minor_mm`
- `neck_diameter_D10_equivalent_mm`
- `neck_diameter_D15_major_mm`
- `neck_diameter_D15_minor_mm`
- `neck_diameter_D15_equivalent_mm`
- `neck_reference_diameter_mm`
- `neck_reference_diameter_source`
- `neck_length_mm`
- `neck_end_s_mm`
- `neck_end_center_xyz`
- `proximal_neck_angulation_deg`
- `proximal_neck_angulation_category`

### C. iliac

left:

- `iliac_treatment_diameter_mm`
- `iliac_reference_diameter_source`
- `distal_iliac_seal_zone_length_mm`
- `distal_iliac_seal_zone_start_xyz`
- `distal_iliac_seal_zone_end_xyz`
- `selected_landing_segment`

right:

- `iliac_treatment_diameter_mm`
- `iliac_reference_diameter_source`
- `distal_iliac_seal_zone_length_mm`
- `distal_iliac_seal_zone_start_xyz`
- `distal_iliac_seal_zone_end_xyz`
- `selected_landing_segment`

### D. measurement_status

- `aortic_neck_diameter_measurement_status`
- `aortic_neck_length_measurement_status`
- `aortic_neck_angulation_measurement_status`
- `left_iliac_diameter_measurement_status`
- `right_iliac_diameter_measurement_status`
- `left_iliac_seal_zone_measurement_status`
- `right_iliac_seal_zone_measurement_status`
- `overall_geometry_measurement_status`

Allowed measurement statuses:

- `measured`
- `not_available`
- `requires_review`
- `failed_to_measure`

### E. metadata

- `source_inputs`
- `units`
- `coordinate_system`
- `measurement_confidence`
- `warnings`
- `open_questions`
- `not_available`
- `assumptions`
- `discovered_arrays`

## Labeled VTP Architecture

Step 4 writes a VTP that preserves the full named surface geometry and adds cell-data arrays that label the infrarenal neck.

The output VTP should include at least these cell-data arrays:

- `Step4RegionId`
  - `0 = other_model_surface`
  - `1 = infrarenal_neck`
- `Step4RegionName`
  - `other_model_surface`
  - `infrarenal_neck`
- `InfrarenalNeckMask`
  - `0 = not neck`
  - `1 = neck`
- `Step4ColorRGB`
  - neutral color for rest of model
  - distinct color for infrarenal neck

If string arrays are difficult, at minimum write numeric arrays:

- `Step4RegionId`
- `InfrarenalNeckMask`
- `Step4ColorRGB`

The labeled VTP must not delete existing useful Step 3 arrays unless unavoidable.

## Acceptance Criteria

- Step 4 has exactly two required outputs: JSON and labeled VTP.
- Step 4 is a geometry-measurement stage.
- Step 4 must not silently guess missing anatomy.
- Step 4 must mark unavailable or uncertain measurements in JSON.
- Step 4 must label the infrarenal neck region in the VTP.
- Step 4 must preserve the full model in the labeled VTP.
- Step 4 must not introduce unrelated modules or speculative variables.
- Step 4 must not include device sizing, catalogue matching, oversizing, calcium, thrombus, plaque, tortuosity, curvature, hemodynamics, CFD, clinical contraindications, access-risk scoring, final clinical pass/fail, device-selection logic, or Step 5 reporting.
