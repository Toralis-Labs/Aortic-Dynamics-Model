# STEP4 EVAR Geometry Measurement Contract

## Purpose

STEP4 measures lumen-derived EVAR geometry from STEP3 named anatomy. It produces
machine-readable measurements and visual/debug geometry for later EVAR/IBE planning
architecture.

STEP4 does not perform clinical recommendation, device selection, IFU matching,
catalogue matching, tissue assessment, or surgeon review. It must not infer thrombus,
calcification, plaque, clinical contraindications, final device compatibility, or final
access suitability from a lumen-only model.

STEP4 measures aortic neck, iliac, common iliac, external iliac, internal iliac,
renal-to-internal-iliac path length, and access-vessel lumen geometry from STEP3 named
anatomy.

Preferred future names:

- Root wrapper: `step4_evar_geometry_measurements.py`
- Package implementation: `src/step4/evar_geometry_measurements.py`
- Main JSON output: `step4_measurements.json`
- Visual/debug output: `step4_evar_geometry_regions.vtp`

## Compatibility wrapper note

`step4_infrarenal_neck.py` is an existing root wrapper name and may be retained
temporarily for compatibility. It must not define the conceptual scope of STEP4. Future
implementation should be `src/step4/evar_geometry_measurements.py`, and a future root
wrapper may be `step4_evar_geometry_measurements.py`.

## Inputs from STEP3

Required STEP3 inputs:

- `named_segmentscolored.vtp`
- `named_centerlines.vtp`
- `step3_naming_orientation_contract.json`

STEP4 uses semantic named anatomy from STEP3. It should resolve vessels by canonical
segment names rather than raw face IDs unless STEP3 explicitly exposes terminal face IDs
as metadata.

Expected priority vessel names:

- `abdominal_aorta`
- `left_renal_artery`
- `right_renal_artery`
- `left_common_iliac`
- `right_common_iliac`
- `left_external_iliac`
- `right_external_iliac`
- `left_internal_iliac`
- `right_internal_iliac`

If exact STEP3 names differ, future STEP4 code must use STEP3's canonical naming
contract rather than hardcoded guesses.

## Core outputs

STEP4 core outputs are:

- `step4_measurements.json`
- `step4_evar_geometry_regions.vtp`

`step4_evar_geometry_regions.vtp` should visualize measurement regions, planes, and
paths across the full EVAR geometry measurement layer.

## Status vocabulary

Top-level step statuses use the existing repository vocabulary:

- `success`
- `requires_review`
- `failed`

Individual measurement statuses may use:

- `measured`
- `derived_summary`
- `unmeasurable`
- `missing_required_landmark`
- `requires_review`
- `not_applicable`

## Measurement object schema

Every individual measurement must be an object, not a bare number.

```json
{
  "value": null,
  "unit": "mm|degrees|ratio|none",
  "status": "measured|derived_summary|unmeasurable|missing_required_landmark|requires_review|not_applicable",
  "method": "",
  "landmarks_used": [],
  "source_segment_names": [],
  "side": "left|right|midline|bilateral|summary|unknown",
  "confidence": null,
  "notes": []
}
```

Diameter measurements may also include:

```text
{
  "major_diameter_mm": null,
  "minor_diameter_mm": null,
  "equivalent_diameter_mm": null,
  "area_mm2": null,
  "plane_origin": [x, y, z],
  "plane_normal": [nx, ny, nz],
  "centerline_abscissa_mm": null
}
```

Generic bilateral summary fields must include:

```json
{
  "status": "derived_summary",
  "summary_rule": "side_specific_values_preferred",
  "left_ref": "<left_field_name>",
  "right_ref": "<right_field_name>",
  "left_value": null,
  "right_value": null,
  "unit": "mm|degrees|ratio|none",
  "notes": []
}
```

## Output JSON top-level schema

Target structure:

```json
{
  "step_name": "STEP4_EVAR_GEOMETRY_MEASUREMENTS",
  "step_status": "success|requires_review|failed",
  "input_paths": {},
  "output_paths": {
    "measurements_json": "step4_measurements.json",
    "measurement_regions_vtp": "step4_evar_geometry_regions.vtp"
  },
  "upstream_references": {
    "step3_contract": "step3_naming_orientation_contract.json"
  },
  "units": {
    "length": "mm",
    "angle": "degrees"
  },
  "measurement_groups": {
    "aortic_neck": {},
    "iliac_summary": {},
    "common_iliac": {},
    "external_iliac": {},
    "internal_iliac": {},
    "renal_to_internal_iliac": {},
    "access": {}
  },
  "unmeasurable_values": [],
  "warnings": [],
  "qa": {}
}
```

## Required measurement fields

AORTIC_NECK:

- proximal_neck_diameter_mm
- proximal_neck_major_diameter_mm
- proximal_neck_minor_diameter_mm
- proximal_neck_equivalent_diameter_mm
- infrarenal_aortic_neck_treatment_diameter_mm
- aortic_treatment_diameter_mm
- proximal_neck_length_mm
- proximal_neck_angulation_deg

ILIAC SUMMARY:

- iliac_treatment_diameter_mm
- left_iliac_treatment_diameter_mm
- right_iliac_treatment_diameter_mm
- distal_iliac_seal_zone_length_mm
- left_distal_iliac_seal_zone_length_mm
- right_distal_iliac_seal_zone_length_mm

COMMON ILIAC:

- common_iliac_diameter_mm
- left_common_iliac_diameter_mm
- right_common_iliac_diameter_mm
- common_iliac_length_mm
- left_common_iliac_length_mm
- right_common_iliac_length_mm

EXTERNAL ILIAC:

- external_iliac_treatment_diameter_mm
- left_external_iliac_treatment_diameter_mm
- right_external_iliac_treatment_diameter_mm
- external_iliac_seal_zone_length_mm
- left_external_iliac_seal_zone_length_mm
- right_external_iliac_seal_zone_length_mm

INTERNAL ILIAC:

- internal_iliac_treatment_diameter_mm
- left_internal_iliac_treatment_diameter_mm
- right_internal_iliac_treatment_diameter_mm
- internal_iliac_seal_zone_length_mm
- left_internal_iliac_seal_zone_length_mm
- right_internal_iliac_seal_zone_length_mm

RENAL TO INTERNAL ILIAC:

- renal_to_internal_iliac_length_mm
- left_renal_to_internal_iliac_length_mm
- right_renal_to_internal_iliac_length_mm

ACCESS:

- access_vessel_min_diameter_mm
- left_access_vessel_min_diameter_mm
- right_access_vessel_min_diameter_mm
- access_vessel_tortuosity
- left_access_vessel_tortuosity
- right_access_vessel_tortuosity

## Measurement methods

Aortic neck diameter:

- Identify the lowest renal artery using centerline abscissa, not global z-coordinate.
- Lowest renal artery means the renal ostium whose projection onto the abdominal aorta
  centerline is most distal downstream from the aortic inlet.
- Identify left and right renal artery ostium/proximal landmarks from STEP3.
- Project those landmarks onto the abdominal aorta centerline.
- Compare downstream centerline abscissa from the aortic inlet.
- The renal artery with the greater downstream abscissa is the lower renal artery.
- Sample a centerline-orthogonal cross-section immediately distal to the lowest renal
  artery.
- Report major diameter, minor diameter, and equivalent diameter.
- Set `proximal_neck_diameter_mm` to `proximal_neck_equivalent_diameter_mm`.
- Preserve `proximal_neck_major_diameter_mm` and `proximal_neck_minor_diameter_mm`
  separately.
- STEP4 reports geometry only; later device-matching logic may choose which diameter
  type to use for IFU/catalogue logic.

Infrarenal aortic neck treatment diameter:

- Same measurement family as proximal neck diameter.
- Do not use aneurysm sac diameter.
- Map to proximal neck equivalent diameter unless future downstream matching uses a
  different controlled rule.

Aortic treatment diameter:

- For this repository phase, equivalent to infrarenal aortic neck treatment diameter for
  AAA/Conformable geometry measurement.
- Do not infer device sizing here.

Proximal neck length:

- Measure centerline distance from the lowest renal artery reference to a candidate neck
  end.
- `proximal_neck_length_mm` should be a provisional measured value when sufficient
  aortic profile data exists.
- The candidate neck end is a provisional geometric detection based on downstream
  aortic diameter profile expansion.
- Default provisional rule: candidate neck end = first downstream aortic profile
  location where equivalent diameter shows sustained expansion relative to
  D0/proximal baseline across consecutive samples.
- Store `candidate_neck_end_abscissa_mm`, `baseline_diameter_mm`, `expansion_rule`,
  `profile_samples`, `confidence`, and notes/warnings.
- Label the value as provisional/profile-derived, not definitive clinical truth.
- If candidate neck end is not robustly detected, mark `proximal_neck_length_mm` as
  `requires_review`.
- STEP4 must not claim definitive clinical aneurysm-neck-end detection.

Proximal neck angulation:

- Use centerline-derived axes.
- Define a neck axis and a proximal/distal aortic reference axis.
- Store method and vectors when implemented.
- If axis quality is poor, mark `requires_review`.

Diameter profile sampling:

- Use 1-2 mm internal sampling along the aortic centerline where practical.
- Explicitly report clinically useful offsets where possible: D0, D5, D10, D15 from the
  lowest renal artery.
- Store measurement-plane metadata and QA for each sampled plane.
- Use profile samples to support the provisional neck-length detector.

Iliac treatment diameter:

- Use side-specific left/right iliac measurements.
- For the generic field, create a summary object with explicit `summary_rule`.
- Do not silently average left/right.

Distal iliac seal-zone length:

- Measure side-specific centerline length of candidate distal iliac sealing zones.
- Tissue quality is not measured in STEP4.

Common iliac diameter:

- Measure side-specific CIA orthogonal diameter.
- Intended for IBE proximal/common iliac context.
- Preserve left/right values.
- Generic field is summary object only.

Common iliac length:

- Measure centerline distance from aortic bifurcation to iliac bifurcation or
  STEP3-defined CIA distal boundary.
- If landmarks are missing, mark `missing_required_landmark` or `requires_review`.

External iliac treatment diameter:

- Measure side-specific EIA orthogonal diameter profile.
- Preserve left/right values.
- Generic field is summary object only.

External iliac seal-zone length:

- Measure side-specific EIA candidate seal-zone centerline length.
- If no femoral extension exists, the EIA distal endpoint is the most distal external
  iliac point available.

Internal iliac treatment diameter:

- Measure side-specific IIA orthogonal diameter profile if IIA branches are present.
- If IIA is missing, mark IBE-specific IIA fields `unmeasurable` or `requires_review`.
- Do not fail global STEP4.

Internal iliac seal-zone length:

- Measure side-specific IIA centerline length if IIA branches are present.
- If missing, mark `unmeasurable` or `requires_review`.

Renal-to-internal-iliac length:

- Compute side-specific path length from lowest major renal artery reference to internal
  iliac target.
- This is a geometry/path length only.
- Do not decide adequacy for a device or component combination.
- Final adequacy depends on later component lengths and overlap rules.

Access vessel minimum diameter:

- Compute side-specific minimum lumen diameter along access path.
- If femoral arteries are present, include the iliofemoral path.
- If femoral arteries are not present, use the most distal external iliac endpoint
  available.
- Store `access_extent` as `iliofemoral`, `iliac_only`, or `unknown_or_incomplete`.
- Do not assess calcification, plaque, or thrombus.

Access vessel tortuosity:

- Compute side-specific centerline tortuosity.
- Acceptable measures to document for future implementation:
  - path length / straight-line length
  - cumulative curvature
  - maximum bend angle
- Store method used.
- Generic `access_vessel_tortuosity` is a summary object only.

## Side-specific summary rules

Left/right values are primary and authoritative. Generic bilateral fields are structured
summaries and must not blindly collapse side-specific values into a single number.

A generic field must include:

- `status`
- `summary_rule`
- `left_ref`
- `right_ref`
- `left_value`
- `right_value`
- `unit`
- `notes`

Examples of valid `summary_rule`:

- `side_specific_values_preferred`
- `min_of_sides`
- `max_of_sides`
- `bilateral_available`
- `unilateral_only`
- `not_computed_until_downstream_rule`

For this resource contract, use:

```json
{
  "status": "derived_summary",
  "summary_rule": "side_specific_values_preferred",
  "left_ref": "<left_field_name>",
  "right_ref": "<right_field_name>",
  "left_value": null,
  "right_value": null,
  "unit": "mm|degrees|ratio|none",
  "notes": []
}
```

Downstream IFU/device matching rules can later decide whether min, max, side-specific, or
bilateral values are needed.

## Unmeasurable / out-of-scope values

These are not STEP4 lumen-geometry measurements and must not be measured by STEP4 from a
lumen-only model:

- `neck_thrombus_max_thickness_mm`
- `neck_thrombus_circumference_percent`
- `iliac_thrombus_max_thickness_mm`
- `iliac_thrombus_circumference_percent`
- `access_calcification_burden`
- `significant_plaque_flag`
- `device_material_allergy_status`
- `systemic_infection_status`
- `patient_followup_compliance`
- `contrast_agent_tolerance`
- `pregnancy_status`
- `nursing_status`
- `age_limit_if_stated`
- `genetic_connective_tissue_disease_status`
- `branch_vessel_preservation_requirement`
- final `access_vessel_suitability`
- final `introducer_sheath_compatibility`
- final `renal_artery_patency_status`
- final `internal_iliac_patency_status`

Out-of-scope categories include:

- thrombus thickness
- thrombus circumference
- calcification burden
- plaque flag
- material allergy
- systemic infection
- pregnancy/nursing/age/clinical status
- contrast tolerance
- follow-up compliance
- branch preservation requirement
- final sheath compatibility
- final device compatibility
- final access suitability
- final renal/internal iliac patency assessment

STEP4 may localize the geometric regions where tissue review would occur, but must not
label tissue findings.

## Failure and requires_review rules

- Missing all STEP3 core inputs is `failed`.
- STEP3 contract unusable or `failed` is `failed`.
- Missing priority aorta, renal, CIA, or EIA anatomy may cause `requires_review` or
  `failed` depending on severity.
- Missing required anatomy for one measurement group marks that group as
  `unmeasurable` or `requires_review`.
- Missing internal iliac branches should not globally fail STEP4.
- Missing internal iliac branches should make IBE-specific IIA and renal-to-IIA fields
  `unmeasurable` or `requires_review`.
- Standard AAA/Conformable aortic neck and iliac measurements can still succeed without
  IIA.
- Global failure applies only when STEP3 inputs are missing/unusable or required
  aortic/iliac anatomy is so incomplete that STEP4 cannot produce a trustworthy
  contract.

## VTP visualization contract

`step4_evar_geometry_regions.vtp` should eventually visualize:

- proximal neck measurement plane
- D0/D5/D10/D15 profile planes where available
- candidate neck-end marker if generated
- iliac measurement planes
- CIA/EIA/IIA seal-zone regions
- renal-to-IIA path polylines
- access path polylines
- region/status arrays for measured, requires_review, unmeasurable

This file should not contain raw tissue variables.
