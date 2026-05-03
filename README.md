# STEP2 Isolated Geometry Contract Branch

This branch isolates STEP2 from the larger vascular pipeline so geometry-contract work can
be run, diagnosed, and improved without later-stage code or product-module noise.

Source branch: `rishu/clean-up-n-fix`

Active entrypoint:

```bash
python step2_geometry_contract.py
```

## Expected Inputs

STEP2 consumes the preserved STEP1 output artifacts:

- `Output files/STEP1/surface_cleaned.vtp`
- `Output files/STEP1/centerline_network.vtp`
- `Output files/STEP1/centerline_network_metadata.json`
- `Output files/STEP1/centerlines_raw_debug.vtp`
- `Output files/STEP1/junction_nodes_debug.vtp`

STEP2 also uses the minimal checked-in demo model inputs:

- `0044_H_ABAO_AAA/0156_0001.vtp`
- `0044_H_ABAO_AAA/face_id_to_name.json`

## Expected Outputs

STEP2 writes runtime artifacts under `Output files/STEP2/`:

- `segmentscolored.vtp`
- `aorta_centerline.vtp`
- `step2_geometry_contract.json`

Optional debug output is written only when requested by the STEP2 CLI.

## Dependencies

At a high level, STEP2 requires:

- Python
- `numpy`
- `vtk`
- `vmtk` / `vmtkscripts` branch tooling for branch extraction, clipping, and sections

## Excluded

This branch intentionally excludes STEP1 implementation code, STEP3 naming, STEP4
measurements, STEP5 validation, IFU matching, device selection, device sizing, CFD, WSS,
ML training, clinical suitability assessment, archives, generated later-stage outputs,
and unrelated prototypes.

This branch is for STEP2 development only.
