# Aortic Dynamics Model

Contract-driven vascular geometry pipeline for abdominal aortic analysis.

The active development path is:

1. `step1segment.py` preserves the existing STEP1 centerline/topology network.
2. `step2_geometry_contract.py` authors the strict geometry contract.
3. `step3_naming_orientation.py` consumes STEP2 geometry and adds vessel names/landmarks.
4. `step4_infrarenal_neck.py` is the current measurement scaffold.
5. `step5_pipeline_manifest.py` validates step outputs and aggregates status.

Generated outputs are written under `Output files/` and are intentionally ignored by Git.
The checked-in demo input is `0044_H_ABAO_AAA/0156_0001.vtp` with
`0044_H_ABAO_AAA/face_id_to_name.json`.

## Runtime Notes

STEP1 requires VMTK for centerline extraction. `step1segment.py` will automatically
relaunch with a VMTK-capable Python when one is found, including the repo-local ignored
environment at `.tools/m/envs/vmtk-step2/python.exe`. You can also point it at another
interpreter with `CENTERLINE_NETWORK_VMTK_PYTHON`.

STEP2 runs with the project Python/VTK stack after STEP1 has written the canonical
`Output files/STEP1` artifacts.
