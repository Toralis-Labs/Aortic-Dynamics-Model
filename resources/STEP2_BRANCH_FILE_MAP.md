# STEP2 Isolated Branch File Map

## Purpose

This branch isolates Step 2 from the larger pipeline so Codex can work on Step 2 without
being distracted by Step 3, Step 4, Step 5, or future product modules.

## Source Branch

`rishu/clean-up-n-fix`

## New Branch

`step2-isolated-geometry-contract`

## Keep Policy

Keep only Step 2 code, Step 2 shared dependencies, Step 2 resources, Step 1 output files
used as Step 2 inputs, and minimal demo input files required to run Step 2.

## Kept Files

Step 2 entrypoint:

- `step2_geometry_contract.py`

Step 2 implementation:

- `src/step2/__init__.py`
- `src/step2/geometry_contract.py`

Shared `src/common` dependencies imported by Step 2:

- `src/__init__.py`
- `src/common/__init__.py`
- `src/common/geometry.py`
- `src/common/json_io.py`
- `src/common/paths.py`
- `src/common/vtk_helpers.py`

Step 1 input artifacts kept only because Step 2 consumes them:

- `Output files/STEP1/surface_cleaned.vtp`
- `Output files/STEP1/centerline_network.vtp`
- `Output files/STEP1/centerline_network_metadata.json`
- `Output files/STEP1/centerlines_raw_debug.vtp`
- `Output files/STEP1/junction_nodes_debug.vtp`

Demo model and face-map inputs:

- `0044_H_ABAO_AAA/0156_0001.vtp`
- `0044_H_ABAO_AAA/face_id_to_name.json`

Step 2 resources:

- `resources/code_architecture.md`
- `resources/clarification.md`
- `resources/STEP2_BRANCH_FILE_MAP.md`

Minimal repo config:

- `.gitattributes`
- `.gitignore`
- `README.md`

## Removed Files

Step 1 implementation:

- `step1segment.py`

Step 3:

- `step3_naming_orientation.py`
- `src/step3/`

Step 4:

- `step4_infrarenal_neck.py`
- `src/step4/`

Step 5:

- `step5_pipeline_manifest.py`
- `src/step5/`
- `Output files/STEP5/`

Generated outputs and temporary runtime files:

- `Output files/STEP2/`
- `Output files/STEP3/`
- `Output files/STEP4/`
- `step2_stdout.txt`
- `step2_stderr.txt`
- `_check.py`
- `_run_step2.py`

Caches and local tooling noise:

- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- `.env`
- `.venv/`
- `venv/`
- `dist/`
- `build/`
- `*.egg-info/`
- `*.log`
- `.DS_Store`
- `.vscode/`

Unrelated future modules and resources:

- IFU matching
- device selection
- device sizing
- CFD
- WSS
- ML model training
- post-processing prototypes
- broad implementation plans and prompt dumps unrelated to this isolated Step 2 branch

## Step 2 Runtime Inputs

- `Output files/STEP1/surface_cleaned.vtp`
- `Output files/STEP1/centerline_network.vtp`
- `Output files/STEP1/centerline_network_metadata.json`
- `Output files/STEP1/centerlines_raw_debug.vtp`
- `Output files/STEP1/junction_nodes_debug.vtp`
- `0044_H_ABAO_AAA/0156_0001.vtp`
- `0044_H_ABAO_AAA/face_id_to_name.json`

## Step 2 Runtime Outputs

- `Output files/STEP2/segmentscolored.vtp`
- `Output files/STEP2/aorta_centerline.vtp`
- `Output files/STEP2/step2_geometry_contract.json`

## Explicitly Out of Scope

This branch does not include:

- Step 1 implementation
- Step 3 naming
- Step 4 measurements
- Step 5 validation
- IFU matching
- device selection
- device sizing
- CFD
- WSS
- ML model training
- clinical suitability assessment

## Validation Commands

```bash
git status --short
python -m py_compile step2_geometry_contract.py src/step2/geometry_contract.py
python -m py_compile src/common/geometry.py src/common/json_io.py src/common/paths.py src/common/vtk_helpers.py
python -c "from src.step2.geometry_contract import main; print('STEP2 import OK:', callable(main))"
python -c "from pathlib import Path; import sys; forbidden_dirs = ['src/step3', 'src/step4', 'src/step5', 'Output files/STEP3', 'Output files/STEP4', 'Output files/STEP5']; remaining = [p for p in forbidden_dirs if Path(p).exists()]; sys.exit('Forbidden later-step paths still exist: ' + ', '.join(remaining)) if remaining else print('No forbidden later-step directories remain.')"
python -c "from pathlib import Path; import sys; required = ['step2_geometry_contract.py', 'src/step2/geometry_contract.py', 'src/common/paths.py', 'resources/STEP2_BRANCH_FILE_MAP.md']; missing = [p for p in required if not Path(p).exists()]; sys.exit('Missing required Step 2 isolated files: ' + ', '.join(missing)) if missing else print('Required Step 2 isolated files exist.')"
python -c "from pathlib import Path; import sys; required = ['Output files/STEP1/surface_cleaned.vtp', 'Output files/STEP1/centerline_network.vtp', 'Output files/STEP1/centerline_network_metadata.json', 'Output files/STEP1/centerlines_raw_debug.vtp', 'Output files/STEP1/junction_nodes_debug.vtp']; missing = [p for p in required if not Path(p).exists()]; sys.exit('Missing STEP1 runtime inputs: ' + ', '.join(missing)) if missing else print('Required STEP1 runtime inputs exist.')"
python -c "import importlib.util; print('VMTK available:', importlib.util.find_spec('vmtk') is not None)"
```

Runtime validation command when VMTK branch tooling is available:

```bash
python step2_geometry_contract.py
```
