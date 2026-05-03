# Codex Rules

## Primary Instruction

Codex must solve one problem:

> Produce a segmented vascular surface and circular cut-boundary rings that separate the aortic body and anonymous branch segments.

## Required Reading

Before editing code, Codex must read:

```text
resources/01_problem.md
resources/02_target_outputs.md
resources/03_boundary_ring_contract.md
resources/04_algorithm_strategy.md
resources/05_validation_and_iteration.md
```

## Files Codex May Edit

Codex may edit:

```text
step2_geometry_contract.py
src/step2/geometry_contract.py
src/common/paths.py
src/common/geometry.py
src/common/vtk_helpers.py
src/common/json_io.py
src/common/__init__.py
```

Codex may create or update:

```text
outputs/.gitkeep
```

Codex may update resources only if the target contract changes intentionally.

## Files Codex Must Not Reintroduce

Codex must not recreate old pipeline or downstream files.

Codex must not recreate:

```text
old architecture documents
old clarification documents
old implementation-plan documents
old prompt dumps
old downstream module folders
old downstream entrypoint scripts
old generated output folders
old demo-model folders
```

Required paths are:

```text
inputs/
outputs/
resources/
src/
```

## Forbidden Concepts

Codex must not make the repository about:

```text
vessel naming
clinical labels
measurement extraction
device matching
simulation
model training
downstream workflow stages
```

## Forbidden Output Labels

Codex must not output old named branch labels.

The only anatomical segment label allowed is:

```text
aortic_body
```

## Allowed Output Labels

Allowed labels:

```text
aortic_body
branch_001
branch_002
branch_003
bifurcation_001
ring_001
ring_002
```

## Path Rules

Inputs must be loaded from:

```text
inputs/
```

Outputs must be written to:

```text
outputs/
```

Do not write required outputs to old generated-output paths.

## Input Role Rules

Use:

```text
inputs/input_roles.json
```

Do not use a face-name map.

Do not require vessel names.

The code may use:

```text
face IDs
terminal IDs
graph topology
centerline routes
surface geometry
```

## Dependency Rule

This isolated geometry segmentation branch intentionally avoids VMTK branch tooling and VMTK compiled wrappers. It uses VTK + NumPy + input centerline/surface artifacts.

Codex must not import or require VMTK in this branch.

Codex must use VTK, NumPy, input centerline artifacts, input surface geometry, input roles, and local surface-cut or ring geometry.

Final ring placement must be validated against surface geometry and parent-child segment assignment.

Codex must not accept a branch-start ring only because:

```text
it is located at a centerline graph node
the branch centerline begins there
```

Codex must validate final circular rings using:

```text
actual surface geometry
centerline tangent direction
local radius or diameter evidence
parent-child surface assignment
ring visibility in boundary_rings.vtp
ring consistency with segmented_surface.vtp
```

If surface evidence and ring validation disagree, Codex must mark the relevant ring or segment as:

```text
requires_review
```

unless the code can refine the ring to a better surface-consistent position.

## Boundary Ring Rules

Circular rings are actual cut-boundaries.

They are not decorative.

For branch starts:

```text
ring normal = child branch tangent
```

For parent pre-bifurcation rings:

```text
ring normal = parent segment tangent
```

For daughter-start rings:

```text
ring normal = daughter segment tangent
```

Preferred radius:

```text
local equivalent diameter / 2
```

Fallback radius:

```text
centerline radius
```

A ring must not be marked successful unless it is consistent with both:

```text
surface geometry
parent-child segment assignment
```

## Minimal Change Rule

Codex must not rewrite the whole codebase unless explicitly instructed.

For each iteration, Codex must identify one failure and make the smallest code change that addresses it.

## Validation Required

After every code change, Codex must run at least:

```bash
python -m py_compile step2_geometry_contract.py src/step2/geometry_contract.py
python -m py_compile src/common/paths.py src/common/geometry.py src/common/json_io.py src/common/vtk_helpers.py
```

If the runtime environment supports the needed VTK and NumPy dependencies, Codex must run:

```bash
python step2_geometry_contract.py
```

Then Codex must check for:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

## Required Final Report From Codex

Every Codex response after a code change must report:

```text
files changed
what failure was targeted
what logic changed
what was not changed
compile result
runtime result if run
output files produced
remaining failures
next recommended smallest change
```

## Do Not Claim Success Without Evidence

Codex must not say the task is complete unless:

```text
compile checks pass
required outputs exist
segmentation_result.json contains required fields
VTP files are generated
labels follow the allowed label rules
rings are recorded in JSON
ring placement was validated against surface geometry
```

If ParaView inspection is required but not performed, Codex must say so.

## Single-Problem Rule

If Codex starts adding unrelated concepts, stop.

The only problem is:

```text
anonymous vascular surface segmentation using circular cut-boundary rings
```
