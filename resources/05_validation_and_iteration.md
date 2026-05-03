# Validation And Iteration Protocol

## Purpose

Every code change must be judged against the same target:

> Does the code produce a visually usable segmented surface and circular cut-boundary rings that correctly separate parent and child segments?

Do not judge progress by whether the code preserves old pipeline behavior.

Do not judge progress by vessel names.

## Required Run Command

Preferred command:

```bash
python step2_geometry_contract.py
```

The entrypoint filename may remain for now even though the repository is now conceptually a geometry segmentation workspace.

## Required Output Check

After every run, confirm these files exist:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

If any are missing, the run failed.

## ParaView Check

Open:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
```

Expected:

```text
segmented_surface.vtp opens
boundary_rings.vtp opens
SegmentId exists
SegmentLabel exists
SegmentColor exists
rings are visible
aortic_body is visible
branch segments are visually separable
```

## JSON Check

Open:

```text
outputs/segmentation_result.json
```

Expected top-level keys:

```text
status
inputs
outputs
segments
boundary_rings
bifurcations
warnings
metrics
```

## Label Check

Allowed readable anatomical label:

```text
aortic_body
```

Allowed anonymous labels:

```text
branch_001
branch_002
branch_003
bifurcation_001
ring_001
ring_002
```

If any old named branch label appears in output VTP labels or JSON segment labels, the output fails.

## Ring Check

Every branch segment must have a proximal boundary ring.

Every bifurcation should have:

```text
one parent_pre_bifurcation ring
one daughter_start ring per daughter branch
```

Every ring must have:

```text
center_xyz
normal_xyz
radius_mm
ring_type
parent_segment_id
child_segment_id if applicable
confidence
status
warnings
```

## Boundary Placement Failure Classes

Classify every failure using one of these terms:

```text
missing_output
missing_segment_array
missing_ring
ring_too_proximal
ring_too_distal
ring_wrong_angle
ring_radius_too_large
ring_radius_too_small
parent_wall_in_child_segment
child_start_absorbed_by_parent
fragmented_boundary
surface_not_openable
ring_not_visible
json_missing_required_fields
old_named_branch_label
```

## Status Rules

Use:

```text
success
```

only when all required outputs are written and no required boundary is uncertain.

Use:

```text
requires_review
```

when outputs are written but one or more boundaries are uncertain.

Use:

```text
failed
```

when outputs are missing, unreadable, or the segmentation cannot be trusted.

## Iteration Loop

Every change must follow this loop:

```text
1. Read resources.
2. Identify the smallest specific failure.
3. Change only the code needed for that failure.
4. Run the script.
5. Check output files.
6. Inspect VTPs.
7. Inspect JSON.
8. Classify remaining failures.
9. Repeat only if the next change is clearly linked to the failure.
```

## Stop Conditions

Stop when:

```text
all required outputs are written
segmented_surface.vtp is viewable
boundary_rings.vtp is viewable
aortic_body is the only anatomical label
all other segments are anonymous
rings are circular
rings are recorded in JSON
branch and bifurcation boundaries are not obviously too proximal or too distal
```

If the code cannot reach this state, stop and document the blocker.

## Do Not Hide Uncertainty

If a ring is guessed, mark it as:

```text
requires_review
```

If a segment boundary is uncertain, write a warning.

If a branch cannot be separated reliably, do not silently label it as successful.

## Required Developer Report After Each Iteration

After a code change, report:

```text
files changed
specific failure targeted
what logic changed
what was not changed
compile result
runtime result
outputs produced
remaining failures
next smallest recommended change
```

## Validation Commands

At minimum, run:

```bash
python -m py_compile step2_geometry_contract.py src/step2/geometry_contract.py
python -m py_compile src/common/paths.py src/common/geometry.py src/common/json_io.py src/common/vtk_helpers.py
```

If runtime dependencies are available, run:

```bash
python step2_geometry_contract.py
```

Then check:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```