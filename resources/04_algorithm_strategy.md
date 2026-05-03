# Algorithm Strategy

## Scope

Use VTK/NumPy centerline and surface-cut logic.

VMTK must not be imported, required, or reintroduced.

Do not add broad architecture.

Do not create new output types.

Do not solve naming, clinical interpretation, measurement extraction, simulation, or downstream workflow problems.

The next code prompt must focus only on:

```text
surface assignment consistency with the selected branch_start ring
visible ring minimalism
```

The next code change must focus on surface assignment, not ring search.

Do not primarily move rings again unless validation proves the selected ring itself is invalid.

## Required Inputs

Use existing input artifacts from:

```text
inputs/
```

Do not edit inputs.

Do not require vessel-name metadata.

## Required Outputs

Write only:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

Optional compact diagnostics are governed by `resources/05_validation_and_iteration.md`.

No candidate VTPs, all-candidate dumps, branch cloud files, or raw component files are allowed unless a future prompt explicitly requests them.

## Stable Daughter Section + Backward Refinement

For branch starts, the final algorithm strategy is:

```text
stable daughter section first
then backward refinement
then last clean candidate before parent contamination
```

For `branch_start` placement, Codex must not treat the topology start as final.

The implementation must:

```text
1. Start from the branch topology origin.
2. Search distally along the child centerline.
3. Generate surface-cut candidates perpendicular to the child tangent.
4. Find a stable daughter-tube section using compactness, radius plausibility, centroid closeness to centerline, and absence of parent-wall contamination.
5. Use that stable section to establish a reliable daughter radius/contour reference.
6. Walk backward toward the parent in small steps.
7. Stop when parent contamination or unstable mixed parent-child geometry appears.
8. Select the most proximal candidate that is still clean and stable.
9. Use that candidate as the circular branch_start cut-boundary.
10. Reassign surface cells so the surface split is consistent with the selected ring.
```

The selected `branch_start` ring must be the earliest stable daughter-boundary after backward refinement, not the first candidate that produces a cut.

Required branch-start algorithm name:

```text
surface_validated_branch_start_ring_v1
```

## Forbidden Branch-Start Strategy

Do not use:

```text
topology start -> first acceptable candidate -> final ring
```

The topology start is only a search origin.

The final ring requires surface-cut evidence and surface assignment consistency.

## Surface Assignment Correction

The selected circular ring must drive parent-child surface assignment.

Surface cells associated with the child branch but proximal to the selected `branch_start` ring plane beyond tolerance must be reassigned to the parent.

`segmented_surface.vtp` must agree with `boundary_rings.vtp`.

If the ring and surface split disagree, the run status must be:

```text
requires_review
```

or:

```text
failed
```

## Ring-Plane-Gated Surface Assignment

The next code change must focus on surface assignment, not ring search.

Ring-plane-gated surface assignment is required because the branch color must begin at the `branch_start` ring.

Algorithm:

```text
1. Use the already selected branch_start ring.
2. Build a signed plane from the ring center and normal.
3. For each cell assigned to the child segment, calculate signed distance to the ring plane.
4. Reassign parent-side cells to the parent if they are beyond tolerance.
5. Keep only the connected distal child-side component.
6. Reassign isolated proximal/parent-wall patches to the parent.
7. Mark requires_review if coloring and ring still disagree.
```

Surface assignment must use the selected `branch_start` ring plane as a hard gate, not only centerline projection.

Centerline projection alone is insufficient for steep branches because parent-wall cells may project onto the child centerline after the selected offset.

The implementation may use `vtkClipPolyData`, direct ring-plane side tests, VTK connectivity logic, or its own cell-adjacency connected-component cleanup.

The selected method must keep outputs minimal.

The implementation must define and report:

```text
RING_PLANE_ASSIGNMENT_TOLERANCE_MM = 0.20
```

The first implementation may use a fixed or adaptive tolerance in the range:

```text
0.10 mm to 0.30 mm
```

Do not create new VTP arrays, debug VTPs, or extra output files for this logic.

## Surface Connectivity Contract

After ring-plane gating, the child-colored region should be the connected distal child-side component.

If disconnected child-colored patches remain proximal to the ring or isolated around the parent wall, they must be reassigned to parent or the output must be:

```text
requires_review
```

The goal is not just to classify individual cells by nearest centerline.

The goal is a visually continuous child branch segment beginning at the `branch_start` ring.

## Bifurcation Strategy

Bifurcation output must stay minimal.

Use visible rings only when they represent distinct operational boundaries:

```text
parent_pre_bifurcation only if distinct and useful
branch_start for child branch starts
```

Preserve measurable parent-to-daughter separation.

Do not collapse bifurcation rings into one point unless the geometry has no measurable separation.

`branch_end` rings are not written by default.

Duplicate `daughter_start` rings are not written by default.

If a `daughter_start` concept is needed for JSON topology, it should reference the corresponding `branch_start` ring ID instead of creating duplicate visible geometry in `boundary_rings.vtp`.

## Acceptance Authority

The final authority is:

```text
clean circular ring placement
surface-cut evidence
ring-to-color consistency
parent-child surface assignment consistency
minimal output contract compliance
```

Centerline topology is an origin and routing guide only.

It is not final placement authority.
