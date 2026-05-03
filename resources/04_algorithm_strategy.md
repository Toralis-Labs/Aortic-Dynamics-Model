# Algorithm Strategy

## Scope

Use VTK/NumPy centerline and surface-cut logic.

VMTK must not be imported, required, or reintroduced.

Do not add broad architecture.

Do not create new output types.

Do not solve naming, clinical interpretation, measurement extraction, simulation, or downstream workflow problems.

The next code prompt must focus only on:

```text
branch_start selection logic
surface assignment correction
```

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

Surface cells associated with the child branch but proximal to the selected `branch_start` ring must be reassigned to the parent.

`segmented_surface.vtp` must agree with `boundary_rings.vtp`.

If the ring and surface split disagree, the run status must be:

```text
requires_review
```

or:

```text
failed
```

## Bifurcation Strategy

Bifurcation output must stay minimal.

Use:

```text
one parent_pre_bifurcation ring
one daughter_start ring per child segment
```

Preserve measurable parent-to-daughter separation.

Do not collapse bifurcation rings into one point unless the geometry has no measurable separation.

## Acceptance Authority

The final authority is:

```text
clean circular ring placement
surface-cut evidence
parent-child surface assignment consistency
minimal output contract compliance
```

Centerline topology is an origin and routing guide only.

It is not final placement authority.
