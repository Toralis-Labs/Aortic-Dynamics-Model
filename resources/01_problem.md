# Problem Definition

## One Problem

This branch solves only one problem:

```text
Make segmented_surface.vtp colors begin cleanly at the selected branch_start rings while boundary_rings.vtp shows only operational branch_start rings by default.
```

The result must be a minimal segmented surface plus minimal ring and JSON metadata.

## Current Failure

The earlier failure was mostly ring placement.

The current failure is now visible interface minimalism and surface boundary correctness:

```text
branch_start rings are improved, but there are still too many visible circles and colors may still cross the branch_start boundary
```

There are more visible rings than useful visible branch starts.

Some visible rings are not useful for the interface and create visual noise.

Some child segment colors extend proximal to the `branch_start` circle or onto parent/aortic or neighboring vessel surface.

The start circle for some branch parents is still not a valid clean interface ring.

Unproven zero-offset `branch_start` rings are not clean operational rings.

The problem is no longer primarily:

```text
move the branch_start rings again
```

The next code change should make:

```text
segmented_surface.vtp itself visually correct at boundary_rings.vtp branch_start rings
```

The code must prevent child branches from including parent/aortic wall proximal to the selected `branch_start` ring.

The code must make the surface itself visually correct, not just add diagnostics.

## Required Fix

The solution is:

```text
branch_start-only visible rings
ring-to-color consistency
ring-plane-gated or clipped surface assignment
```

The branch color must begin at the `branch_start` ring.

The selected `branch_start` ring defines where the child segment starts.

Surface assignment by centerline projection alone is insufficient for steep branches because parent-wall cells may project onto the child centerline after the selected offset.

Cell recoloring is not enough if the visual boundary is wrong.

If whole-cell recoloring cannot make the visible color boundary match the circular ring, the implementation must use `vtkClipPolyData` or equivalent polygonal clipping/splitting logic, or mark the result `requires_review`.

The topology start is only a search origin.

The final `branch_start` ring must be selected by:

```text
stable daughter section first
backward refinement second
last clean candidate before parent contamination
```

## Out Of Scope

The solution is not:

```text
naming
clinical interpretation
device planning
measurement extraction
simulation
new pipeline stages
extra outputs
extra labels
large diagnostics
broad architecture rewrites
```

Do not solve multiple problems at once.

## Label Rule

The only anatomical/readable output label allowed is:

```text
aortic_body
```

All branch, bifurcation, and ring labels must be anonymous:

```text
branch_001
bifurcation_001
ring_001
```

Do not change the label rules while fixing ring placement.

## Success Definition

Success means:

```text
boundary_rings.vtp contains only operational branch_start rings by default
visible_ring_count must equal branch_start_ring_count
segmented_surface.vtp agrees with boundary_rings.vtp
branch color must begin at the branch_start ring
child color does not extend proximal to the branch_start ring plane beyond 0.10 mm tolerance
the color should follow the child vessel contour, not surrounding vessels
ambiguous near-ring cells are not harmless and are resolved or counted against success
required outputs remain minimal
uncertain topology-only or unproven zero-offset rings are requires_review and hidden from default visible rings
```
