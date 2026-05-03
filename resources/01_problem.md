# Problem Definition

## One Problem

This branch solves only one problem:

```text
Make segmented_surface.vtp colors obey the selected visible branch_start rings while keeping boundary_rings.vtp minimal and operational.
```

The result must be a minimal segmented surface plus minimal ring and JSON metadata.

## Current Failure

The earlier failure was mostly ring placement.

The current failure is now surface assignment consistency and visible ring minimalism:

```text
branch_start rings are improved, but colors still cross the ring boundary
```

Some child segment colors extend proximal to the `branch_start` circle.

Too many visible rings also clutter the interface.

Some `daughter_start` rings duplicate `branch_start` rings, and routine `branch_end` rings distract from the segment-start boundaries the user needs to inspect.

The problem is no longer primarily:

```text
move the branch_start rings again
```

The next code change should make:

```text
segmented_surface.vtp respect boundary_rings.vtp
```

The code must prevent child branches from including parent/aortic wall proximal to the selected `branch_start` ring.

## Required Fix

The solution is:

```text
ring-to-color consistency
ring-plane-gated surface assignment
minimal operational visible rings
```

The branch color must begin at the `branch_start` ring.

The selected `branch_start` ring defines where the child segment starts.

Surface assignment by centerline projection alone is insufficient for steep branches because parent-wall cells may project onto the child centerline after the selected offset.

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
branch_start rings do not begin on parent/aortic wall
bifurcation rings are actual circular cut-boundaries
segmented_surface.vtp agrees with boundary_rings.vtp
branch color must begin at the branch_start ring
child color does not extend proximal to the branch_start ring plane beyond tolerance
boundary_rings.vtp is a minimal interface output
segmentation_result.json is compact and bounded
required outputs remain minimal
uncertain topology-only rings are requires_review
```
