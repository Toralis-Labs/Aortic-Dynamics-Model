# Problem Definition

## One Problem

This branch solves only one problem:

```text
Accurately place clean circular branch_start and bifurcation cut-boundary rings so branch segments begin at the correct ostium/cut boundary and do not include parent/aortic wall.
```

The result must be a minimal segmented surface plus minimal ring and JSON metadata.

## Current Failure

The current failure is:

```text
branch_start rings are too proximal / too early
```

This is most visible for steep branches close to the aortic body.

Too proximal means:

```text
the child segment includes parent/aortic wall
```

The code must prevent child branches from including parent/aortic wall.

## Required Fix

The solution is better:

```text
surface-validated ring selection
parent-child surface cell assignment
ring/surface consistency validation
```

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
segmentation_result.json is compact and bounded
required outputs remain minimal
uncertain topology-only rings are requires_review
```
