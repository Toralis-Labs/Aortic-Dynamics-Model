# Geometry Segmentation Resources

## Contract Role

This is a strict geometry segmentation workspace.

The resources are constraints, not suggestions. Future code prompts must obey these files before editing code.

The one current target is:

```text
minimal branch_start-only visible rings with surface colors beginning cleanly at those rings
```

The current failure is:

```text
branch_start rings are improved, but boundary_rings.vtp still shows extra visible circles and child branch colors can cross the visible branch_start boundary
```

The next code work must fix only:

```text
branch_start-only visible rings
ring-to-color consistency
surface coloring that begins at the selected branch_start ring
RING_PLANE_ASSIGNMENT_TOLERANCE_MM = 0.10
clipped ring boundary when recoloring is insufficient
```

`boundary_rings.vtp` is an interface file.

By default, `boundary_rings.vtp` must contain only visible operational `branch_start` rings.

The branch color must begin at the `branch_start` ring.

The selected `branch_start` ring is the visible operational boundary, not only a marker.

If whole-cell recoloring cannot make the visible `SegmentColor` boundary match the ring, the implementation must use `vtkClipPolyData` or equivalent polygonal clipping/splitting logic.

Codex must prefer deletion, suppression of non-operational visible rings, simplification, and tighter contracts over new outputs, new arrays, broad helper systems, or broad architecture.

There must be no extra VTP arrays and no extra output files to explain this problem.

VTP and JSON outputs are intentionally minimal.

## Required Outputs

Required output files only:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

Optional compact diagnostic file:

```text
outputs/segmentation_diagnostics.json
```

No other output file is required or allowed unless a future prompt explicitly requests it.

## Required Strategy

For branch starts:

```text
stable daughter section first
then backward refinement
then last clean candidate before parent contamination
```

The topology start is only a search origin.

It is not the final ostium by default.

## Required Reading

Before future code edits, read:

```text
resources/01_problem.md
resources/02_target_outputs.md
resources/03_boundary_ring_contract.md
resources/04_algorithm_strategy.md
resources/05_validation_and_iteration.md
resources/06_codex_rules.md
```
