# Geometry Segmentation Resources

## Contract Role

This is a strict geometry segmentation workspace.

The resources are constraints, not suggestions. Future code prompts must obey these files before editing code.

The one current target is:

```text
ring-to-color consistency
```

The current failure is:

```text
branch_start rings are improved, but child branch colors can still cross proximal to the visible branch_start ring
```

The next code work must fix only:

```text
ring-plane-gated surface assignment
surface coloring that begins exactly at the selected branch_start ring
minimal operational visible rings
```

The branch color must begin at the `branch_start` ring.

The selected `branch_start` ring is the visible operational boundary, not only a marker.

`boundary_rings.vtp` is a minimal interface output, not a debug dump.

Codex must prefer deletion, suppression of duplicate visible rings, simplification, and tighter contracts over new outputs, new arrays, broad helper systems, or broad architecture.

Codex must not add new files or VTP arrays to solve uncertainty.

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
