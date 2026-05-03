# Geometry Segmentation Resources

## Contract Role

This is a strict geometry segmentation workspace.

The resources are constraints, not suggestions. Future code prompts must obey these files before editing code.

The one current target is:

```text
correct circular branch_start and bifurcation cut-boundary placement
```

The current failure is:

```text
branch_start rings are too proximal / too early, especially for steep branches close to the aortic body
```

The next code work must fix only:

```text
surface-validated ring selection logic
surface assignment consistency with the selected ring
```

Codex must prefer deletion, simplification, and tighter contracts over new outputs, new arrays, new JSON fields, broad helper systems, or broad architecture.

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
