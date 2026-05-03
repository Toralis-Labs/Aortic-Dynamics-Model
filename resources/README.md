# Geometry Segmentation Resources

This folder defines the rules for the geometry segmentation workspace.

This branch solves one problem only:

> Segment a vascular surface into an aortic body and anonymous connected branch segments using clean circular cut-boundaries.

The code must not solve vessel naming, clinical interpretation, measurement extraction, device matching, simulation, machine learning, or downstream workflow integration.

The only readable anatomical label allowed in outputs is:

```text
aortic_body
```

All other structures must use anonymous geometry labels:

```text
branch_001
branch_002
branch_003
bifurcation_001
ring_001
ring_002
ring_003
```

The circular boundary ring is not decorative.

The circular boundary ring represents the actual cut-boundary used to separate a parent segment from a child segment.

Required inputs live in:

```text
inputs/
```

Required outputs must be written to:

```text
outputs/
```

Required output files:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

These resources are the controlling documents for Codex.

Codex must read these files before editing code:

```text
1. resources/01_problem.md
2. resources/02_target_outputs.md
3. resources/03_boundary_ring_contract.md
4. resources/04_algorithm_strategy.md
5. resources/05_validation_and_iteration.md
6. resources/06_codex_rules.md
```

The code must stay focused on geometry segmentation.

The code must not reintroduce old pipeline language, named branch labels, clinical labels, or extra downstream modules.

This branch is not the full vascular planning pipeline.

This branch is a focused geometry workspace for building, debugging, validating, and improving anonymous vascular surface segmentation using circular cut-boundary rings.