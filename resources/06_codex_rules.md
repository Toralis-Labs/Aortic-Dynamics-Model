# Codex Rules

## Primary Rule

Codex must solve only this current problem:

```text
Accurately place clean circular branch_start and bifurcation cut-boundary rings so branch segments begin at the correct ostium/cut boundary and do not include parent/aortic wall.
```

The current code target is:

```text
surface-validated branch_start ring selection
surface assignment correction
```

## Required Reading

Before future code edits, Codex must read:

```text
resources/README.md
resources/01_problem.md
resources/02_target_outputs.md
resources/03_boundary_ring_contract.md
resources/04_algorithm_strategy.md
resources/05_validation_and_iteration.md
resources/06_codex_rules.md
```

Future code prompts must obey the resource folder.

## Resource Editing Rule

Codex must delete obsolete resource wording, not only append new wording.

Resources must stay strict, short, and controlling.

## Output File Rules

Codex must not add new output files unless a future prompt explicitly requests them.

Required output files only:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

Allowed optional compact diagnostic file:

```text
outputs/segmentation_diagnostics.json
```

Forbidden debug output files include:

```text
candidate_rings.vtp
candidate_cuts.vtp
debug_surface_sections.vtp
all_candidates.vtp
raw_cut_components.vtp
branch_clouds.vtp
ostium_debug.vtp
```

## VTP Array Rules

`outputs/segmented_surface.vtp` must contain only:

```text
SegmentId
SegmentLabel
SegmentColor
```

`outputs/boundary_rings.vtp` must contain only:

```text
RingId
RingLabel
RingType
ParentSegmentId
ChildSegmentId
SegmentId
RadiusMm
Confidence
Status
```

Codex must not add new VTP arrays unless explicitly requested.

Do not put candidate metrics or debug metadata into VTP arrays.

## JSON Rules

`outputs/segmentation_result.json` must contain only compact decision metadata.

Required top-level keys only:

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

Codex must not add large JSON debug dumps.

Codex must not store raw candidate points, raw cut contour points, every plane-cut component, or candidate metric arrays by default.

## Algorithm Rules

Codex must use:

```text
surface_validated_branch_start_ring_v1
```

For branch starts, the required strategy is:

```text
stable daughter section first
then backward refinement
then last clean candidate before parent contamination
```

The topology start is only a search origin.

topology-only branch_start rings must be requires_review.

Codex must not accept a `branch_start` ring as `success` when it is supported only by centerline topology.

Codex must not accept the first valid-looking surface cut as final when a stable daughter section and backward refinement have not been performed.

## Surface Assignment Rules

The selected circular ring must update surface cell assignment.

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring must be reassigned to the parent segment.

If `segmented_surface.vtp` and `boundary_rings.vtp` disagree, mark:

```text
requires_review
```

or:

```text
failed
```

## Forbidden Expansion

Codex must not add:

```text
VMTK
vessel-name logic
clinical labels
device planning
measurement extraction
simulation
machine learning
new pipeline stages
broad helper systems
broad architecture rewrites
extra labels
extra debug files
large diagnostics
```

Codex must not change label rules.

Codex must not solve multiple problems at once.

## Uncertainty Rule

When uncertain, Codex must mark:

```text
requires_review
```

Codex must not add more outputs to hide or explain uncertainty.

## Change Report Rule

Every code change must report whether it changed the minimal output contract.

If a code change modifies the minimal output contract, that is a failure unless a future prompt explicitly requested the contract change.
