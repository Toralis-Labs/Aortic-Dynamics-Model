# Codex Rules

## Primary Rule

Codex must solve only this current problem:

```text
Make segmented_surface.vtp colors obey the selected branch_start rings while keeping boundary_rings.vtp minimal and operational.
```

The current code target is:

```text
ring-to-color consistency
ring-plane-gated surface assignment
visible ring minimalism
```

The branch color must begin at the `branch_start` ring.

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

Codex must not solve ring-color inconsistency by adding more VTP arrays.

Codex must not solve ring clutter by adding more ring files.

Codex must not create debug VTP outputs.

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

Compact JSON may report only bounded ring-to-color consistency metadata allowed by `resources/02_target_outputs.md`.

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

For the next code prompt, Codex must not primarily move rings again unless validation proves the selected ring itself is invalid.

The next code prompt must make surface coloring obey the already selected `branch_start` ring.

## Surface Assignment Rules

The selected circular ring must update surface cell assignment.

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring plane beyond tolerance must be reassigned to the parent segment.

Surface cells on the parent side of the ring plane must be reassigned to parent.

Surface assignment must be ring-plane-gated.

Centerline projection alone is insufficient for steep branches.

After ring-plane gating, keep only the connected distal child-side component.

If `segmented_surface.vtp` and `boundary_rings.vtp` disagree, mark:

```text
requires_review
```

or:

```text
failed
```

If the selected ring and segment coloring disagree, Codex must fix the surface assignment or mark `requires_review`.

Codex must not hide the disagreement with extra diagnostics.

## Visible Ring Rules

Codex must treat `boundary_rings.vtp` as a minimal interface file.

Codex must remove or suppress duplicate visible rings rather than add more.

Default visible rings are:

```text
aortic_body_start
aortic_body_end
branch_start
parent_pre_bifurcation only if distinct and useful
```

`branch_end` rings are not written by default.

Duplicate `daughter_start` rings are not written by default.

If a `daughter_start` concept is needed for JSON topology, it should reference the corresponding `branch_start` ring ID instead of creating duplicate visible geometry.

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

Future code changes must report whether the visible ring count decreased.

Future code changes must report whether `branch_end` rings and duplicate `daughter_start` rings were suppressed.

Future code changes must report whether segment coloring is ring-plane-gated.
