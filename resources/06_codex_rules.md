# Codex Rules

## Primary Rule

Codex must solve only this current problem:

```text
Make segmented_surface.vtp colors begin cleanly at the selected branch_start rings while boundary_rings.vtp shows only operational branch_start rings by default.
```

The current code target is:

```text
branch_start-only visible rings
ring-to-color consistency
ring-plane-gated or clipped surface assignment
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

No extra output files are allowed for this fix.

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

No extra VTP arrays are allowed for this fix.

Codex must not add new VTP arrays unless explicitly requested.

Forbidden VTP arrays include:

```text
ring plane distance
tolerance
clip status
original cell id
debug flags
region id
parent-side violation
connectivity region
candidate metrics
```

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

Topology-only `branch_start` rings must be `requires_review`.

Unproven zero-offset `branch_start` rings are not clean operational rings.

Codex must not accept a `branch_start` ring as `success` when it is supported only by centerline topology.

Codex must not accept a `branch_start` ring selected at `0.0 mm` as a clean visible boundary when `zero_offset_proof_passed` is false.

For the next code prompt, Codex must not primarily move rings again unless validation proves the selected ring itself is invalid.

The next code prompt must make surface coloring obey the already selected `branch_start` ring.

## Surface Assignment Rules

The selected circular ring must update surface cell assignment.

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring plane beyond tolerance must be reassigned to the parent segment, clipped, or counted against success.

Surface cells on the parent side of the ring plane must be reassigned to parent.

Surface assignment must be ring-plane-gated.

Centerline projection alone is insufficient for final surface coloring.

The tolerance must be:

```text
RING_PLANE_ASSIGNMENT_TOLERANCE_MM = 0.10
```

Any tolerance above `0.15 mm` must force `requires_review` unless explicitly justified.

After ring-plane gating, keep only the connected distal child-side component.

The color should follow the child vessel contour, not surrounding vessels.

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

## Cell Recoloring Is Not Enough If The Visual Boundary Is Wrong

The current mesh may contain cells that cross the selected `branch_start` ring plane.

If the code only recolors whole cells based on cell centers, the visible boundary may not align with the circular ring.

Cell recoloring is not enough if the visual boundary is wrong.

If color-ring mismatch remains after ring-plane gating, Codex must implement clipping/splitting with `vtkClipPolyData` or equivalent logic, or mark the result `requires_review`.

Codex must not solve this by adding debug files or extra arrays.

Codex must solve it by making `segmented_surface.vtp` itself visually correct.

Codex must report whether `vtkClipPolyData` or equivalent clipping/splitting was used.

## Visible Ring Rules

Codex must treat `boundary_rings.vtp` as an interface file.

Codex must remove or suppress non-`branch_start` visible rings by default.

Codex must not add more rings to explain uncertainty.

Default visible ring type:

```text
branch_start
```

`visible_ring_count` must equal `branch_start_ring_count` by default.

`non_branch_start_visible_ring_count` must be `0` by default.

`parent_pre_bifurcation` is internal/hidden by default.

`aortic_body_start`, `aortic_body_end`, `branch_end`, and `daughter_start` are internal/hidden by default.

If `daughter_start` topology is needed for JSON, it must reference the corresponding `branch_start` ring ID instead of creating duplicate visible geometry.

## Ambiguous Cell Rules

Ambiguous near-ring cells are not harmless.

Codex must report `ambiguous_near_ring_total` and `high_ambiguity_ring_count`.

Ambiguous cells must be clipped, reassigned, or counted against success.

A high `ambiguous_near_ring_total` must force `requires_review`.

Codex must not report `ring_surface_consistency_status = success` while thousands of cells remain ambiguous around that ring unless those cells were clipped/resolved or explicitly justified.

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

Future code changes must report:

```text
visible_ring_count
branch_start_ring_count
non_branch_start_visible_ring_count
ambiguous_near_ring_total
high_ambiguity_ring_count
clip_boundary_used
clip_boundary_required_count
clip_boundary_unresolved_count
segments_with_color_crossing_ring_count
segments_with_neighbor_contour_leak_count
```
