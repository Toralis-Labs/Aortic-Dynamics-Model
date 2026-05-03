# Algorithm Strategy

## High-Level Strategy

The code should use the input vascular surface and centerline/topology artifacts to produce:

```text
aortic_body
anonymous branch segments
anonymous bifurcation records
circular cut-boundary rings
parent-child topology
segmentation result JSON
```

The algorithm must stay geometry-first.

It must not become a vessel-naming algorithm.

## Inputs

Use:

```text
inputs/surface_cleaned.vtp
inputs/centerline_network.vtp
inputs/centerline_network_metadata.json
inputs/input_roles.json
```

The input roles file identifies:

```text
aortic inlet face
terminal outlet faces
```

It must not identify named vessels.

## Core Algorithm Steps

### 1. Load Input Artifacts

Load:

```text
surface_cleaned.vtp
centerline_network.vtp
centerline_network_metadata.json
input_roles.json
```

Validate:

```text
surface exists
centerline network exists
metadata exists
aortic inlet face exists
terminal outlet faces exist
```

### 2. Build Centerline Graph

Use the centerline network to build a graph.

The graph should support:

```text
node coordinates
edge IDs
edge lengths
parent-child routing
branch paths
bifurcation detection
```

### 3. Resolve Aortic Body

Use the input aortic inlet face to identify the root.

Resolve the aortic body as the main root segment.

The output label must be:

```text
aortic_body
```

Do not use older trunk names.

Only use:

```text
aortic_body
```

### 4. Identify Anonymous Branch Segments

Every connected branch segment after the aortic body should be anonymous.

Use:

```text
branch_001
branch_002
branch_003
```

Branch IDs should be stable within one run.

The exact numbering does not need to imply anatomy.

### 5. Identify Bifurcations

A bifurcation occurs where a parent segment splits into two or more child segments.

Bifurcations should be recorded as anonymous structures:

```text
bifurcation_001
bifurcation_002
```

For each bifurcation, record:

```text
parent_segment_id
child_segment_ids
parent_pre_bifurcation_ring_id
daughter_start_ring_ids
```

### 6. Estimate Boundary Ring Locations

For every branch start, estimate the best boundary location.

A branch-start ring should be:

```text
as close as possible to the true branch origin
without including parent wall inside the child segment
without starting too far inside the child branch
```

For bifurcations, estimate:

```text
parent_pre_bifurcation ring
daughter_start rings
```

The algorithm must preserve measurable distances between parent and daughter rings.

### 7. Estimate Ring Orientation

Branch-start ring:

```text
normal = child branch tangent
```

Parent pre-bifurcation ring:

```text
normal = parent segment tangent
```

Daughter-start ring:

```text
normal = daughter segment tangent
```

### 8. Estimate Ring Radius

Preferred:

```text
radius = local equivalent diameter / 2
```

Fallback:

```text
radius = centerline radius
```

If radius is estimated with weak evidence, mark the ring as:

```text
requires_review
```

### 9. Use Rings as Cut Boundaries

The circular ring must be used as the actual cut-boundary.

The segmentation should assign parent and child surface cells consistently with the ring.

The final surface segmentation should not depend only on a visual ring if the surface cells are separated differently.

### 10. Write Outputs

Write:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

## Dependency Strategy

This isolated geometry segmentation branch intentionally avoids VMTK branch tooling and VMTK compiled wrappers. It uses VTK + NumPy + input centerline/surface artifacts.

VMTK branch tooling is intentionally out of scope for this isolated branch. The implementation uses VTK/NumPy centerline and surface-cut logic instead.

The final authority is whether the circular ring correctly separates parent and child surface geometry.

The correct strategy is:

```text
centerline topology and tangent direction
+ local surface evidence
+ circular ring candidate evaluation
+ parent-child surface assignment checks
= final accepted boundary ring
```

A ring must not be accepted only because it is located at a centerline graph node.

The ring must be checked against the actual surface and parent-child segment assignment.

## Helper Tool Role

Existing VTK, NumPy, and general Python centerline/surface helpers may be used.

Allowed helper use:

```text
initial branch grouping
centerline splitting
surface partition estimate
surface cut proposal
branch section estimation from local surface evidence
surface partition validation
```

But helper output is not final authority.

If a helper-produced boundary is wrong, the code must refine or replace it.

The final authority is whether the circular cut-boundary ring correctly separates parent and child geometry.

## Ring-First Refinement Strategy

The best strategy is to treat branch boundaries as ring placement problems.

For each branch:

```text
1. Find the branch centerline path.
2. Estimate the branch origin region from centerline topology.
3. Estimate a local diameter or radius from surface cuts or nearby surface points.
4. Place a circular ring perpendicular to the local branch tangent.
5. Test whether the ring is too proximal or too distal.
6. Test whether the ring is consistent with the actual surface and parent-child partition.
7. Use the ring as the parent-child cut boundary.
8. Record confidence and warnings.
```

For bifurcations:

```text
1. Find the parent segment.
2. Find daughter segments.
3. Place a parent_pre_bifurcation ring on the parent.
4. Place daughter_start rings on each daughter.
5. Preserve the measurable distance between these rings.
6. Validate the rings against the surface and parent-child partition.
7. Record topology and ring IDs in JSON.
```

## Ring Candidate Acceptance Strategy

A circular ring candidate should be accepted only when it passes geometric checks.

Required checks:

```text
ring is near the branch origin
ring normal follows the local child or parent tangent
ring radius is consistent with local branch size
ring does not include obvious parent wall inside the child segment
ring does not start too far distal inside the child branch
ring is consistent with local surface evidence
ring is consistent with final surface cell assignment
ring is visible in boundary_rings.vtp
ring is recorded in segmentation_result.json
```

If any required check is uncertain, the ring should be marked:

```text
requires_review
```

If a ring is only supported by centerline topology and not by surface evidence, it should not be marked successful.

## Forbidden Algorithm Goals

Do not optimize for vessel names.

Do not optimize for clinical labels.

Do not introduce device, simulation, or measurement objectives.

Do not reframe this repository as part of a larger workflow.

## Minimal Acceptable Version

A minimal acceptable version must:

```text
load inputs
use VTK/NumPy centerline and surface-cut logic
write segmented_surface.vtp
write boundary_rings.vtp
write segmentation_result.json
label only aortic_body by name
label all other segments anonymously
create visible circular rings
record every ring in JSON
mark uncertainty clearly
```

## Better Version

A better version additionally:

```text
uses rings as actual cut boundaries
preserves parent-child topology
handles branches of branches
handles bifurcations with parent and daughter rings
validates rings against actual surface geometry
reports confidence and failure modes
```

## Best Version

A best version:

```text
places every branch and bifurcation ring correctly
avoids parent wall contamination
avoids overly distal branch starts
keeps rings circular and visually clean
preserves measurable distance between parent and daughter bifurcation rings
creates ParaView-friendly outputs every run
records disagreements between surface evidence and ring validation
```
