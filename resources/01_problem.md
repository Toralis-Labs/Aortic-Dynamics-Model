# Problem Definition

## Single Problem

The code must solve one geometric segmentation problem:

> Given a vascular lumen surface and centerline/topology input artifacts, separate the aortic body and every connected branch or bifurcation segment using clean circular cut-boundaries.

This repository is not solving vessel naming.

This repository is not solving clinical interpretation.

This repository is not solving device matching.

This repository is not solving measurement extraction.

This repository is not solving simulation.

This repository is not solving a downstream pipeline.

## Allowed Semantic Label

Only one anatomical/readable segment label is allowed:

```text
aortic_body
```

Every other segment must be anonymous:

```text
branch_001
branch_002
branch_003
branch_004
```

Every bifurcation marker must be anonymous:

```text
bifurcation_001
bifurcation_002
```

Every boundary ring must be anonymous:

```text
ring_001
ring_002
ring_003
```

## Core Goal

The code must identify:

```text
where the aortic body begins
where the aortic body ends
where each branch segment begins
where each branch segment ends
where bifurcations occur
which child segments come from which parent segment
```

The result must be a surface that can be visually inspected and later manipulated.

The segment boundaries must be clean enough that a branch segment can be selected, separated, isolated, or removed.

## Main Failure Being Solved

The main failure is incorrect boundary placement.

A branch or bifurcation boundary is wrong if it is too proximal.

Too proximal means:

```text
the child segment includes part of the parent segment wall
```

A branch or bifurcation boundary is also wrong if it is too distal.

Too distal means:

```text
the child segment starts inside the daughter branch instead of near its true origin
```

A boundary is also wrong if it is:

```text
missing
fragmented
jagged
multi-patch
not circular
angled incorrectly
not visible in the ring output
not recorded in the JSON output
not consistent with the surface split
```

## Desired Boundary Representation

The desired boundary representation is a clean circular ring.

This ring should be:

```text
centered at the estimated boundary location
oriented according to the local segment direction
sized according to local vessel diameter
visible in the ring VTP
recorded in the JSON output
used as the actual cut-boundary for separating parent and child surface cells
```

The ring does not need to reproduce every irregular ridge or saddle shape of the true surface opening.

The ring is a simplified geometric cut boundary.

The goal is not perfect anatomical surface reconstruction.

The goal is a reliable, clean, circular boundary that is correctly positioned.

## Segment Tree Concept

The model should be treated as a connected geometric tree.

There is one root segment:

```text
aortic_body
```

Connected child segments are anonymous:

```text
branch_001
branch_002
branch_003
```

A branch can itself be a parent to additional branches.

Example:

```text
aortic_body
  branch_001
    branch_004
    branch_005
  branch_002
  branch_003
```

The final JSON does not need to use this exact nested text format.

However, it must preserve parent-child relationships.

## Bifurcation Concept

A bifurcation is a location where one parent segment splits into two or more child segments.

At a bifurcation, the output should preserve geometry, not collapse all rings into one point.

Preferred representation:

```text
one parent_pre_bifurcation ring
one daughter_start ring for each child segment
```

The distance between these rings matters.

If there is measurable distance between the parent ring and daughter rings, the JSON must preserve that distance.

If the distance is effectively zero, the JSON must state that explicitly.

## What Codex Must Focus On

Codex must focus on:

```text
geometry
surface segmentation
centerline topology
parent-child segment relationships
circular cut-boundary rings
output VTP files
output JSON file
validation status
```

## What Codex Must Not Focus On

Codex must not focus on:

```text
vessel naming
clinical labels
device selection
measurement extraction
simulation
CFD
WSS
machine learning
downstream workflow stages
```

Old named branch labels from previous input metadata must not control the output.

The only output label with anatomical meaning is:

```text
aortic_body
```