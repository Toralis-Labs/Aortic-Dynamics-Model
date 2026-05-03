# STEP2 Geometry Contract Architecture

## Purpose

This branch contains only the code, inputs, and documentation needed to run and diagnose
STEP2 geometry-contract authoring.

STEP1 implementation code is intentionally absent. Only the STEP1 output artifacts needed
by STEP2 are preserved as runtime inputs.

## Active Design

STEP2 is the geometry-authoring layer. It consumes STEP1 outputs explicitly, resolves the
abdominal aorta inlet and pre-bifurcation end, creates one aorta trunk polyline, follows
named outlet tunnels through the STEP1 graph, authors anatomical tunnel cuts, partitions
the surface into topology segments, and writes the canonical STEP2 JSON contract.

The active STEP2 implementation uses these locked algorithm identifiers:

- `PROXIMAL_BOUNDARY_SELECTION_ALGORITHM = "vmtk_branch_clip_group_boundary_v1"`
- `MESH_PARTITION_ALGORITHM = "vmtk_branch_clipper_v1"`
- `SURFACE_ASSIGNMENT_ALGORITHM = "vmtk_branch_group_segmentation_v1"`
- `TUNNEL_ASSIGNMENT_ALGORITHM = "face_map_outlet_routes_parent_junction_v1"`

## Output Rules

STEP2 core outputs are:

- `Output files/STEP2/segmentscolored.vtp`
- `Output files/STEP2/aorta_centerline.vtp`
- `Output files/STEP2/step2_geometry_contract.json`

Optional debug outputs are runtime artifacts and are not part of the source branch.

## Cleanup Policy

Only STEP2 code, shared STEP2 dependencies, STEP2 resources, STEP1 output artifacts used
as STEP2 inputs, and minimal demo inputs belong in this branch. Later pipeline stages,
archives, generated runtime outputs, local environments, caches, and prototypes are
removed or ignored.
