# Toralis Vascular Pipeline Code Architecture

## Purpose

The codebase converts a vascular lumen surface into contract-driven anatomical geometry
that can later support EVAR measurements, IFU matching, CFD preparation, and device
planning. The current phase starts from a demo `.vtp` model and a manually prepared
`face_id_to_name.json`; DICOM-to-surface segmentation remains outside this repository
phase.

## Active Design

- STEP1 is preserved. It sanitizes the lumen surface, computes VMTK centerlines, rebuilds
  the shared centerline network, and writes the STEP1 metadata contract.
- STEP2 is the geometry-authoring layer. It consumes STEP1 outputs explicitly, resolves
  the abdominal aorta inlet and pre-bifurcation end, creates one aorta trunk polyline,
  follows named outlet tunnels through the STEP1 graph, authors anatomical tunnel cuts,
  partitions the surface into topology segments, and writes the canonical STEP2 JSON
  contract.
- STEP3 is the naming and landmark layer. It consumes STEP2 as geometric truth and adds
  vessel names, priority classification, proximal/distal metadata, and named VTP outputs.
- STEP4 is the EVAR lumen-geometry measurement layer. It will consume STEP3 named
  anatomy and write grouped aortic neck, iliac, renal-to-internal-iliac path, and access
  measurements plus visual/debug measurement regions.
- STEP4 is not tissue assessment, IFU matching, component selection, or device
  recommendation.
- STEP5 is the validation layer. It checks required outputs, validates contract statuses,
  and writes a pipeline manifest plus human summary.

## Output Rules

- Each step has required core outputs and optional debug outputs.
- Each step writes one authoritative machine-readable JSON contract.
- STEP2 core outputs are `segmentscolored.vtp`, `aorta_centerline.vtp`, and
  `step2_geometry_contract.json`.
- STEP3 core outputs are `named_segmentscolored.vtp`, `named_centerlines.vtp`, and
  `step3_naming_orientation_contract.json`.
- STEP4 core outputs are `step4_measurements.json` and
  `step4_evar_geometry_regions.vtp`.
- STEP5 core outputs are `pipeline_manifest.json` and `pipeline_summary.txt`.

## Cleanup Policy

Only active code, shared helpers, docs, and one minimal demo input belong in Git. Generated
outputs, local environments, caches, old prototype scripts, archives, and raw chat dumps
are removed or ignored.
