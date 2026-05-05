# Project Context

This repository is being cleaned into a small, step-based vascular geometry pipeline.
The immediate goal is not final EVAR planning or stent selection. The goal is to create
reliable anatomical geometry contracts that later planning, measurement, simulation, and
device workflows can consume.

## Active Pipeline

- STEP1 preserves the current centerline/topology network extraction.
- STEP2 authors the geometry contract: aorta inlet, pre-bifurcation aorta end, bounded
  vessel-like surface segments, a single aorta trunk centerline, and assignment QA.
- STEP3 consumes STEP2 geometry and adds anatomical vessel names and landmark metadata.
- STEP4 will measure aortic neck, iliac, common iliac, external iliac, internal iliac,
  renal-to-internal-iliac path length, and access-vessel lumen geometry from STEP3
  named anatomy. STEP4 does not measure tissue variables, clinical intake variables,
  device compatibility, or final suitability.
- STEP5 validates required step outputs and aggregates statuses/warnings.

## Core Principles

- Use STEP1 graph/topology as a routing skeleton, not final anatomical authority.
- Use surface-derived boundaries whenever possible for ostia and bifurcations.
- Each step writes exactly one main JSON contract.
- Core VTP outputs stay minimal; diagnostic arrays belong in optional debug files or JSON.
- Generated outputs are reproducible and should not be treated as source files.
