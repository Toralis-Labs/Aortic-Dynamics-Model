# Implementation TODO

## Cleanup

- [x] Add `.gitignore` for generated outputs, caches, local environments, archives, and
  one-off instruction files.
- [x] Remove hardcoded user-specific VS Code paths.
- [x] Distill resources into active docs and remove raw chat history.
- [x] Remove tracked generated outputs, caches, archives, old prototypes, and duplicate
  demo/raw data.

## STEP2

- [x] Emit canonical top-level contract fields.
- [x] Ensure STEP3 consumes canonical STEP2 without legacy fallback.
- [x] Keep default STEP2 outputs to the three approved core artifacts.
- [x] Write optional debug artifacts only when requested.
- [x] Rewrite STEP2 around face-map-driven outlet tunnels and anatomical cut partitioning.

## STEP3

- [ ] Validate named surface and named centerline outputs against the canonical STEP2
  contract.
- [ ] Keep core VTP arrays minimal.

## STEP4

- [ ] Add future implementation under `src/step4/evar_geometry_measurements.py`.
- [ ] Keep/adjust root wrapper later as needed.
- [ ] Write `step4_measurements.json`.
- [ ] Write `step4_evar_geometry_regions.vtp`.
- [ ] Follow `resources/step4_measurement_contract.md`.

## STEP5

- [x] Add manifest validator.
- [x] Write `pipeline_manifest.json`.
- [x] Write `pipeline_summary.txt`.

## Verification

- [x] Run static import checks.
- [x] Run STEP1/STEP2 smoke test with VMTK available.
- [x] Run STEP3 smoke test against the refreshed STEP2 contract.
- [x] Confirm generated outputs stay ignored by Git.
