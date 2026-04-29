# STEP4 Geometry Measurement Clarification Questions

These questions are limited to Step 4 geometry measurement and current pipeline integration.

- Does Step 3 always provide reliable proximal boundary metadata for `left_renal_artery` and `right_renal_artery`?
- Does `named_centerlines.vtp` store anatomical names as cell data, point data, or field data?
- Is `abdominal_aorta_trunk` oriented from proximal/inlet to distal/bifurcation?
- Does `named_segmentscolored.vtp` include the full aortic lumen from renal arteries to bifurcation?
- Are `left_common_iliac` and `right_common_iliac` consistently named?
- Are `left_external_iliac` and `right_external_iliac` consistently named when present?
- Are `left_internal_iliac` and `right_internal_iliac` consistently named when present?
- What exact rule should define the distal end of the infrarenal neck?
- What exact geometric definition should be used for proximal neck angulation?
- Should `neck_reference_diameter_mm` use maximum major diameter or equivalent diameter?
- For the labeled VTP, should the infrarenal neck be defined from lowest renal to algorithmic `neck_end`, or from lowest renal to a fixed IFU-relevant length such as 10 or 15 mm?
