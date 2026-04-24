Save this JSON file next to your VTP file, or in the same project folder.

Recommended usage:
- VTP geometry file: abdominal_model.vtp
- Sidecar mapping file: face_id_to_name.json

The code should:
1. read the VTP,
2. read the ModelFaceID array from the VTP,
3. read face_id_to_name.json,
4. map each ModelFaceID value to the anatomical name.

This file was created from the user's manually identified terminal face IDs.
