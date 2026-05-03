import sys, traceback, faulthandler
faulthandler.enable()

sys.argv = [
    "step2_geometry_contract.py",
    "--project-root", ".",
    "--input-vtp", "0044_H_ABAO_AAA/0156_0001.vtp",
    "--face-map", "0044_H_ABAO_AAA/face_id_to_name.json",
    "--step1-metadata", "Output files/STEP1/centerline_network_metadata.json",
    "--surface-cleaned", "Output files/STEP1/surface_cleaned.vtp",
    "--centerlines-raw-debug", "Output files/STEP1/centerlines_raw_debug.vtp",
    "--centerline-network", "Output files/STEP1/centerline_network.vtp",
    "--output-dir", "Output files/STEP2",
    "--write-debug",
]

print("importing...", flush=True)
from src.step2.geometry_contract import main
print("calling main...", flush=True)

try:
    raise SystemExit(main())
except SystemExit as e:
    print(f"EXIT: {e.code}", flush=True)
    raise
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
