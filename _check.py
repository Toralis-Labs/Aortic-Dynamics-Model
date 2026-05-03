import json

with open("Output files/STEP2/step2_debug.json") as f:
    d = json.load(f)
c = d.get("assignment_counts", {})
for k in ["branch_labeled_parent_aorta_wall_before_rescue", "branch_labeled_parent_aorta_wall_cells",
          "branch_owned_aorta_labeled_cells", "branch_groups_labeled_as_aorta", "aorta_groups_labeled_as_branch"]:
    if k in c:
        print(f"{k}: {c[k]}")

with open("Output files/STEP2/step2_geometry_contract.json") as f:
    g = json.load(f)
print("status:", g.get("step_status"))
for s in g.get("segment_summary", []):
    print(f"  seg {s['segment_id']:2} {str(s['name_hint']):35} cells={s['cell_count']}")
