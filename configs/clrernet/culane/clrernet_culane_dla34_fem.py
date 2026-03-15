_base_ = [
    "./clrernet_culane_dla34.py",
]

cfg_name = "clrernet_culane_dla34_fem.py"

# Warm-start from baseline checkpoint.
# Existing layers are loaded, newly added FEM params are randomly initialized.
load_from = "clrernet_culane_dla34.pth"

# Keep original test threshold for fair comparison.
model = dict(test_cfg=dict(conf_threshold=0.41))

# Separate output directory for ablation.
work_dir = "work_dirs/clrernet_culane_dla34_fem"
