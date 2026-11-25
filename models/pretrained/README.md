# Optional pretrained ALIGNN baseline

Place a compatible ALIGNN PyTorch checkpoint here and update
`train/experiments/optional/O03_pretrained_alignn.yaml`.

The loader accepts a raw state dictionary or a checkpoint containing `model`,
`state_dict`, or `model_state_dict`. Only shape-compatible ALIGNN backbone
parameters are loaded; task heads remain newly initialized. This experiment is
not included in the default pipeline because no checkpoint was supplied.
