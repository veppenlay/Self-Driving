"""Geometry-decoupled lane control exploration package.

Reusable, dependency-light building blocks for the "geometric decoupling"
exploration described in the plan `robust-lane-control-exploration`:

- `controller`  : frozen lateral controllers (PID / Stanley / Pure-Pursuit),
                  numpy-only, gains calibrated once on the source domain.
- `perturb`     : purely synthetic appearance perturbations (lighting / blur /
                  noise / frame-drop) with ZERO target-domain priors.
- `eval_harness`: offline proxy scoring of a candidate's per-frame predictions
                  against the CV geometric reference, plus clean-vs-perturbed
                  degradation tables.
- `calibrate_controller`: fit + freeze controller gains on the source domain.

The heavier torch/cv2 pieces (affordance training, segmentation training,
online CV perception, label extension) live in sibling modules and are meant to
run on the remote GPU environment.
"""
