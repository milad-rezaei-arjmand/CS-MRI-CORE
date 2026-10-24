# Release validation

Preparation date: 2026-10-24. This record concerns the release copy only.

- Seven synthetic/static tests passed on Python 3.12.14, Windows x64, NumPy
  2.3.5 and SciPy 1.18.1.
- All fourteen documented command plans (five actions, both training methods
  across five repeats) completed without creating a research workspace.
- Six portable script copies matched their source computational AST after
  excluding only the declared path assignments and repeat metadata.
- All recorded upstream dependency hashes matched the inspected official
  MRI-CORE commit in `configs/upstream_required_files.json`.
- The published repeat table reproduced both primary paired Wilcoxon statistics,
  mean gains and aggregate rounding without medical data or model inference.
- Synthetic HD95 checks covered empty masks and anisotropic physical spacing.

PyTorch/torchvision/MONAI/nibabel were unavailable in the preparation runtime.
Their import, GPU loading, optimization, full data pipeline, and inference
were not tested. No MRI image, research checkpoint, or locked holdout was
evaluated, and no new scientific score was produced. Installing the complete
suggested environment is not a claim of historical bitwise reproducibility.

The official MRI-CORE README's license link could not be verified as an existing
LICENSE file. Its source/weights are excluded pending applicable rights checks.
The existing repository's MIT LICENSE was preserved unchanged. Its scope for
the new release is original Python code. Figures are independently excluded;
no MIT or Creative Commons license is assigned to them. Test results do not
verify ownership or publication permission, which require independent
confirmation from the relevant rights holders.
