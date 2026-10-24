# Changes from the research scripts

The scientific computations are retained from the original scripts. The public
copies change the following filesystem and repeat-selection details only:

- `PROJECT` points to this checkout's ignored `.runtime/` directory.
- `MRI_CORE` points to this checkout's ignored `.external/mri_foundation/`.
- The cross-slice import directory points to the public `scripts/` directory.
- Training list/output paths and the recorded repeat number select Repeat00-04
  through the launcher. Random seeds, loss, optimizer, schedule, augmentation,
  architecture, model selection, inference, and metric calculations are retained.
- Python AST formatting removes redundant whitespace/comments. Exact source
  SHA-256 values and exported-file hashes are in `source_provenance.json`.

The launcher defaults to a plan. `--execute` is required to perform research
work on a separate, locally supplied authorized dataset. It never points to the
protected research archive. No training or final-holdout inference was executed
while preparing this release.

The original evaluator remains intentionally specific to the locked 50-case,
five-repeat study. Its checksum and overlap checks have not been relaxed. It
requires separately authorized manifests and checkpoints. It is not a generic
evaluation runner for arbitrary datasets.

The upstream model builder uses CUDA when loading its pretrained state. CPU-only
synthetic tests do not establish that pretrained-model loading works on CPU.
The proposed dependency file follows upstream version guidance and compatible
ranges, rather than claiming to recover the historical experiment environment.
