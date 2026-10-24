# Data and exact-replay requirements

Access BraTS-MEN-RT from its official Synapse record. Keep all data local and
outside Git tracking. The release workspace has the following private layout:

```text
.runtime/
  datasets/BraTS-MEN-RT/
    extracted/BraTS-MEN-RT-Train-v2/<case>/<case>_t1c.nii.gz
    extracted/BraTS-MEN-RT-Train-v2/<case>/<case>_gtv.nii.gz
    processed_mricore/
      images/                       # local PNG cache
      masks/                        # local PNG masks
      cache_manifest.csv
      lists/repeat_00_train.txt      # exactly five centers
      lists/repeat_00_val.txt        # five separate centers
      lists/...                     # repeat_01 through repeat_09
      lists/dev_holdout_full_volume.txt
  reports/
    dataset_audit/brats_men_rt_split_seed2026_final.csv
    dataset_audit/brats_men_rt_final_holdout_seed2026_locked.csv
    fewshot_manifests/fewshot_10_repeats_master.csv
  experiments/
    baseline_mricore/repeat_00/best_model.pth
    mean_fusion_mricore/repeat_00/best_model.pth
    ...                             # repeats 01-04
```

These paths describe inputs supplied by an authorized user; they are not
downloadable from this repository. Do not copy or link the original research
archive as a writable workspace. Keep any new run in this separate checkout.

`build_mricore_cache.py` expects the private split and few-shot master CSVs.
The master CSV has `repeat,role,case_id,z` fields, with `role` equal to `train`
or `val`. Split CSVs use `case_id,split` (and retain any original extra fields).
The cache manifest has `case_id,z,filename,original_split,orientation,spacing_z_mm,
is_positive,gtv_pixels,total_z_slices`. MRI-CORE lists contain one
`image_filename,mask_filename` record per slice. The final CSV requires exactly
50 unique `case_id` rows with `split=final_holdout` and the published lock hash.

The original final evaluation verifies all ten predefined training/validation
manifests for overlap even though only Repeat00-04 enter the published results.
Full-volume masks are reconstructed using native NIfTI shape and spacing.
Binary logits use `> 0` at 256 x 256, followed by nearest-neighbor resizing.
Inference neighbors replicate the edge slice at volume boundaries.

HD95 returns 0 for both-empty masks, and NaN with an explicit status for a
single empty mask. The reported evaluation contained 50 valid HD95 cases for
each method and repeat; no case was excluded. This observed denominator does
not change the implementation's behavior on hypothetical empty predictions.

The manifest checksums in `configs/locked_protocol.json` support provenance,
but cannot reconstruct patient selections. Exact replay of the paper also
requires the authorized original manifests, original MRI-CORE/SAM initialization,
paired trained checkpoints or retraining, and the historical compute environment.
Those restricted/local artifacts are not included. Public code and synthetic
checks alone do not establish bitwise reproduction of the paper's scores.
