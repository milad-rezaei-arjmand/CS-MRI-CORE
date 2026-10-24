# Portable release copy; see docs/PORTABILITY.md.
from release_paths import release_workspace, release_upstream
"""
Build the locked Final Holdout cache for CS-MRI-CORE without rebuilding
or overwriting the existing MRI-CORE cache.

This script is preprocessing-only:
- NO model loading
- NO inference
- NO training
- NO thresholding or post-processing of model predictions

It reproduces scripts/build_mricore_cache.py for the locked 50-case
Final Holdout:
  * per-slice min-max normalization
  * uint8 T1c PNGs
  * binary 0/255 mask PNGs
  * identical short_filename() convention
  * identical manifest fields

Safety:
- verifies the locked Final Holdout CSV SHA256
- requires exactly 50 unique cases marked final_holdout
- aborts if any Final Holdout case is already present in cache_manifest.csv
- aborts on filename collision with existing cache
- stages all PNGs before modifying the live cache
- backs up cache_manifest.csv before modification
- atomically replaces the manifest
- writes an audit JSON and a CSV containing only added rows
"""
from pathlib import Path
from collections import Counter
import csv
import re
import json
import hashlib
import shutil
import tempfile
import os
from datetime import datetime
import nibabel as nib
import numpy as np
from PIL import Image
PROJECT = release_workspace()
DATA_ROOT = PROJECT / 'datasets/BraTS-MEN-RT/extracted/BraTS-MEN-RT-Train-v2'
FINAL_CSV = PROJECT / 'reports/dataset_audit/brats_men_rt_final_holdout_seed2026_locked.csv'
EXPECTED_FINAL_CSV_SHA256 = '01dc8be8c1bfd88950959d06a76ba952f9760c249f58204c7ca0b1b9d37af1db'
OUTPUT_ROOT = PROJECT / 'datasets/BraTS-MEN-RT/processed_mricore'
IMAGE_DIR = OUTPUT_ROOT / 'images'
MASK_DIR = OUTPUT_ROOT / 'masks'
CACHE_MANIFEST = OUTPUT_ROOT / 'cache_manifest.csv'
REPORT_DIR = PROJECT / 'reports/final_holdout_evaluation'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ADDITIONS_CSV = REPORT_DIR / 'final_holdout_cache_additions_seed2026.csv'
AUDIT_JSON = REPORT_DIR / 'final_holdout_cache_build_seed2026.json'
EXPECTED_FIELDS = ['case_id', 'z', 'filename', 'original_split', 'orientation', 'spacing_z_mm', 'is_positive', 'gtv_pixels', 'total_z_slices']

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def normalize_uint8(x):
    x = x.astype(np.float32)
    vmin = float(x.min())
    vmax = float(x.max())
    if vmax > vmin:
        x = (x - vmin) / (vmax - vmin)
    else:
        x = np.zeros_like(x)
    return np.uint8(np.clip(x * 255.0, 0, 255))

def short_filename(case_id, z):
    match = re.fullmatch('BraTS-MEN-RT-(\\d+)-(\\d+)', case_id)
    if not match:
        raise ValueError(f'Unexpected case ID: {case_id}')
    subject = match.group(1)
    scan = match.group(2)
    return f'{subject}-{scan}-z{z:03d}.png'

def main():
    print('=' * 68)
    print('LOCKED FINAL-HOLDOUT CACHE BUILD — PREPROCESSING ONLY')
    print('NO MODEL / NO INFERENCE')
    print('=' * 68)
    for p in [FINAL_CSV, CACHE_MANIFEST, IMAGE_DIR, MASK_DIR]:
        if not p.exists():
            raise FileNotFoundError(p)
    actual_final_sha = sha256_file(FINAL_CSV)
    print('\nFinal Holdout CSV SHA256:')
    print('  actual  :', actual_final_sha)
    print('  expected:', EXPECTED_FINAL_CSV_SHA256)
    if actual_final_sha != EXPECTED_FINAL_CSV_SHA256:
        raise RuntimeError('FINAL HOLDOUT CSV SHA256 MISMATCH — ABORT')
    with FINAL_CSV.open(newline='', encoding='utf-8') as f:
        final_rows = list(csv.DictReader(f))
    if len(final_rows) != 50:
        raise RuntimeError(f'Expected exactly 50 Final Holdout rows, found {len(final_rows)}')
    final_cases = [row['case_id'] for row in final_rows]
    if len(set(final_cases)) != 50:
        raise RuntimeError('Final Holdout contains duplicate case IDs')
    if any((row.get('split') != 'final_holdout' for row in final_rows)):
        bad = [row['case_id'] for row in final_rows if row.get('split') != 'final_holdout']
        raise RuntimeError(f'Rows not marked final_holdout: {bad}')
    final_cases = sorted(final_cases)
    final_case_set = set(final_cases)
    manifest_before_sha = sha256_file(CACHE_MANIFEST)
    with CACHE_MANIFEST.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames
        existing_rows = list(reader)
    if existing_fields != EXPECTED_FIELDS:
        raise RuntimeError(f'cache_manifest.csv field order/schema differs from expected:\nactual={existing_fields}\nexpected={EXPECTED_FIELDS}')
    existing_case_set = {row['case_id'] for row in existing_rows}
    case_overlap = sorted(final_case_set & existing_case_set)
    if case_overlap:
        raise RuntimeError(f'At least one Final Holdout case is already present in cache_manifest.csv; refusing to append: {case_overlap}')
    existing_filenames = {row['filename'] for row in existing_rows}
    print('\nExisting cache:')
    print('  manifest rows :', len(existing_rows))
    print('  manifest SHA  :', manifest_before_sha)
    print('  Final overlap :', len(case_overlap))
    case_meta = {}
    for case_id in final_cases:
        case_dir = DATA_ROOT / case_id
        image_path = case_dir / f'{case_id}_t1c.nii.gz'
        mask_path = case_dir / f'{case_id}_gtv.nii.gz'
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        image_nii = nib.load(str(image_path))
        mask_nii = nib.load(str(mask_path))
        if image_nii.shape != mask_nii.shape:
            raise RuntimeError(f'{case_id}: image/mask shape mismatch {image_nii.shape} vs {mask_nii.shape}')
        if not np.allclose(image_nii.affine, mask_nii.affine, rtol=0, atol=1e-05):
            raise RuntimeError(f'{case_id}: image/mask affine mismatch')
        orientation = ''.join(nib.aff2axcodes(image_nii.affine))
        spacing_z = float(image_nii.header.get_zooms()[2])
        total_z = int(image_nii.shape[2])
        case_meta[case_id] = {'image_path': image_path, 'mask_path': mask_path, 'orientation': orientation, 'spacing_z_mm': spacing_z, 'total_z_slices': total_z}
        for z in range(total_z):
            filename = short_filename(case_id, z)
            if filename in existing_filenames:
                raise RuntimeError(f'Filename collision in existing manifest: {filename}')
            image_live = IMAGE_DIR / filename
            mask_live = MASK_DIR / filename
            if image_live.exists() or mask_live.exists():
                raise RuntimeError(f'Untracked existing PNG collision; refusing to overwrite: {filename}')
    expected_new_slices = sum((m['total_z_slices'] for m in case_meta.values()))
    print('\nValidated raw Final Holdout:')
    print('  cases          :', len(final_cases))
    print('  expected slices:', expected_new_slices)
    staging_root = Path(tempfile.mkdtemp(prefix='final_holdout_cache_stage_', dir=str(OUTPUT_ROOT)))
    stage_images = staging_root / 'images'
    stage_masks = staging_root / 'masks'
    stage_images.mkdir()
    stage_masks.mkdir()
    additions = []
    try:
        for case_number, case_id in enumerate(final_cases, start=1):
            meta = case_meta[case_id]
            image_nii = nib.load(str(meta['image_path']))
            mask_nii = nib.load(str(meta['mask_path']))
            image = np.asanyarray(image_nii.dataobj)
            mask = np.asanyarray(mask_nii.dataobj)
            total_z = image.shape[2]
            for z in range(total_z):
                filename = short_filename(case_id, z)
                image_slice = image[:, :, z]
                mask_slice = mask[:, :, z]
                image_u8 = normalize_uint8(image_slice)
                mask_u8 = np.uint8(mask_slice > 0) * 255
                Image.fromarray(image_u8).save(stage_images / filename)
                Image.fromarray(mask_u8).save(stage_masks / filename)
                additions.append({'case_id': case_id, 'z': z, 'filename': filename, 'original_split': 'final_holdout', 'orientation': meta['orientation'], 'spacing_z_mm': meta['spacing_z_mm'], 'is_positive': int(np.any(mask_slice > 0)), 'gtv_pixels': int(np.count_nonzero(mask_slice)), 'total_z_slices': total_z})
            if case_number % 5 == 0 or case_number == len(final_cases):
                print(f'  staged cases: {case_number}/{len(final_cases)}')
        if len(additions) != expected_new_slices:
            raise RuntimeError(f'Staged row count mismatch: {len(additions)} vs {expected_new_slices}')
        staged_image_files = list(stage_images.glob('*.png'))
        staged_mask_files = list(stage_masks.glob('*.png'))
        if len(staged_image_files) != expected_new_slices:
            raise RuntimeError('Staged image PNG count mismatch')
        if len(staged_mask_files) != expected_new_slices:
            raise RuntimeError('Staged mask PNG count mismatch')
        if len({row['filename'] for row in additions}) != len(additions):
            raise RuntimeError('Duplicate filenames generated within Final Holdout')
        additions.sort(key=lambda r: (r['case_id'], int(r['z'])))
        with ADDITIONS_CSV.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
            writer.writeheader()
            writer.writerows(additions)
        additions_sha = sha256_file(ADDITIONS_CSV)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_manifest = REPORT_DIR / f'cache_manifest_before_final_holdout_{timestamp}.csv'
        shutil.copy2(CACHE_MANIFEST, backup_manifest)
        backup_sha = sha256_file(backup_manifest)
        if backup_sha != manifest_before_sha:
            raise RuntimeError('Backup manifest checksum mismatch')
        combined_rows = existing_rows + additions
        combined_rows.sort(key=lambda r: (r['case_id'], int(r['z'])))
        manifest_tmp = REPORT_DIR / f'cache_manifest_with_final_holdout_{timestamp}.tmp.csv'
        with manifest_tmp.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
            writer.writeheader()
            writer.writerows(combined_rows)
        with manifest_tmp.open(newline='', encoding='utf-8') as f:
            verify_rows = list(csv.DictReader(f))
        if len(verify_rows) != len(existing_rows) + len(additions):
            raise RuntimeError('Temporary combined manifest row count mismatch')
        moved_images = 0
        moved_masks = 0
        for row in additions:
            filename = row['filename']
            src_img = stage_images / filename
            src_mask = stage_masks / filename
            dst_img = IMAGE_DIR / filename
            dst_mask = MASK_DIR / filename
            if dst_img.exists() or dst_mask.exists():
                raise RuntimeError(f'Collision appeared during commit: {filename}')
            os.replace(src_img, dst_img)
            moved_images += 1
            os.replace(src_mask, dst_mask)
            moved_masks += 1
        if moved_images != expected_new_slices:
            raise RuntimeError('Not all staged image PNGs were committed')
        if moved_masks != expected_new_slices:
            raise RuntimeError('Not all staged mask PNGs were committed')
        os.replace(manifest_tmp, CACHE_MANIFEST)
        manifest_after_sha = sha256_file(CACHE_MANIFEST)
        with CACHE_MANIFEST.open(newline='', encoding='utf-8') as f:
            final_manifest_rows = list(csv.DictReader(f))
        final_added_rows = [r for r in final_manifest_rows if r['case_id'] in final_case_set]
        if len(final_added_rows) != expected_new_slices:
            raise RuntimeError('Final live manifest does not contain expected number of Final Holdout slices')
        if {r['case_id'] for r in final_added_rows} != final_case_set:
            raise RuntimeError('Final live manifest Final Holdout case set mismatch')
        positive_slices = 0
        per_case_counts = {}
        for case_id in final_cases:
            rows = [r for r in final_added_rows if r['case_id'] == case_id]
            rows.sort(key=lambda r: int(r['z']))
            expected_total = case_meta[case_id]['total_z_slices']
            zs = [int(r['z']) for r in rows]
            if zs != list(range(expected_total)):
                raise RuntimeError(f'Incomplete z coverage after commit: {case_id}')
            for row in rows:
                filename = row['filename']
                if not (IMAGE_DIR / filename).exists():
                    raise RuntimeError(f'Missing live image PNG: {filename}')
                if not (MASK_DIR / filename).exists():
                    raise RuntimeError(f'Missing live mask PNG: {filename}')
                positive_slices += int(row['is_positive'])
            per_case_counts[case_id] = len(rows)
        orientation_counts = Counter((row['orientation'] for row in final_added_rows))
        audit = {'status': 'FINAL_HOLDOUT_CACHE_BUILD_PASS', 'preprocessing_only': True, 'model_inference_performed': False, 'source_script_equivalence': {'normalization': 'per-slice min-max -> uint8', 'mask': 'binary >0 -> uint8 0/255', 'filename': 'subject-scan-zXXX.png'}, 'final_holdout_csv': str(FINAL_CSV), 'final_holdout_csv_sha256': actual_final_sha, 'expected_final_holdout_csv_sha256': EXPECTED_FINAL_CSV_SHA256, 'final_cases': len(final_cases), 'final_case_ids': final_cases, 'new_cached_slices': expected_new_slices, 'new_positive_slices': positive_slices, 'orientation_counts': dict(orientation_counts), 'per_case_slice_counts': per_case_counts, 'cache_manifest_before_sha256': manifest_before_sha, 'cache_manifest_after_sha256': manifest_after_sha, 'cache_manifest_rows_before': len(existing_rows), 'cache_manifest_rows_after': len(final_manifest_rows), 'manifest_backup': str(backup_manifest), 'manifest_backup_sha256': backup_sha, 'additions_csv': str(ADDITIONS_CSV), 'additions_csv_sha256': additions_sha}
        AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print('\n' + '=' * 68)
        print('FINAL HOLDOUT CACHE BUILD: PASS')
        print('=' * 68)
        print('Cases added             :', len(final_cases))
        print('Slices added            :', expected_new_slices)
        print('Positive slices         :', positive_slices)
        print('Orientation counts      :', dict(orientation_counts))
        print('Manifest rows before    :', len(existing_rows))
        print('Manifest rows after     :', len(final_manifest_rows))
        print('Manifest SHA before     :', manifest_before_sha)
        print('Manifest SHA after      :', manifest_after_sha)
        print('Backup                  :', backup_manifest)
        print('Additions CSV           :', ADDITIONS_CSV)
        print('Audit JSON              :', AUDIT_JSON)
        print('\nNO MODEL INFERENCE WAS PERFORMED.')
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
if __name__ == '__main__':
    main()
