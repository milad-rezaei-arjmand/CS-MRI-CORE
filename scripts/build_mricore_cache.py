# Portable release copy; see docs/PORTABILITY.md.
from release_paths import release_workspace, release_upstream
from pathlib import Path
from collections import defaultdict
import csv
import re
import nibabel as nib
import numpy as np
from PIL import Image
PROJECT = release_workspace()
DATA_ROOT = PROJECT / 'datasets/BraTS-MEN-RT/extracted/BraTS-MEN-RT-Train-v2'
SPLIT_FILE = PROJECT / 'reports/dataset_audit/brats_men_rt_split_seed2026_final.csv'
FEWSHOT_MASTER = PROJECT / 'reports/fewshot_manifests/fewshot_10_repeats_master.csv'
OUTPUT_ROOT = PROJECT / 'datasets/BraTS-MEN-RT/processed_mricore'
IMAGE_DIR = OUTPUT_ROOT / 'images'
MASK_DIR = OUTPUT_ROOT / 'masks'
LIST_DIR = OUTPUT_ROOT / 'lists'
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
LIST_DIR.mkdir(parents=True, exist_ok=True)

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
with open(SPLIT_FILE, newline='', encoding='utf-8') as f:
    split_rows = list(csv.DictReader(f))
split_by_case = {r['case_id']: r for r in split_rows}
test_cases = sorted((r['case_id'] for r in split_rows if r['split'] == 'test'))
with open(FEWSHOT_MASTER, newline='', encoding='utf-8') as f:
    fewshot_rows = list(csv.DictReader(f))
required_by_case = defaultdict(set)
central_samples = defaultdict(lambda: defaultdict(list))
for row in fewshot_rows:
    repeat = int(row['repeat'])
    role = row['role']
    case_id = row['case_id']
    z = int(row['z'])
    central_samples[repeat][role].append((case_id, z))
    for zz in [z - 1, z, z + 1]:
        required_by_case[case_id].add(zz)
for case_id in test_cases:
    case_dir = DATA_ROOT / case_id
    image_path = case_dir / f'{case_id}_t1c.nii.gz'
    nii = nib.load(str(image_path))
    total_z = nii.shape[2]
    required_by_case[case_id].update(range(total_z))
print('=== MRI-CORE CACHE PLAN ===')
print('Cases requiring cached slices:', len(required_by_case))
print('Test cases:', len(test_cases))
print('Few-shot records:', len(fewshot_rows))
print('Requested unique slices:', sum((len(v) for v in required_by_case.values())))
manifest_rows = []
cases = sorted(required_by_case.keys())
for case_number, case_id in enumerate(cases, start=1):
    case_dir = DATA_ROOT / case_id
    image_path = case_dir / f'{case_id}_t1c.nii.gz'
    mask_path = case_dir / f'{case_id}_gtv.nii.gz'
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    image = np.asanyarray(image_nii.dataobj)
    mask = np.asanyarray(mask_nii.dataobj)
    if image.shape != mask.shape:
        raise RuntimeError(f'{case_id}: image/mask mismatch')
    total_z = image.shape[2]
    orientation = ''.join(nib.aff2axcodes(image_nii.affine))
    spacing_z = float(image_nii.header.get_zooms()[2])
    valid_z = sorted((z for z in required_by_case[case_id] if 0 <= z < total_z))
    for z in valid_z:
        filename = short_filename(case_id, z)
        image_out = IMAGE_DIR / filename
        mask_out = MASK_DIR / filename
        image_slice = image[:, :, z]
        mask_slice = mask[:, :, z]
        image_u8 = normalize_uint8(image_slice)
        mask_u8 = np.uint8(mask_slice > 0) * 255
        if not image_out.exists():
            Image.fromarray(image_u8).save(image_out)
        if not mask_out.exists():
            Image.fromarray(mask_u8).save(mask_out)
        manifest_rows.append({'case_id': case_id, 'z': z, 'filename': filename, 'original_split': split_by_case[case_id]['split'], 'orientation': orientation, 'spacing_z_mm': spacing_z, 'is_positive': int(np.any(mask_slice > 0)), 'gtv_pixels': int(np.count_nonzero(mask_slice)), 'total_z_slices': total_z})
    if case_number % 10 == 0:
        print(f'Processed cases: {case_number}/{len(cases)}')
CACHE_MANIFEST = OUTPUT_ROOT / 'cache_manifest.csv'
manifest_rows = sorted(manifest_rows, key=lambda r: (r['case_id'], int(r['z'])))
with open(CACHE_MANIFEST, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
    writer.writeheader()
    writer.writerows(manifest_rows)
for repeat in range(10):
    for role in ['train', 'val']:
        samples = central_samples[repeat][role]
        output = LIST_DIR / f'repeat_{repeat:02d}_{role}.txt'
        with open(output, 'w', encoding='utf-8') as f:
            for case_id, z in samples:
                filename = short_filename(case_id, z)
                f.write(f'{filename},{filename}\n')
manifest_lookup = {(row['case_id'], int(row['z'])): row for row in manifest_rows}
TEST_FULL = LIST_DIR / 'test_full_volume.txt'
TEST_POSITIVE = LIST_DIR / 'test_positive_only.txt'
full_count = 0
positive_count = 0
with open(TEST_FULL, 'w', encoding='utf-8') as f_full, open(TEST_POSITIVE, 'w', encoding='utf-8') as f_pos:
    for case_id in test_cases:
        case_rows = sorted([r for r in manifest_rows if r['case_id'] == case_id], key=lambda r: int(r['z']))
        for row in case_rows:
            filename = row['filename']
            line = f'{filename},{filename}\n'
            f_full.write(line)
            full_count += 1
            if int(row['is_positive']) == 1:
                f_pos.write(line)
                positive_count += 1
for repeat in range(10):
    for role in ['train', 'val']:
        file_path = LIST_DIR / f'repeat_{repeat:02d}_{role}.txt'
        with open(file_path, encoding='utf-8') as f:
            lines = [x.strip() for x in f if x.strip()]
        assert len(lines) == 5
assert positive_count == 1340
print('\n=== CACHE SUMMARY ===')
print('Cached image slices:', len(manifest_rows))
print('Full-volume Test slices:', full_count)
print('Positive Test slices:', positive_count)
print('Train/Val lists:', 20)
print('\nImages:')
print(IMAGE_DIR)
print('\nMasks:')
print(MASK_DIR)
print('\nLists:')
print(LIST_DIR)
print('\nManifest:')
print(CACHE_MANIFEST)
print('\nMRI-CORE CACHE BUILD: PASS')
