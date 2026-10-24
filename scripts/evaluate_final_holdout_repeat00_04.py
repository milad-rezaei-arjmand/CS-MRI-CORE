# Portable release copy; see docs/PORTABILITY.md.
from release_paths import release_workspace, release_upstream
"""
Locked Final-Holdout Evaluation for CS-MRI-CORE, Repeat00-Repeat04.

Two-stage usage
---------------
1) SAFETY/PREP ONLY (no model inference):
   python scripts/evaluate_final_holdout_repeat00_04.py --prepare-only

2) After PREP PASS, one-time final evaluation:
   python scripts/evaluate_final_holdout_repeat00_04.py --evaluate

Scientific lock
---------------
- Final-holdout source CSV SHA256 is hard-coded and verified.
- Exactly 50 final-holdout cases are required.
- Zero overlap is required with Development Holdout and all Repeat00-09
  train/validation manifests.
- A deterministic full-volume list is created only from cache_manifest.csv.
- Single and Cross methods use the same official Repeat00-04 checkpoints.
- Threshold is fixed at logit > 0.
- Cross context is fixed to z-1, z, z+1 with edge replication and arithmetic
  mean of embeddings.
- No retraining, threshold sweep, LCC, calibration, or post-processing.
- Completed jobs are resumable: a valid existing summary is skipped.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import csv
import json
import re
import time
import hashlib
import argparse
import itertools
import statistics
from collections import Counter
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
PROJECT = release_workspace()
MRI_CORE = release_upstream()
FINAL_CSV = PROJECT / 'reports/dataset_audit/brats_men_rt_final_holdout_seed2026_locked.csv'
EXPECTED_FINAL_CSV_SHA256 = '01dc8be8c1bfd88950959d06a76ba952f9760c249f58204c7ca0b1b9d37af1db'
RAW_DATA = PROJECT / 'datasets/BraTS-MEN-RT/extracted/BraTS-MEN-RT-Train-v2'
PROCESSED = PROJECT / 'datasets/BraTS-MEN-RT/processed_mricore'
IMAGE_DIR = PROCESSED / 'images'
MASK_DIR = PROCESSED / 'masks'
LIST_DIR = PROCESSED / 'lists'
CACHE_MANIFEST = PROCESSED / 'cache_manifest.csv'
DEV_HOLDOUT_LIST = LIST_DIR / 'dev_holdout_full_volume.txt'
FINAL_LIST = LIST_DIR / 'final_holdout_full_volume_seed2026.txt'
FINAL_LIST_SHA = Path(str(FINAL_LIST) + '.sha256')
LOCK_DIR = PROJECT / 'reports/final_holdout_evaluation'
LOCK_DIR.mkdir(parents=True, exist_ok=True)
PREP_REPORT = LOCK_DIR / 'final_holdout_prepare_lock.json'
OUTPUT_ROOT = PROJECT / 'experiments/final_holdout_evaluation'
REPEATS = list(range(5))
SEED = 0
BATCH_SIZE = 1
SINGLE_CKPT = {r: PROJECT / f'experiments/baseline_mricore/repeat_{r:02d}/best_model.pth' for r in REPEATS}
CROSS_CKPT = {r: PROJECT / f'experiments/mean_fusion_mricore/repeat_{r:02d}/best_model.pth' for r in REPEATS}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_sha_file(path, digest):
    Path(path).write_text(f'{digest}  {FINAL_LIST.name}\n', encoding='utf-8')

def first_token(line):
    return line.strip().split(',')[0].strip()

def read_nonempty_lines(path):
    with open(path, encoding='utf-8') as f:
        return [x.strip() for x in f if x.strip()]

def list_case_ids(path, filename_info):
    ids = set()
    unknown = []
    for line in read_nonempty_lines(path):
        token = first_token(line)
        if token in filename_info:
            ids.add(filename_info[token]['case_id'])
            continue
        base = Path(token).name
        if base in filename_info:
            ids.add(filename_info[base]['case_id'])
            continue
        if token.startswith('BraTS-MEN-RT-') and (not token.lower().endswith('.png')):
            ids.add(token)
            continue
        unknown.append(token)
    if unknown:
        raise RuntimeError(f'Could not map {len(unknown)} entries from {path}: {unknown[:5]}')
    return ids

def exact_wilcoxon_two_sided(a, b):
    """Small-n exact signed-rank test by sign enumeration."""
    diffs = [float(y - x) for x, y in zip(a, b)]
    nz = [d for d in diffs if abs(d) > 1e-15]
    n = len(nz)
    if n == 0:
        return {'W': 0.0, 'p_exact_two_sided': 1.0, 'n_nonzero': 0, 'diffs': diffs}
    absvals = [abs(d) for d in nz]
    order = sorted(range(n), key=lambda i: absvals[i])
    ranks = [0.0] * n
    pos = 0
    while pos < n:
        end = pos
        v = absvals[order[pos]]
        while end + 1 < n and abs(absvals[order[end + 1]] - v) < 1e-15:
            end += 1
        avg_rank = (pos + 1 + (end + 1)) / 2.0
        for k in range(pos, end + 1):
            ranks[order[k]] = avg_rank
        pos = end + 1
    wplus_obs = sum((r for r, d in zip(ranks, nz) if d > 0))
    total = sum(ranks)
    center = total / 2.0
    obs_dist = abs(wplus_obs - center)
    all_wplus = []
    for signs in itertools.product([-1, 1], repeat=n):
        wp = sum((r for r, s in zip(ranks, signs) if s > 0))
        all_wplus.append(wp)
    extreme = sum((abs(wp - center) >= obs_dist - 1e-12 for wp in all_wplus))
    p = extreme / len(all_wplus)
    W = min(wplus_obs, total - wplus_obs)
    return {'W': float(W), 'p_exact_two_sided': float(p), 'n_nonzero': int(n), 'diffs': diffs}

def prepare_final_holdout():
    print('============================================================')
    print('FINAL HOLDOUT PREPARATION / SAFETY CHECK')
    print('NO MODEL INFERENCE IS PERFORMED IN THIS STAGE')
    print('============================================================')
    required = [FINAL_CSV, CACHE_MANIFEST, DEV_HOLDOUT_LIST]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)
    actual_sha = sha256_file(FINAL_CSV)
    print('Final CSV SHA256 :', actual_sha)
    print('Expected SHA256  :', EXPECTED_FINAL_CSV_SHA256)
    if actual_sha != EXPECTED_FINAL_CSV_SHA256:
        raise RuntimeError('FINAL HOLDOUT CSV SHA256 MISMATCH')
    with open(FINAL_CSV, newline='', encoding='utf-8') as f:
        final_rows = list(csv.DictReader(f))
    if len(final_rows) != 50:
        raise RuntimeError(f'Expected 50 final cases, found {len(final_rows)}')
    final_cases = [r['case_id'] for r in final_rows]
    if len(set(final_cases)) != 50:
        raise RuntimeError('Final holdout case IDs are not unique')
    if any((r.get('split') != 'final_holdout' for r in final_rows)):
        raise RuntimeError('At least one CSV row is not marked final_holdout')
    with open(CACHE_MANIFEST, newline='', encoding='utf-8') as f:
        cache_rows = list(csv.DictReader(f))
    filename_info = {r['filename']: r for r in cache_rows}
    final_case_set = set(final_cases)
    dev_cases = list_case_ids(DEV_HOLDOUT_LIST, filename_info)
    overlap_dev = sorted(final_case_set & dev_cases)
    overlap_repeat = {}
    for r in range(10):
        for phase in ('train', 'val'):
            p = LIST_DIR / f'repeat_{r:02d}_{phase}.txt'
            if not p.exists():
                raise FileNotFoundError(p)
            cases = list_case_ids(p, filename_info)
            overlap = sorted(final_case_set & cases)
            overlap_repeat[f'repeat_{r:02d}_{phase}'] = overlap
    if overlap_dev:
        raise RuntimeError(f'Final Holdout overlaps Development Holdout: {overlap_dev}')
    bad = {k: v for k, v in overlap_repeat.items() if v}
    if bad:
        raise RuntimeError(f'Final Holdout overlaps few-shot manifests: {bad}')
    selected_rows = [r for r in cache_rows if r['case_id'] in final_case_set]
    cached_cases = {r['case_id'] for r in selected_rows}
    missing_cases = sorted(final_case_set - cached_cases)
    if missing_cases:
        raise RuntimeError(f'Final cases missing from cache_manifest: {missing_cases}')
    per_case_rows = {}
    for row in selected_rows:
        per_case_rows.setdefault(row['case_id'], []).append(row)
    completeness = {}
    for case_id in sorted(final_case_set):
        rows = per_case_rows[case_id]
        zs = sorted((int(r['z']) for r in rows))
        totals = {int(r['total_z_slices']) for r in rows}
        if len(totals) != 1:
            raise RuntimeError(f'Inconsistent total_z_slices for {case_id}: {totals}')
        total_z = next(iter(totals))
        expected_z = list(range(total_z))
        if zs != expected_z:
            raise RuntimeError(f'Incomplete/duplicate z coverage for {case_id}: got {len(zs)} rows, expected {total_z}')
        completeness[case_id] = {'slices': len(zs), 'total_z_slices': total_z}
    selected_rows.sort(key=lambda r: (r['case_id'], int(r['z'])))
    lines = [f"{r['filename']},{r['filename']}\n" for r in selected_rows]
    FINAL_LIST.write_text(''.join(lines), encoding='utf-8')
    final_list_sha = sha256_file(FINAL_LIST)
    write_sha_file(FINAL_LIST_SHA, final_list_sha)
    orientation_counts = Counter((r['mri_orientation'] for r in final_rows))
    lesion_group_counts = Counter((r['lesion_size_group'] for r in final_rows))
    report = {'status': 'PREP_PASS', 'final_holdout_csv': str(FINAL_CSV), 'final_holdout_csv_sha256': actual_sha, 'expected_final_holdout_csv_sha256': EXPECTED_FINAL_CSV_SHA256, 'final_cases': 50, 'final_case_ids': sorted(final_case_set), 'dev_holdout_overlap_count': len(overlap_dev), 'dev_holdout_overlap': overlap_dev, 'repeat_manifest_overlaps': overlap_repeat, 'orientation_counts': dict(orientation_counts), 'lesion_size_group_counts': dict(lesion_group_counts), 'final_full_volume_list': str(FINAL_LIST), 'final_full_volume_list_sha256': final_list_sha, 'final_full_volume_slices': len(selected_rows), 'per_case_slice_completeness': completeness}
    PREP_REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSAFETY CHECKS')
    print('  Final cases                :', 50)
    print('  Development overlap        :', len(overlap_dev))
    print('  Repeat00-09 train/val overlap:', sum((len(v) for v in overlap_repeat.values())))
    print('  Orientation counts         :', dict(orientation_counts))
    print('  Lesion-size counts         :', dict(lesion_group_counts))
    print('  Final full-volume slices   :', len(selected_rows))
    print('  Final list SHA256          :', final_list_sha)
    print('  Final list                 :', FINAL_LIST)
    print('  Prep report                :', PREP_REPORT)
    print('\nFINAL HOLDOUT PREPARATION: PASS')
    return (report, filename_info)

def setup_mricore():
    os.chdir(MRI_CORE)
    sys.path.insert(0, str(MRI_CORE))
    import cfg
    from utils.dataset import Public_dataset
    from models.sam import sam_model_registry
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _saved_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = cfg.parse_args()
    finally:
        sys.argv = _saved_argv
    args.arch = 'vit_b'
    args.image_size = 1024
    args.out_size = 256
    args.num_cls = 1
    args.if_update_encoder = True
    args.if_encoder_adapter = True
    args.encoder_adapter_depths = [0, 1, 10, 11]
    args.if_mask_decoder_adapter = True
    args.decoder_adapt_depth = 2
    return (args, device, Public_dataset, sam_model_registry)
resize_image = transforms.Resize((1024, 1024))
tensor_normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def safe_minmax_pil(image):
    arr = np.asarray(image, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr = np.uint8(np.clip(arr * 255.0, 0, 255))
    return Image.fromarray(arr)

def load_cross_tensor(filename):
    image = Image.open(IMAGE_DIR / filename).convert('RGB')
    image = resize_image(image)
    image = safe_minmax_pil(image)
    return tensor_normalize(image)

def parse_filename(filename):
    m = re.fullmatch('(.+)-z(\\d+)\\.png', filename)
    if m is None:
        raise RuntimeError(filename)
    return (m.group(1), int(m.group(2)))

def filename_at_z(center_filename, new_z):
    prefix, _ = parse_filename(center_filename)
    return f'{prefix}-z{new_z:03d}.png'

class FinalCrossDataset(Dataset):

    def __init__(self, list_file, filename_info):
        self.filename_info = filename_info
        self.center_names = [first_token(x) for x in read_nonempty_lines(list_file)]

    def __len__(self):
        return len(self.center_names)

    def __getitem__(self, index):
        center_name = self.center_names[index]
        info = self.filename_info[center_name]
        z = int(info['z'])
        total_z = int(info['total_z_slices'])
        prev_z = max(0, z - 1)
        next_z = min(total_z - 1, z + 1)
        prev_name = filename_at_z(center_name, prev_z)
        next_name = filename_at_z(center_name, next_z)
        return {'image_prev': load_cross_tensor(prev_name), 'image_center': load_cross_tensor(center_name), 'image_next': load_cross_tensor(next_name), 'img_name': center_name}

def dice_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    pred_sum = int(prediction.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    intersection = np.logical_and(prediction, target).sum()
    return float(2.0 * intersection / (pred_sum + target_sum))

def get_surface(mask):
    mask = mask.astype(bool)
    eroded = binary_erosion(mask, border_value=0)
    return np.logical_xor(mask, eroded)

def hd95_mm(prediction, target, spacing):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    pred_exists = bool(prediction.any())
    gt_exists = bool(target.any())
    if not pred_exists and (not gt_exists):
        return (0.0, 'both_empty')
    if not pred_exists:
        return (np.nan, 'prediction_empty')
    if not gt_exists:
        return (np.nan, 'ground_truth_empty')
    pred_surface = get_surface(prediction)
    gt_surface = get_surface(target)
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_to_gt[pred_surface], dt_to_pred[gt_surface]])
    return (float(np.percentile(distances, 95)), 'valid')

class CaseAccumulator:

    def __init__(self):
        self.case_results = []
        self.current_case = None
        self.current_gt = None
        self.current_pred = None
        self.current_spacing = None
        self.current_positive_slice_dices = []
        self.processed_slices = 0

    def _load_case(self, case_id):
        mask_path = RAW_DATA / case_id / f'{case_id}_gtv.nii.gz'
        nii = nib.load(str(mask_path))
        gt = np.asanyarray(nii.dataobj) > 0
        spacing = tuple((float(x) for x in nii.header.get_zooms()[:3]))
        pred = np.zeros(gt.shape, dtype=bool)
        return (gt, pred, spacing)

    def _finalize(self):
        if self.current_case is None:
            return
        gt = self.current_gt
        pred = self.current_pred
        full_dsc = dice_score(pred, gt)
        hd95, hd_status = hd95_mm(pred, gt, self.current_spacing)
        positive_z = np.where(np.any(gt, axis=(0, 1)))[0]
        negative_z = np.where(~np.any(gt, axis=(0, 1)))[0]
        positive_only_3d = dice_score(pred[:, :, positive_z], gt[:, :, positive_z])
        fp_slices = int(sum((np.any(pred[:, :, z]) for z in negative_z)))
        missed_slices = int(sum((not np.any(pred[:, :, z]) for z in positive_z)))
        mean_positive_2d = float(np.mean(self.current_positive_slice_dices))
        self.case_results.append({'case_id': self.current_case, 'dsc_3d_full_volume': full_dsc, 'hd95_mm': hd95, 'hd95_status': hd_status, 'dsc_3d_positive_slices_only': positive_only_3d, 'mean_dsc_2d_positive_slices': mean_positive_2d, 'gt_positive_voxels': int(gt.sum()), 'pred_positive_voxels': int(pred.sum()), 'gt_positive_slices': int(len(positive_z)), 'false_positive_negative_slices': fp_slices, 'missed_positive_slices': missed_slices, 'spacing_x_mm': self.current_spacing[0], 'spacing_y_mm': self.current_spacing[1], 'spacing_z_mm': self.current_spacing[2], 'shape_x': gt.shape[0], 'shape_y': gt.shape[1], 'shape_z': gt.shape[2]})

    def add(self, case_id, z, pred_native):
        if self.current_case is None or case_id != self.current_case:
            if self.current_case is not None:
                self._finalize()
            self.current_case = case_id
            self.current_gt, self.current_pred, self.current_spacing = self._load_case(case_id)
            self.current_positive_slice_dices = []
        gt_slice = self.current_gt[:, :, z]
        self.current_pred[:, :, z] = pred_native
        if np.any(gt_slice):
            self.current_positive_slice_dices.append(dice_score(pred_native, gt_slice))
        self.processed_slices += 1

    def finish(self):
        self._finalize()

def summarize_case_results(case_results, method, repeat, final_csv_sha, final_list_sha, total_slices, elapsed_seconds, peak_gpu_memory_gb):
    full = np.array([x['dsc_3d_full_volume'] for x in case_results], dtype=float)
    pos3d = np.array([x['dsc_3d_positive_slices_only'] for x in case_results], dtype=float)
    pos2d = np.array([x['mean_dsc_2d_positive_slices'] for x in case_results], dtype=float)
    hd = np.array([x['hd95_mm'] for x in case_results], dtype=float)
    valid_hd = hd[np.isfinite(hd)]
    return {'evaluation_split': 'untouched_final_holdout_seed2026', 'final_holdout_csv_sha256': final_csv_sha, 'final_holdout_full_volume_list_sha256': final_list_sha, 'method': method, 'repeat': repeat, 'test_cases': len(case_results), 'test_slices': total_slices, 'threshold': 'logit > 0', 'context': 'z only' if method == 'single' else 'z-1,z,z+1', 'fusion': None if method == 'single' else 'arithmetic mean of embeddings', 'boundary_policy': None if method == 'single' else 'edge replication', 'mean_full_volume_dsc': float(np.mean(full)), 'std_full_volume_dsc_across_cases': float(np.std(full, ddof=1)), 'median_full_volume_dsc': float(np.median(full)), 'mean_positive_slice_only_3d_dsc': float(np.mean(pos3d)), 'std_positive_slice_only_3d_dsc_across_cases': float(np.std(pos3d, ddof=1)), 'mean_casewise_positive_2d_dsc': float(np.mean(pos2d)), 'std_casewise_positive_2d_dsc_across_cases': float(np.std(pos2d, ddof=1)), 'hd95_valid_cases': int(len(valid_hd)), 'hd95_invalid_cases': int(len(hd) - len(valid_hd)), 'mean_hd95_mm_valid_cases': float(np.mean(valid_hd)) if len(valid_hd) else None, 'std_hd95_mm_valid_cases_across_cases': float(np.std(valid_hd, ddof=1)) if len(valid_hd) > 1 else None, 'prediction_empty_cases': [x['case_id'] for x in case_results if x['hd95_status'] == 'prediction_empty'], 'total_false_positive_negative_slices': int(sum((x['false_positive_negative_slices'] for x in case_results))), 'total_missed_positive_slices': int(sum((x['missed_positive_slices'] for x in case_results))), 'elapsed_seconds': float(elapsed_seconds), 'peak_gpu_memory_gb': peak_gpu_memory_gb}

def save_job(out_dir, case_results, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = out_dir / 'case_metrics.csv'
    summary_json = out_dir / 'summary.json'
    with open(case_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=case_results[0].keys())
        writer.writeheader()
        writer.writerows(case_results)
    summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

def valid_existing_summary(summary_path, method, repeat, final_csv_sha, final_list_sha):
    if not summary_path.exists():
        return False
    try:
        s = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        return False
    return s.get('evaluation_split') == 'untouched_final_holdout_seed2026' and s.get('final_holdout_csv_sha256') == final_csv_sha and (s.get('final_holdout_full_volume_list_sha256') == final_list_sha) and (s.get('method') == method) and (int(s.get('repeat', -1)) == repeat) and (int(s.get('test_cases', -1)) == 50)

def evaluate_all(prep_report, filename_info):
    final_csv_sha = prep_report['final_holdout_csv_sha256']
    final_list_sha = prep_report['final_full_volume_list_sha256']
    total_slices = prep_report['final_full_volume_slices']
    args, device, Public_dataset, sam_model_registry = setup_mricore()
    print('\n============================================================')
    print('FINAL HOLDOUT ONE-TIME EVALUATION')
    print('============================================================')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    for r in REPEATS:
        for p in (SINGLE_CKPT[r], CROSS_CKPT[r]):
            if not p.exists():
                raise FileNotFoundError(p)
    single_dataset = Public_dataset(args=args, img_folder=str(IMAGE_DIR), mask_folder=str(MASK_DIR), img_list=str(FINAL_LIST), phase='test', targets=['combine_all'], normalize_type='ours', if_prompt=False, crop_size=1024, delete_empty_masks=False, target_cls=-1, if_spatial=False, few_shot=False)
    if len(single_dataset) != total_slices:
        raise RuntimeError(f'Single dataset count mismatch: {len(single_dataset)} vs {total_slices}')
    single_loader = DataLoader(single_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    cross_dataset = FinalCrossDataset(FINAL_LIST, filename_info)
    if len(cross_dataset) != total_slices:
        raise RuntimeError(f'Cross dataset count mismatch: {len(cross_dataset)} vs {total_slices}')
    cross_loader = DataLoader(cross_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    print('Final Holdout cases :', 50)
    print('Final Holdout slices:', total_slices)
    print('Threshold           : logit > 0')
    print('Cross fusion        : mean(z-1,z,z+1)')
    print('Boundary policy     : edge replication')
    print('\nBuilding MRI-CORE ViT-B...')
    model = sam_model_registry['vit_b'](args, checkpoint=str(MRI_CORE / 'pretrained_weights/mri_foundation.pth'), num_classes=1, image_size=1024, pretrained_sam=True)
    model = model.to(device)
    model.eval()

    def load_ckpt(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

    def decode(logits, native_h, native_w):
        binary_256 = (logits > 0).float()
        pred_native = F.interpolate(binary_256, size=(native_h, native_w), mode='nearest')
        return pred_native[:, 0].detach().cpu().numpy() > 0.5

    def eval_single(repeat):
        method = 'single'
        out_dir = OUTPUT_ROOT / f'repeat_{repeat:02d}' / method
        summary_path = out_dir / 'summary.json'
        if valid_existing_summary(summary_path, method, repeat, final_csv_sha, final_list_sha):
            print(f'[SKIP VALID] Repeat{repeat:02d} SINGLE: {summary_path}')
            return json.loads(summary_path.read_text(encoding='utf-8'))
        print(f'\n--- Repeat{repeat:02d} SINGLE / FINAL HOLDOUT ---')
        load_ckpt(SINGLE_CKPT[repeat])
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        acc = CaseAccumulator()
        t0 = time.time()
        with torch.inference_mode():
            for batch in single_loader:
                images = batch['image'].to(device, non_blocking=True)
                embedding = model.image_encoder(images)
                sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
                logits, _ = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
                names = list(batch['img_name'])
                for i, filename in enumerate(names):
                    info = filename_info[filename]
                    case_id = info['case_id']
                    z = int(info['z'])
                    if acc.current_case is None or case_id != acc.current_case:
                        gt_nii = nib.load(str(RAW_DATA / case_id / f'{case_id}_gtv.nii.gz'))
                        shape = gt_nii.shape
                        native_h, native_w = (shape[0], shape[1])
                    else:
                        native_h = acc.current_gt.shape[0]
                        native_w = acc.current_gt.shape[1]
                    pred = decode(logits[i:i + 1], native_h, native_w)[0]
                    acc.add(case_id, z, pred)
                if acc.processed_slices % 500 < len(names):
                    print(f'  processed {acc.processed_slices}/{total_slices} slices')
        acc.finish()
        elapsed = time.time() - t0
        peak = float(torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == 'cuda' else None
        if acc.processed_slices != total_slices:
            raise RuntimeError(f'Processed {acc.processed_slices}, expected {total_slices}')
        if len(acc.case_results) != 50:
            raise RuntimeError(f'Got {len(acc.case_results)} cases, expected 50')
        summary = summarize_case_results(acc.case_results, method, repeat, final_csv_sha, final_list_sha, total_slices, elapsed, peak)
        save_job(out_dir, acc.case_results, summary)
        print(f"  Full={summary['mean_full_volume_dsc']:.6f} | Pos3D={summary['mean_positive_slice_only_3d_dsc']:.6f} | Pos2D={summary['mean_casewise_positive_2d_dsc']:.6f} | HD95={summary['mean_hd95_mm_valid_cases']:.3f} | FP={summary['total_false_positive_negative_slices']} | Miss={summary['total_missed_positive_slices']}")
        return summary

    def eval_cross(repeat):
        method = 'cross'
        out_dir = OUTPUT_ROOT / f'repeat_{repeat:02d}' / method
        summary_path = out_dir / 'summary.json'
        if valid_existing_summary(summary_path, method, repeat, final_csv_sha, final_list_sha):
            print(f'[SKIP VALID] Repeat{repeat:02d} CROSS: {summary_path}')
            return json.loads(summary_path.read_text(encoding='utf-8'))
        print(f'\n--- Repeat{repeat:02d} CROSS / FINAL HOLDOUT ---')
        load_ckpt(CROSS_CKPT[repeat])
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        acc = CaseAccumulator()
        t0 = time.time()
        with torch.inference_mode():
            for batch in cross_loader:
                prev_images = batch['image_prev'].to(device, non_blocking=True)
                center_images = batch['image_center'].to(device, non_blocking=True)
                next_images = batch['image_next'].to(device, non_blocking=True)
                emb_prev = model.image_encoder(prev_images)
                emb_center = model.image_encoder(center_images)
                emb_next = model.image_encoder(next_images)
                fused = (emb_prev + emb_center + emb_next) / 3.0
                sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
                logits, _ = model.mask_decoder(image_embeddings=fused, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
                names = list(batch['img_name'])
                for i, filename in enumerate(names):
                    info = filename_info[filename]
                    case_id = info['case_id']
                    z = int(info['z'])
                    if acc.current_case is None or case_id != acc.current_case:
                        gt_nii = nib.load(str(RAW_DATA / case_id / f'{case_id}_gtv.nii.gz'))
                        shape = gt_nii.shape
                        native_h, native_w = (shape[0], shape[1])
                    else:
                        native_h = acc.current_gt.shape[0]
                        native_w = acc.current_gt.shape[1]
                    pred = decode(logits[i:i + 1], native_h, native_w)[0]
                    acc.add(case_id, z, pred)
                if acc.processed_slices % 500 < len(names):
                    print(f'  processed {acc.processed_slices}/{total_slices} slices')
        acc.finish()
        elapsed = time.time() - t0
        peak = float(torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == 'cuda' else None
        if acc.processed_slices != total_slices:
            raise RuntimeError(f'Processed {acc.processed_slices}, expected {total_slices}')
        if len(acc.case_results) != 50:
            raise RuntimeError(f'Got {len(acc.case_results)} cases, expected 50')
        summary = summarize_case_results(acc.case_results, method, repeat, final_csv_sha, final_list_sha, total_slices, elapsed, peak)
        save_job(out_dir, acc.case_results, summary)
        print(f"  Full={summary['mean_full_volume_dsc']:.6f} | Pos3D={summary['mean_positive_slice_only_3d_dsc']:.6f} | Pos2D={summary['mean_casewise_positive_2d_dsc']:.6f} | HD95={summary['mean_hd95_mm_valid_cases']:.3f} | FP={summary['total_false_positive_negative_slices']} | Miss={summary['total_missed_positive_slices']}")
        return summary
    summaries = {'single': {}, 'cross': {}}
    for r in REPEATS:
        summaries['single'][r] = eval_single(r)
        summaries['cross'][r] = eval_cross(r)
    metric_map = {'Full DSC': 'mean_full_volume_dsc', 'Positive 3D DSC': 'mean_positive_slice_only_3d_dsc', 'Positive 2D DSC': 'mean_casewise_positive_2d_dsc', 'HD95 (mm)': 'mean_hd95_mm_valid_cases', 'FP negative slices': 'total_false_positive_negative_slices', 'Missed positive slices': 'total_missed_positive_slices'}
    aggregate = {'evaluation_split': 'untouched_final_holdout_seed2026', 'final_holdout_csv_sha256': final_csv_sha, 'final_holdout_full_volume_list_sha256': final_list_sha, 'repeats': REPEATS, 'metrics': {}}
    aggregate_rows = []
    for label, key in metric_map.items():
        av = [float(summaries['single'][r][key]) for r in REPEATS]
        bv = [float(summaries['cross'][r][key]) for r in REPEATS]
        dv = [b - a for a, b in zip(av, bv)]
        row = {'metric': label, 'single_mean': statistics.mean(av), 'single_sd_across_repeats': statistics.stdev(av), 'cross_mean': statistics.mean(bv), 'cross_sd_across_repeats': statistics.stdev(bv), 'paired_delta_mean': statistics.mean(dv), 'paired_delta_sd': statistics.stdev(dv), 'cross_better_repeats': sum((d > 0 for d in dv))}
        if label in ('HD95 (mm)', 'FP negative slices', 'Missed positive slices'):
            row['direction_aware_cross_better_repeats'] = sum((d < 0 for d in dv))
        else:
            row['direction_aware_cross_better_repeats'] = sum((d > 0 for d in dv))
        aggregate_rows.append(row)
        aggregate['metrics'][label] = row
    for label, key in (('Positive 3D DSC', 'mean_positive_slice_only_3d_dsc'), ('Positive 2D DSC', 'mean_casewise_positive_2d_dsc')):
        av = [float(summaries['single'][r][key]) for r in REPEATS]
        bv = [float(summaries['cross'][r][key]) for r in REPEATS]
        aggregate['metrics'][label]['wilcoxon_exact'] = exact_wilcoxon_two_sided(av, bv)
    per_repeat_rows = []
    for r in REPEATS:
        a = summaries['single'][r]
        b = summaries['cross'][r]
        per_repeat_rows.append({'repeat': f'Repeat{r:02d}', 'single_full_dsc': a['mean_full_volume_dsc'], 'cross_full_dsc': b['mean_full_volume_dsc'], 'delta_full_dsc': b['mean_full_volume_dsc'] - a['mean_full_volume_dsc'], 'single_positive3d_dsc': a['mean_positive_slice_only_3d_dsc'], 'cross_positive3d_dsc': b['mean_positive_slice_only_3d_dsc'], 'delta_positive3d_dsc': b['mean_positive_slice_only_3d_dsc'] - a['mean_positive_slice_only_3d_dsc'], 'single_positive2d_dsc': a['mean_casewise_positive_2d_dsc'], 'cross_positive2d_dsc': b['mean_casewise_positive_2d_dsc'], 'delta_positive2d_dsc': b['mean_casewise_positive_2d_dsc'] - a['mean_casewise_positive_2d_dsc'], 'single_hd95_mm': a['mean_hd95_mm_valid_cases'], 'cross_hd95_mm': b['mean_hd95_mm_valid_cases'], 'delta_hd95_mm': b['mean_hd95_mm_valid_cases'] - a['mean_hd95_mm_valid_cases'], 'single_fp_negative_slices': a['total_false_positive_negative_slices'], 'cross_fp_negative_slices': b['total_false_positive_negative_slices'], 'delta_fp_negative_slices': b['total_false_positive_negative_slices'] - a['total_false_positive_negative_slices'], 'single_missed_positive_slices': a['total_missed_positive_slices'], 'cross_missed_positive_slices': b['total_missed_positive_slices'], 'delta_missed_positive_slices': b['total_missed_positive_slices'] - a['total_missed_positive_slices']})
    agg_json = LOCK_DIR / 'aggregate_final_holdout_repeat00_04.json'
    agg_csv = LOCK_DIR / 'aggregate_final_holdout_repeat00_04.csv'
    per_csv = LOCK_DIR / 'per_repeat_final_holdout_repeat00_04.csv'
    agg_json.write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    with open(agg_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys(), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with open(per_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=per_repeat_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_repeat_rows)
    print('\n============================================================')
    print('FINAL HOLDOUT AGGREGATE — REPEAT00-04')
    print('============================================================')
    for row in aggregate_rows:
        print(f"{row['metric']:22s} | Single {row['single_mean']:.6f} ± {row['single_sd_across_repeats']:.6f} | Cross {row['cross_mean']:.6f} ± {row['cross_sd_across_repeats']:.6f} | Δ {row['paired_delta_mean']:+.6f} ± {row['paired_delta_sd']:.6f}")
    for label in ('Positive 3D DSC', 'Positive 2D DSC'):
        w = aggregate['metrics'][label]['wilcoxon_exact']
        print(f"{label} | W={w['W']} | exact two-sided p={w['p_exact_two_sided']}")
    print('\nOutputs:')
    print(' Aggregate JSON :', agg_json)
    print(' Aggregate CSV  :', agg_csv)
    print(' Per-repeat CSV :', per_csv)
    print('\nFINAL HOLDOUT REPEAT00-04 EVALUATION: PASS')

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-only', action='store_true', help='Verify locks/overlaps and create final full-volume list only.')
    group.add_argument('--evaluate', action='store_true', help='Run the one-time official final-holdout evaluation.')
    cli = parser.parse_args()
    prep_report, filename_info = prepare_final_holdout()
    if cli.prepare_only:
        return
    if cli.evaluate:
        evaluate_all(prep_report, filename_info)
if __name__ == '__main__':
    main()
