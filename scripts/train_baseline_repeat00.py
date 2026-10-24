# Portable release copy; see docs/PORTABILITY.md.
from release_paths import release_workspace, release_upstream
import os
RELEASE_REPEAT = int(os.environ.get("CS_MRI_CORE_REPEAT", "0"))
assert RELEASE_REPEAT in range(5)
from pathlib import Path
import os
import sys
import csv
import json
import math
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision
import monai
from torch.utils.data import DataLoader
PROJECT = release_workspace()
MRI_CORE = release_upstream()
DATA_ROOT = PROJECT / 'datasets/BraTS-MEN-RT/processed_mricore'
IMAGE_DIR = DATA_ROOT / 'images'
MASK_DIR = DATA_ROOT / 'masks'
LIST_DIR = DATA_ROOT / 'lists'
TRAIN_LIST = LIST_DIR / f'repeat_{RELEASE_REPEAT:02d}_train.txt'
VAL_LIST = LIST_DIR / f'repeat_{RELEASE_REPEAT:02d}_val.txt'
MRI_CORE_CKPT = MRI_CORE / 'pretrained_weights/mri_foundation.pth'
RUN_DIR = PROJECT / f'experiments/baseline_mricore/repeat_{RELEASE_REPEAT:02d}'
RUN_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = RUN_DIR / 'best_model.pth'
LOG_FILE = RUN_DIR / 'training_log.csv'
CONFIG_FILE = RUN_DIR / 'config.json'
os.chdir(MRI_CORE)
sys.path.insert(0, str(MRI_CORE))
import cfg
from utils.dataset import Public_dataset
from utils.metrics import compute_dice
from models.sam import sam_model_registry
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
args = cfg.parse_args()
args.arch = 'vit_b'
args.image_size = 1024
args.out_size = 256
args.num_cls = 1
args.lr = 0.0001
args.if_warmup = True
args.warmup_period = 200
args.if_update_encoder = True
args.if_encoder_adapter = True
args.encoder_adapter_depths = [0, 1, 10, 11]
args.if_mask_decoder_adapter = True
args.decoder_adapt_depth = 2
EPOCHS = 3000
VAL_FREQ = 10
MICRO_BATCH = 1
EFFECTIVE_BATCH = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = Public_dataset(args=args, img_folder=str(IMAGE_DIR), mask_folder=str(MASK_DIR), img_list=str(TRAIN_LIST), phase='train', targets=['combine_all'], normalize_type='ours', if_prompt=False, crop_size=1024, delete_empty_masks=True, target_cls=-1, if_spatial=True, few_shot=False)
val_dataset = Public_dataset(args=args, img_folder=str(IMAGE_DIR), mask_folder=str(MASK_DIR), img_list=str(VAL_LIST), phase='val', targets=['combine_all'], normalize_type='ours', if_prompt=False, crop_size=1024, delete_empty_masks=True, target_cls=-1, if_spatial=False, few_shot=False)
generator = torch.Generator()
generator.manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH, shuffle=True, num_workers=0, generator=generator)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
print('=== MRI-CORE BASELINE REPEAT 00 ===')
print('Device:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
print('Train slices:', len(train_dataset))
print('Val slices:', len(val_dataset))
print('Epoch limit:', EPOCHS)
print('Micro batch:', MICRO_BATCH)
print('Effective batch:', EFFECTIVE_BATCH)
print('\nLoading MRI-CORE...')
model = sam_model_registry['vit_b'](args, checkpoint=str(MRI_CORE_CKPT), num_classes=1, image_size=1024, pretrained_sam=True)
model = model.to(DEVICE)
for name, parameter in model.named_parameters():
    if 'Adapter' not in name:
        parameter.requires_grad = False
trainable_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))
total_parameters = sum((p.numel() for p in model.parameters()))
print('\nTrainable parameters:')
print(f'{trainable_parameters:,}')
print('Trainable percentage:', f'{100 * trainable_parameters / total_parameters:.4f}%')
initial_lr = args.lr / args.warmup_period
optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
criterion = monai.losses.DiceCELoss(sigmoid=True)
resize_mask = torchvision.transforms.Resize((args.out_size, args.out_size))
updates_per_epoch = math.ceil(len(train_dataset) / EFFECTIVE_BATCH)
max_iterations = EPOCHS * updates_per_epoch
print('Optimizer updates/epoch:', updates_per_epoch)
print('Maximum optimizer updates:', max_iterations)

def adjust_lr(iter_num):
    if args.if_warmup and iter_num < args.warmup_period:
        lr = args.lr * (iter_num + 1) / args.warmup_period
    else:
        shift_iter = iter_num - args.warmup_period
        lr = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr

def forward(images, masks):
    images = images.to(DEVICE, non_blocking=True)
    masks = resize_mask(masks).to(DEVICE, non_blocking=True)
    embedding = model.image_encoder(images)
    sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=None, masks=None)
    prediction, _ = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, multimask_output=True)
    loss = criterion(prediction, masks)
    return (prediction, masks, loss)

@torch.no_grad()
def validate():
    model.eval()
    losses = []
    dices = []
    for batch in val_loader:
        prediction, masks, loss = forward(batch['image'], batch['mask'])
        binary_prediction = (prediction > 0).to(torch.long).cpu()
        binary_mask = masks.to(torch.long).cpu()
        dice = compute_dice(binary_prediction, binary_mask)
        losses.append(float(loss.item()))
        dices.append(float(dice))
    return (float(np.mean(losses)), float(np.mean(dices)))
config = {'model': 'MRI-CORE ViT-B', 'repeat': RELEASE_REPEAT, 'seed': SEED, 'train_slices': len(train_dataset), 'val_slices': len(val_dataset), 'epochs': EPOCHS, 'val_frequency': VAL_FREQ, 'micro_batch': MICRO_BATCH, 'effective_batch': EFFECTIVE_BATCH, 'gradient_accumulation': True, 'learning_rate': args.lr, 'warmup_iterations': args.warmup_period, 'adapter_depths': args.encoder_adapter_depths, 'decoder_adapter_depth': args.decoder_adapt_depth, 'image_size': args.image_size, 'mask_output_size': args.out_size, 'checkpoint': str(MRI_CORE_CKPT)}
with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)
best_val_dice = -1.0
best_epoch = -1
last_update_epoch = 0
optimizer_iteration = 0
logs = []
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
print('\n=== TRAINING START ===')
stop_training = False
for epoch in range(EPOCHS):
    model.train()
    sample_losses = []
    loader_iterator = iter(train_loader)
    micro_index = 0
    total_micro_batches = len(train_loader)
    while micro_index < total_micro_batches:
        remaining = total_micro_batches - micro_index
        group_size = min(EFFECTIVE_BATCH, remaining)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(group_size):
            batch = next(loader_iterator)
            prediction, masks, loss = forward(batch['image'], batch['mask'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss detected.')
            sample_losses.append(float(loss.item()))
            (loss / group_size).backward()
            micro_index += 1
        optimizer.step()
        lr = adjust_lr(optimizer_iteration)
        optimizer_iteration += 1
    train_loss = float(np.mean(sample_losses))
    val_loss = ''
    val_dice = ''
    if epoch % VAL_FREQ == 0:
        val_loss, val_dice = validate()
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            last_update_epoch = epoch
            torch.save(model.state_dict(), BEST_CKPT)
            status = 'NEW BEST'
        else:
            status = ''
        print(f'Epoch {epoch:04d} | train {train_loss:.4f} | val loss {val_loss:.4f} | val Dice {val_dice:.4f} | best {best_val_dice:.4f} @ {best_epoch} | lr {lr:.8f} {status}')
        if epoch >= 1000 and epoch - last_update_epoch >= 200:
            print('\nEARLY STOPPING')
            stop_training = True
    elif epoch % 50 == 0:
        print(f'Epoch {epoch:04d} | train {train_loss:.4f} | lr {lr:.8f}')
    logs.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_dice': val_dice, 'best_val_dice': best_val_dice, 'best_epoch': best_epoch, 'learning_rate': lr, 'optimizer_iteration': optimizer_iteration})
    if stop_training:
        break
with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=logs[0].keys())
    writer.writeheader()
    writer.writerows(logs)
assert BEST_CKPT.exists()
print('\n=== BASELINE REPEAT 00 COMPLETE ===')
print('Best epoch:', best_epoch)
print('Best validation Dice:', best_val_dice)
print('Optimizer iterations:', optimizer_iteration)
if DEVICE.type == 'cuda':
    print('Peak GPU memory:', round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3), 'GB')
print('\nBest checkpoint:')
print(BEST_CKPT)
print('\nTraining log:')
print(LOG_FILE)
print('\nMRI-CORE BASELINE REPEAT 00: PASS')
