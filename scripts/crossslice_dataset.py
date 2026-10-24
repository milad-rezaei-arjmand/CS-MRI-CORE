# Portable release copy; see docs/PORTABILITY.md.
from release_paths import release_workspace, release_upstream
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
PROJECT = release_workspace()
DATA_ROOT = PROJECT / 'datasets/BraTS-MEN-RT/processed_mricore'
IMAGE_DIR = DATA_ROOT / 'images'
MASK_DIR = DATA_ROOT / 'masks'

def neighbor_filename(center_name, offset):
    match = re.fullmatch('(.+)-z(\\d+)\\.png', center_name)
    if match is None:
        raise RuntimeError(f'Unexpected filename: {center_name}')
    prefix = match.group(1)
    z = int(match.group(2))
    return f'{prefix}-z{z + offset:03d}.png'

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

class CrossSliceDataset(Dataset):

    def __init__(self, list_file, phase='train'):
        self.phase = phase
        with open(list_file, encoding='utf-8') as f:
            self.lines = [x.strip() for x in f if x.strip()]
        self.resize_image = transforms.Resize((1024, 1024))
        self.resize_mask = transforms.Resize((1024, 1024), interpolation=InterpolationMode.NEAREST)
        self.photo = transforms.Compose([transforms.RandomEqualize(p=0.1), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)])
        self.to_tensor_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.spatial = transforms.Compose([transforms.RandomResizedCrop(1024, scale=(0.5, 1.5), interpolation=InterpolationMode.NEAREST), transforms.RandomRotation(45, interpolation=InterpolationMode.NEAREST)])

    def __len__(self):
        return len(self.lines)

    def apply_same_photometric(self, images):
        initial_state = torch.get_rng_state()
        outputs = []
        final_state = None
        for index, image in enumerate(images):
            torch.set_rng_state(initial_state)
            transformed = self.photo(image)
            if index == 0:
                final_state = torch.get_rng_state()
            outputs.append(transformed)
        torch.set_rng_state(final_state)
        return outputs

    def __getitem__(self, index):
        line = self.lines[index]
        center_name, mask_name = line.split(',')
        prev_name = neighbor_filename(center_name, -1)
        next_name = neighbor_filename(center_name, +1)
        names = [prev_name, center_name, next_name]
        pil_images = []
        for name in names:
            path = IMAGE_DIR / name
            if not path.exists():
                raise FileNotFoundError(path)
            image = Image.open(path).convert('RGB')
            image = self.resize_image(image)
            image = safe_minmax_pil(image)
            pil_images.append(image)
        mask = Image.open(MASK_DIR / mask_name).convert('L')
        mask = self.resize_mask(mask)
        if self.phase == 'train':
            pil_images = self.apply_same_photometric(pil_images)
        images = [self.to_tensor_norm(image) for image in pil_images]
        mask_np = (np.asarray(mask, dtype=np.int64) > 0).astype(np.int64)
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)
        if self.phase == 'train':
            mask_3ch = mask_tensor.unsqueeze(0).repeat(3, 1, 1).float()
            stack = torch.stack([images[0], images[1], images[2], mask_3ch], dim=0)
            stack = self.spatial(stack)
            images = [stack[0], stack[1], stack[2]]
            mask_tensor = stack[3, 0].long()
        return {'image_prev': images[0], 'image_center': images[1], 'image_next': images[2], 'mask': mask_tensor.unsqueeze(0), 'img_name': center_name, 'prev_name': prev_name, 'next_name': next_name}
