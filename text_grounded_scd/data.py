import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json

class ToTensor(object):
    def __call__(self, data):
        t1, t2, label = data['t1'], data['t2'], data['label']
        out ={'t1': F.to_tensor(t1),'t2': F.to_tensor(t2), 'label': F.to_tensor(label)}
        if 'sem_t1' in data and 'sem_t2' in data:
            out['sem_t1'] = torch.from_numpy(np.array(data['sem_t1'])).long()
            out['sem_t2'] = torch.from_numpy(np.array(data['sem_t2'])).long()
        return out
    
class Color(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8):
        self.p = p
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data):
        if random.random() < self.p:
            fn_idx, b_factor, c_factor, s_factor, h_factor = \
                self.jitter.get_params(self.jitter.brightness, self.jitter.contrast, 
                                       self.jitter.saturation, self.jitter.hue)

            def apply_jitter(img):
                for fn_id in fn_idx:
                    if fn_id == 0 and b_factor is not None: img = F.adjust_brightness(img, b_factor)
                    elif fn_id == 1 and c_factor is not None: img = F.adjust_contrast(img, c_factor)
                    elif fn_id == 2 and s_factor is not None: img = F.adjust_saturation(img, s_factor)
                    elif fn_id == 3 and h_factor is not None: img = F.adjust_hue(img, h_factor)
                return img

            data['t1'] = apply_jitter(data['t1'])
            data['t2'] = apply_jitter(data['t2'])
        return data


class RandomRotate(object):
    def __call__(self, data):
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            data['t1'] = F.rotate(data['t1'], angle)
            data['t2'] = F.rotate(data['t2'], angle)
            data['label'] = F.rotate(data['label'], angle)
            if 'sem_t1' in data:
                data['sem_t1'] = F.rotate(data['sem_t1'], angle)
                data['sem_t2'] = F.rotate(data['sem_t2'], angle)
        return data

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, data):
        t1, t2, label = data['t1'], data['t2'], data['label']
        w, h = t1.size
        if w > self.size and h > self.size:
            i = random.randint(0, h - self.size)
            j = random.randint(0, w - self.size)
            data['t1'] = F.crop(t1, i, j, self.size, self.size)
            data['t2'] = F.crop(t2, i, j, self.size, self.size)
            data['label'] = F.crop(label, i, j, self.size, self.size)
            if 'sem_t1' in data:
                data['sem_t1'] = F.crop(data['sem_t1'], i, j, self.size, self.size)
                data['sem_t2'] = F.crop(data['sem_t2'], i, j, self.size, self.size)
        return data

class RandomFlip(object):
    def __call__(self, data):
        if random.random() > 0.5:
            data['t1'] = F.hflip(data['t1'])
            data['t2'] = F.hflip(data['t2'])
            data['label'] = F.hflip(data['label'])
            if 'sem_t1' in data:
                data['sem_t1'] = F.hflip(data['sem_t1'])
                data['sem_t2'] = F.hflip(data['sem_t2'])
        if random.random() > 0.5:
            data['t1'] = F.vflip(data['t1'])
            data['t2'] = F.vflip(data['t2'])
            data['label'] = F.vflip(data['label'])
            if 'sem_t1' in data:
                data['sem_t1'] = F.vflip(data['sem_t1'])
                data['sem_t2'] = F.vflip(data['sem_t2'])
        return data
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std = std 
    def __call__(self, data):
        data['t1'] = F.normalize(data['t1'], self.mean, self.std)
        data['t2'] = F.normalize(data['t2'], self.mean, self.std)
        return data

class SemanticChangeDataset(Dataset):
    def __init__(self, image_t1, image_t2, gt_root, size, mode, sem_t1_root=None, sem_t2_root=None, mean=None, std=None): 
        exts = ('.jpg', '.png', '.tif', '.tiff')
        self.images_t1 = sorted([os.path.join(image_t1, f) for f in os.listdir(image_t1) if f.lower().endswith(exts)])
        self.images_t2 = sorted([os.path.join(image_t2, f) for f in os.listdir(image_t2) if f.lower().endswith(exts)])
        self.sem_t1_paths = []
        self.sem_t2_paths = []
        self.derive_binary = False
        self.img_size = size
        self.has_gt = False
        self.gts = []
        if gt_root and os.path.exists(gt_root):
            valid_files = [f for f in os.listdir(gt_root) if f.lower().endswith(exts)]
            if len(valid_files) > 0:
                self.gts = sorted([os.path.join(gt_root, f) for f in valid_files])
                self.has_gt = True
        if sem_t1_root and os.path.exists(sem_t1_root):
            self.sem_t1_paths = sorted([os.path.join(sem_t1_root, f) for f in os.listdir(sem_t1_root) if f.lower().endswith(exts)])
        if sem_t2_root and os.path.exists(sem_t2_root):
            self.sem_t2_paths = sorted([os.path.join(sem_t2_root, f) for f in os.listdir(sem_t2_root) if f.lower().endswith(exts)])
        if not self.has_gt and len(self.sem_t1_paths) > 0 and len(self.sem_t2_paths) > 0:
            self.derive_binary = True
        if mode == 'train':
            self.transform = transforms.Compose([
                RandomCrop(size), 
                RandomFlip(),
                RandomRotate(),            
                Color(p=0.8), 
                ToTensor(),
                Normalize(mean, std) 
            ])
        else:
            self.transform = transforms.Compose([
                ToTensor(),
                Normalize(mean, std) 
            ])

    def __getitem__(self, idx):
        t1 = Image.open(self.images_t1[idx]).convert('RGB')
        t2 = Image.open(self.images_t2[idx]).convert('RGB')
        sem_t1 = None
        sem_t2 = None
        if self.sem_t1_paths and idx < len(self.sem_t1_paths):
            sem_t1 = Image.open(self.sem_t1_paths[idx]).convert('L')
        if self.sem_t2_paths and idx < len(self.sem_t2_paths):
            sem_t2 = Image.open(self.sem_t2_paths[idx]).convert('L')
        label = None
        if self.has_gt:
            try:
                label = Image.open(self.gts[idx]).convert('L').point(lambda x: 255 if x > 128 else 0)
            except IndexError:
                pass 
        if label is None and self.derive_binary and sem_t1 is not None and sem_t2 is not None:
            s1_arr = np.array(sem_t1)
            s2_arr = np.array(sem_t2)
            diff = (s1_arr != s2_arr).astype(np.uint8) * 255
            label = Image.fromarray(diff).convert('L')
        if label is None:
            label = Image.new('L', t1.size, 0)
        data = {
            't1': t1,
            't2': t2,
            'label': label
        }
        if sem_t1 is not None: data['sem_t1'] = sem_t1
        if sem_t2 is not None: data['sem_t2'] = sem_t2
        return self.transform(data)
    def __len__(self):
        return len(self.images_t1)

def get_loaders_from_config(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    mean = cfg.get('mean')
    std = cfg.get('std')
    paths = cfg['paths']


    train_dataset = SemanticChangeDataset(
        paths['train_t1'], 
        paths['train_t2'], 
        paths.get('train_mask'), 
        cfg['img_size'], 
        'train', 
        paths.get('train_sem_t1'), 
        paths.get('train_sem_t2'),
        mean=mean, 
        std=std   
    )
    
    val_dataset = SemanticChangeDataset(
        paths['val_t1'], 
        paths['val_t2'], 
        paths.get('val_mask'), 
        cfg['img_size'], 
        'val', 
        paths.get('val_sem_t1'), 
        paths.get('val_sem_t2'),
        mean=mean, 
        std=std    
    )
    test_dataset = SemanticChangeDataset(
        paths['test_t1'], 
        paths['test_t2'], 
        paths.get('test_mask'), 
        cfg['img_size'], 
        'test', 
        paths.get('test_sem_t1'), 
        paths.get('test_sem_t2'),
        mean=mean, 
        std=std    
    )
    
    return (DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4),
            DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4))
