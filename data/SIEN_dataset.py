import ipdb
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(input, target, patch_size, scale=1, ix=-1, iy=-1):
    ih, iw, channels = input.shape
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    # (tx, ty) = (scale * ix, scale * iy)

    input = input[ix:ix + ip, iy:iy + ip, :]  # [:, ty:ty + tp, tx:tx + tp]
    target = target[ix:ix + ip, iy:iy + ip, :]  # [:, iy:iy + ip, ix:ix + ip]

    return input, target


def augment(inputs, target, hflip, rot):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot180 = rot and random.random() < 0.5

    def _augment(inputs, target):
        if hflip:
            inputs = inputs[:, ::-1, :]
            target = target[:, ::-1, :]
        if vflip:
            inputs = inputs[::-1, :, :]
            target = target[::-1, :, :]
        if rot180:
            inputs = cv2.rotate(inputs, cv2.ROTATE_180)
            target = cv2.rotate(target, cv2.ROTATE_180)
        return inputs, target

    inputs, target = _augment(inputs, target)

    return inputs, target


def get_image_ldr(img):
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    # if img.shape[1]*img.shape[2] >= 800*800:
    #     img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    w, h = img.shape[0], img.shape[1]
    while w % 4 != 0:
        w += 1
    while h % 4 != 0:
        h += 1
    img = cv2.resize(img, (h, w))
    return img


def load_image_train2(group):
    # images = [get_image(img) for img in group]
    # inputs = images[:-1]
    # target = images[-1]
    inputs = get_image_ldr(group[0])
    target = get_image_ldr(group[1])
    # if black_edges_crop == True:
    #     inputs = [indiInput[70:470, :, :] for indiInput in inputs]
    #     target = target[280:1880, :, :]
    #     return inputs, target
    # else:
    return inputs, target


def load_image_numpy(group, patch_size):
    dataset_name = group[0].split('/')[5]  # which means using over/under exposures
    path = group[0].replace(dataset_name, f'{dataset_name}_np')
    path = path.replace('JPG', 'npy')
    input_np = np.load(path)  # (512, 512)

    if patch_size:
        ix = random.randrange(0, input_np.shape[0] - patch_size + 1)
        iy = random.randrange(0, input_np.shape[1] - patch_size + 1)
        input_np = input_np[ix:ix + patch_size, iy:iy + patch_size]

    ret, input_np = cv2.threshold(input_np, 0, 1, cv2.THRESH_BINARY)
    input_tensor = torch.from_numpy(input_np).float()
    return input_tensor.unsqueeze(0)


def load_image_numpy_gt(group, patch_size):
    dataset_name = group[1].split('/')[5]  # which means using over/under exposures
    path = group[1].replace(dataset_name, f'{dataset_name}_np')
    path = path.replace('JPG', 'npy').replace('jpg', 'npy')
    input_np = np.load(path)

    if patch_size:
        ix = random.randrange(0, input_np.shape[0] - patch_size + 1)
        iy = random.randrange(0, input_np.shape[1] - patch_size + 1)
        input_np = input_np[ix:ix + patch_size, iy:iy + patch_size]

    ret, input_np = cv2.threshold(input_np, 0, 1, cv2.THRESH_BINARY)
    input_tensor = torch.from_numpy(input_np).float()
    return input_tensor.unsqueeze(0)


def transform():
    return Compose([
        ToTensor(),
    ])


def BGR2RGB_toTensor(inputs, target):
    inputs = inputs[:, :, [2, 1, 0]]
    target = target[:, :, [2, 1, 0]]
    inputs = torch.from_numpy(np.ascontiguousarray(np.transpose(inputs, (2, 0, 1)))).float()
    target = torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))).float()
    return inputs, target


class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """

    def __init__(self, upscale_factor, data_augmentation, group_file, patch_size, black_edges_crop, hflip, rot,
                 transform=transform(), mask=True):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.black_edges_crop = black_edges_crop
        self.hflip = hflip
        self.rot = rot

    def __getitem__(self, index):

        inputs, target = load_image_train2(self.image_filenames[index])

        if self.patch_size != None:
            inputs, target = get_patch(inputs, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            inputs, target = augment(inputs, target, self.hflip, self.rot)

        if self.transform:
            inputs, target = BGR2RGB_toTensor(inputs, target)

        return {'LQ': inputs,
                'GT': target,
                'LQ_path': self.image_filenames[index][0],
                'GT_path': self.image_filenames[index][1]}

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromFolderSingle(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """

    def __init__(self, upscale_factor, data_augmentation, group_file, patch_size, black_edges_crop, hflip, rot,
                 transform=transform(), mask=True):
        super(DatasetFromFolderSingle, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.black_edges_crop = black_edges_crop
        self.hflip = hflip
        self.rot = rot
        self.mask = mask

    def __getitem__(self, index):
        inputs, target = load_image_train2(self.image_filenames[index])

        if self.patch_size != None:
            inputs, target = get_patch(inputs, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            inputs, target = augment(inputs, target, self.hflip, self.rot)

        if self.transform:
            inputs, target = BGR2RGB_toTensor(inputs, target)

        return {'LQ': inputs,
                'GT': target,
                'LQ_path': self.image_filenames[index][0],
                'GT_path': self.image_filenames[index][1]}

    def __len__(self):
        return len(self.image_filenames)
