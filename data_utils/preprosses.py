import os
import torchvision.transforms.functional as F
import random
import numpy as np

def get_image_ids(root_dir):
    ids = os.listdir(root_dir)
    ids = [os.path.splitext(id)[0] for id in ids]
    return ids

def split_ids(image_ids, n=2):
    return [(id, i) for id in image_ids for i in range(n)]


class Resize(object):
    def __init__(self, size):
        self.size = size

    def  __call__(self, img, mask, pos=None):
        img = F.resize(img, self.size)

        mask = F.resize(mask, self.size)
        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        return img, mask

class RandCrop(object):
    def __call__(self, img, mask, pos):
        h = img.shape[0]
        if pos == 0:
            img = img[:,:h]
            mask = mask[:,:h]
        else:
            img = img[:,-h:]
            mask = mask[:,-h:]

        return img, mask

class ToTensor(object):
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)
        return img, mask

class Normalization(object):
    def __call__(self, img, mask=None, pos=None):
        img = img / 255.
        if mask is not None:
            mask = mask / 255.
        return img, mask

class ToCHW(object):
    def __call__(self, img, mask, pos=None):
        img = np.transpose(img, (2, 0, 1))
        return img, mask

class RandomHorizontalFlip(object):
    def __call__(self, img, mask):
        if random.randint(0, 1):
            img = img[:,::-1].copy()
            mask = mask[:,::-1].copy()
        return img, mask

class RandomVerticalFlip(object):
    def __call__(self, img, mask):
        if random.randint(0, 1):
            img = img[::-1, :].copy()
            mask = mask[::-1, :].copy()
        return img, mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, pos=None):
        for t in self.transforms:
            img, mask = t(img, mask, pos)
        return img, mask



def merge_image(image_left, image_right, width):
    h = image_left.shape[0]
    full_image = np.zeros((h, width), dtype=np.uint8)
    full_image[:,:width//2+1] = image_left[:, :width//2+1]
    full_image[:,width//2+1:] = image_right[:, -(width//2-1):]
    return full_image




