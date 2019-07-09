from torch.utils.data import Dataset
from .preprosses import *
from PIL import Image



class CaravanDataset(Dataset):
    def __init__(self, data_dir, mask_dir, image_size=512):
        super(CaravanDataset, self).__init__()
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.image_list = split_ids(get_image_ids(data_dir), 2)
        self.image_size = image_size


    def __getitem__(self, id):
        image_id, pos = self.image_list[id]
        img = Image.open(os.path.join(self.data_dir, image_id+'.jpg'))
        mask = Image.open(os.path.join(self.mask_dir, image_id+'_mask.gif'))

        img, mask = self.transforms(img, mask, pos)
        return {'img':img, 'mask':mask}

    def __len__(self):
        return len(self.image_list)

    def transforms(self, img, mask, pos):
        img, mask = Resize(self.image_size)(img, mask)
        img, mask = RandCrop()(img, mask, pos)
        img, _ = Normalization()(img)
        img, mask = ToCHW()(img, mask)
        return img, mask


class MembraneDataset(Dataset):
    def __init__(self, data_dir, mask_dir, image_size=512, train=True):
        super(MembraneDataset, self).__init__()
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.image_ids = get_image_ids(data_dir)
        self.train = train

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = Image.open(os.path.join(self.data_dir, image_id+'.png'))
        mask = Image.open(os.path.join(self.mask_dir, image_id+'.png'))
        img, mask = self.transforms(img, mask, self.train)
        return {"img":img, "mask":mask}

    def __len__(self):
        return len(self.image_ids)

    def transforms(self, img, mask, train=True):
        img, mask = Resize(self.image_size)(img, mask)
        if train:
            img, mask = RandomHorizontalFlip()(img, mask)
            img, mask = RandomVerticalFlip()(img, mask)
        img, mask = Normalization()(img, mask)
        img = np.expand_dims(img, 0)  ##add the channel dim
        return img, mask









