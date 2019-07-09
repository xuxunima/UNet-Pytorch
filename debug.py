import matplotlib.pyplot as plt
import numpy as np
from data_utils import MembraneDataset
from torch.utils.data import DataLoader
import torchvision

data_dir = "D:\\study\\paper\\segment\\U-Net\\unet-master\\data\\membrane\\train\\image"
mask_dir = "D:\\study\\paper\\segment\\U-Net\\unet-master\\data\\membrane\\train\\label"

menbrane_dataset = MembraneDataset(data_dir, mask_dir, 512, True)

dataloader_train = DataLoader(menbrane_dataset, batch_size=4, shuffle=True)
data = next(iter(dataloader_train))
img = data['img']
mask = data['mask']
mask = mask.unsqueeze(1)
img = np.transpose(torchvision.utils.make_grid(img).numpy(), (1,2,0))
mask = np.transpose(torchvision.utils.make_grid(mask).numpy(), (1,2,0))

plt.subplot(2,1,1)
plt.imshow(img)
plt.subplot(2,1,2)
plt.imshow(mask)
plt.show()



# print(len(menbrane_dataset))
# data = menbrane_dataset[0]
# img = data['img']
# mask = data['mask']
# print(img.shape)
# print(mask.shape)
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(mask, cmap='gray')
# plt.show()