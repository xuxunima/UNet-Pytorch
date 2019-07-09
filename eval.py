from dice_loss import dice_coeff
import numpy as np
from skimage import io
import torch
import torchvision
import os
import time

def eval_net(net, val_data, loss_fn, gpu, save_image, epoch):
    net.eval()
    total = 0.0
    total_loss = 0.0
    for i, data in enumerate(val_data):
        img = data['img']
        true_mask = data['mask']
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        pred_mask = net(img)
        pred_mask_flat = pred_mask.view(-1)
        true_mask_falt = true_mask.view(-1)
        loss = loss_fn(pred_mask_flat, true_mask_falt)
        total_loss += loss.item()

        pred_mask = (pred_mask > 0.5).float()
        total += dice_coeff(pred_mask, true_mask).item()

        if save_image:
            if gpu:
                img = img.cpu()
                pred_mask = pred_mask.cpu()
                true_mask = true_mask.cpu()
            true_mask = true_mask.unsqueeze(1)
            img = np.transpose(torchvision.utils.make_grid(img, normalize=True).numpy(), (1, 2, 0))
            pred_mask = np.transpose(torchvision.utils.make_grid(pred_mask, normalize=True).numpy(), (1, 2, 0))
            true_mask = np.transpose(torchvision.utils.make_grid(true_mask, normalize=True).numpy(), (1, 2, 0))


            full_img = np.concatenate([img, pred_mask, true_mask], axis=0)
            full_img *= 255
            full_img = np.array(full_img, dtype=np.uint8)

            if not os.path.exists(save_image):
                os.mkdir(save_image)
            io.imsave(os.path.join(save_image, "epoch_%d_%d.jpg"%(epoch, i+1)), full_img)
            if i == 9:
                save_image = False
    return total / (i + 1), total_loss / (i+1)



