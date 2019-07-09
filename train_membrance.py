import torch
import torch.nn as nn
import os
from eval import eval_net
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import CaravanDataset, MembraneDataset
import argparse
from unet import UNet
import numpy as np
import torchvision
import random
from skimage import io


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", "-e", default=30, type=int,dest='epochs',
                        help="number of epochs")
    parser.add_argument("--batch-size", "-b", default=4, type=int,dest='batchsize',
                        help="batch size")
    parser.add_argument("--save-path", "-s", default='snapshots', type=str,dest='savepath',
                        help="the dir to save the model")
    parser.add_argument("--save-image", '-m', default=False, dest='saveimage',
                        help="the dir to save the evaludated image")
    parser.add_argument("--gpu", "-g", default=False, action='store_true', dest='gpu',
                        help="if use cuda")
    parser.add_argument("--learning-rate","-lr", default=1e-3, type=float, dest='lr',
                        help='learning rate')
    parser.add_argument("--pre-model", "-pre", default=False, dest='load',
                        help="path of the pretrained model")
    parser.add_argument("--print-every", "-p", default=100, type=int, dest='print',
                        help="how many step to print the loss")

    return parser.parse_args()


def train_net(net, train_data, val_data, optimizer, lr_scheduler, loss_fn, epochs, gpu=True, save_model=None, save_image=False, print_step=100):
    num_batch = len(train_data)
    for epoch in range(epochs):
        net.train()
        print("Epoch %d/%d"%(epoch+1, epochs))
        epoch_loss = 0.0
        for step, data in enumerate(train_data):
            imgs, masks = data['img'], data['mask']
            if gpu:
                imgs = imgs.cuda()
                masks = masks.cuda()
            pred_masks = net(imgs)

            pred_masks_flat = pred_masks.view(-1)
            masks_flat = masks.view(-1)

            loss = loss_fn(pred_masks_flat, masks_flat)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % print_step == 0:
                print("Step %d/%d  Train loss: %.4f" % (step+1, num_batch, epoch_loss/(step+1)))

        print("Epoch %d/%d Finished!   Loss: %.4f"%(epoch+1, epochs, epoch_loss / (step+1)))
        lr_scheduler.step(epoch_loss)
        val_dice, val_loss = eval_net(net, val_data, loss_fn, gpu, save_image, epoch=epoch+1)
        print('Validation dice coeff is %f, loss is %.4f'%(val_dice, val_loss))

        if save_model:
            if not os.path.exists(save_model):
                os.mkdir(save_model)
            save_path = os.path.join(save_model, "epoch_%d_dice_%f.pth"%(epoch+1, val_dice))
            torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    args = get_args()

    net = UNet(1, 1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print("Load pretrained model from %s"%(args.load))

    if args.gpu:
        net.cuda()



    data_dir = "data/image"
    mask_dir = "data/mask"

    image_size = 512
    epochs = args.epochs

    print("Process training data......")

    #carvana_dataset = CaravanDataset(data_dir, mask_dir, image_size)
    menbrane_dataset = MembraneDataset(data_dir, mask_dir, image_size, True)
    menbrane_dataset_val = MembraneDataset(data_dir, mask_dir, image_size, False)


    # num_data = len(carvana_dataset)
    # num_train = int(num_data * 0.95)
    # indices = np.arange(num_data)

    num_data = len(menbrane_dataset)
    num_train = 25
    indices = np.arange(num_data)
    random.shuffle(indices)

    dataset_train = torch.utils.data.Subset(menbrane_dataset, indices[:num_train])
    dataset_val = torch.utils.data.Subset(menbrane_dataset_val, indices[num_train:])

    dataloader_train = DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=True)


    print("Training start\ntrain_samples: %d, val_sample: %d"%(len(dataset_train), len(dataset_val)))

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    loss_fn = nn.BCELoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True,
                                                       min_lr=1e-6, patience=10)

    train_net(net, dataloader_train, dataloader_val, optimizer,lr_scheduler,
              loss_fn, epochs, args.gpu, save_model=args.savepath, save_image=args.saveimage, print_step=args.print)

