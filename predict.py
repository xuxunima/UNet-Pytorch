import torch
import numpy as np
from unet import UNet
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from data_utils import merge_image
import torchvision.transforms as trfm
import os
from skimage import io
import torchvision.transforms.functional as F

def predict_net(net, image_path, save_path, image_size, gpu):
    net.eval()
    img = Image.open(image_path)
    original_w, original_h = img.size[0], img.size[1]

    if (original_h > original_w):
        raise Exception("the width of input image must large than the height!")

    img = np.array(F.resize(img, image_size), dtype=np.float32)
    img_left = img[:,:image_size] / 255
    img_right = img[:,-image_size:] / 255

    img_left = torch.from_numpy(np.transpose(img_left, (2, 0, 1)))
    img_right = torch.from_numpy(np.transpose(img_right, (2, 0, 1)))

    if gpu:
        img_left = img_left.cuda()
        img_right = img_right.cuda()

    img_left = img_left.unsqueeze(0)
    img_right = img_right.unsqueeze(0)
    with torch.no_grad():
        pred_left = net(img_left)[0]
        pred_right = net(img_right)[0]

        pred_left = pred_left.squeeze().cpu().numpy()
        pred_right = pred_right.squeeze().cpu().numpy()

        tf = trfm.Compose([trfm.ToPILImage(),
                           trfm.Resize(original_h),
                           trfm.ToTensor()])
        pred_left = tf(pred_left)
        pred_right = tf(pred_right)

        pred_left = pred_left.squeeze().cpu().numpy()
        pred_right = pred_right.squeeze().cpu().numpy()
    pred_left = pred_left > 0.5
    pred_right = pred_right > 0.5
    pred_left = np.array(pred_left, dtype=np.uint8) * 255
    pred_right = np.array(pred_right, dtype=np.uint8) * 255

    full_image = merge_image(pred_left, pred_right, original_w)
    io.imsave(save_path, full_image)

def predict_mem_net(net, image_path, save_path, image_size, gpu):
    net.eval()
    img = Image.open(image_path)
    original_w, original_h = img.size[0], img.size[1]


    img = np.array(F.resize(img, image_size), dtype=np.float32)
    img = img / 255
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)


    if gpu:
        img = img.cuda()

    img = img.unsqueeze(0)
    with torch.no_grad():
        pred_mask = net(img)[0]
        pred_mask = pred_mask.squeeze().cpu().numpy()

        tf = trfm.Compose([trfm.ToPILImage(),
                           trfm.Resize(original_h),
                           trfm.ToTensor()])
        pred_mask = tf(pred_mask)

        pred_mask = pred_mask.squeeze().numpy()

    pred_mask = pred_mask > 0.5
    pred_mask = np.array(pred_mask, dtype=np.uint8) * 255

    io.imsave(save_path, pred_mask)

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--gpu", "-g", default=False, dest='gpu', action='store_true',
                      help="if use cuda")
    parse.add_argument("--model-path", '-l', type=str, dest='load', required=True,
                       help='the pre trained model path')
    return parse.parse_args()

if __name__ == "__main__":
    args = get_args()

    image_path = "image.jpg"
    output_path = "output.jpg"
    image_size = 512

    net = UNet(in_channel=1, num_classes=1)

    if args.gpu:
        print("Use CUDA!")
        net = net.cuda()
        net.load_state_dict(torch.load(args.load))
        print("Load model from ", args.load)
    else:
        print("Use CPU!")
        net = net.cpu()
        net.load_state_dict(torch.load(args.load, map_location='cpu'))
        print("Load model from ", args.load)

    #predict_net(net, image_path, output_path, image_size, args.gpu)
    predict_mem_net(net, image_path, output_path, image_size, args.gpu)



