import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import pandas as pd
import numpy as np
from model import pre_res_net
from PIL import Image
import os


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])


def to_data(image_dir, net, transform):
    data = []
    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        img = load_image(os.path.join(image_dir, image), transform)
        fea = net(to_var(img))
        data.append(pd.DataFrame(fea.data.squeeze(2).squeeze(2).numpy()))
        if(i%100==0):
            print("[%d/%d] has been completed." % (i, num_images))
    data = pd.concat(data, ignore_index=True)
    return data
net = pre_res_net()
'''if you want to extract database for those video just use two lines below'''
# data = to_data('/Users/zhangzhihao/Documents/webbrain/data/resized_images', net, transform)
# data.to_csv('/Users/zhangzhihao/Documents/webbrain/data/features/feature_matrix.csv')
