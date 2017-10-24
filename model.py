import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
class pre_res_net(nn.Module):
    def __init__(self):
        super(pre_res_net, self).__init__()
        model = models.resnet152(pretrained=True)
        modules = list(model.children())[:-1]
        self.resnet = nn.Sequential(*modules)
    def forward(self,images):
        features = self.resnet(images)
        return features
