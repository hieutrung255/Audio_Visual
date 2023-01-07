import torch
import torchvision
from .network import Resnet18, VisualVoiceUNet, weights_init
import os
import sys
sys.path.insert(0, '..')
from utils.utils import load_json

class ModelBuilder():
    #build for audio stream
    def build_unet(self, ngf=64, input_nc=1, output_nc=1, audioVisual_feature_dim=1280, identity_feature_dim=64, weights=''):
        net = VisualVoiceUNet(ngf, input_nc, output_nc, audioVisual_feature_dim)
        net.apply(weights_init)

        if len(weights) > 0:
            print('Loading weights for UNet')
            net.load_state_dict(torch.load(weights))
        return net

    

    