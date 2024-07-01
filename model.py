import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
import wandb
import random
import numpy as np


class UNet_1(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0.5):
        super(UNet_1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.up1 = DoubleConv(1024, 256)
        self.up2 = DoubleConv(512, 128)
        self.up3 = DoubleConv(256, 64)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid_activation = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(torch.cat([x4, x5], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
        logits = self.outc(x)
        # logits = self.sigmoid_activation(x)

        return logits




class UNet_light(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0.5):
        super(UNet_light, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = DoubleConv(16, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 128)
        self.up1 = DoubleConv(256, 64)
        self.up2 = DoubleConv(128, 32)
        self.up3 = DoubleConv(64, 16)
        self.up4 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)
        self.sigmoid_activation = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(torch.cat([x4, x5], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
        logits = self.outc(x)
        # logits = self.sigmoid_activation(x)

        return logits


class BothNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BothNet, self).__init__()

        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 64)

        self.top_up_1 = DoubleConv(64, 32)
        self.top_up_2 = DoubleConv(32, 16)
        self.top_up_3 = DoubleConv(16, 8)
        self.top_up_4 = DoubleConv(8, 8)

        self.top_down_1 = DoubleConv(16, 16)
        self.top_down_2 = DoubleConv(32, 32)
        self.top_down_3 = DoubleConv(64, 64)

        self.bot_up_1 = DoubleConv(1024, 256)
        self.bot_up_2 = DoubleConv(512, 128)
        self.bot_up_3 = DoubleConv(256, 64)

        self.bot_down_1 = DoubleConv(64, 128)
        self.bot_down_2 = DoubleConv(128, 256)
        self.bot_down_3 = DoubleConv(256, 512)
        self.bot_down_4 = DoubleConv(512, 512)

        self.out_mid = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

        self.sigmoid_activation = nn.Sigmoid()

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.inc(x)

        xu1 = self.top_up_1(x1)
        xu2 = self.top_up_2(xu1)
        xu3 = self.top_up_3(xu2)
        xud1 = self.top_up_4(xu3)

        xud2 = self.top_down_1(torch.cat([xud1, xu3], dim=1))
        xud3 = self.top_down_2(torch.cat([xud2, xu2], dim=1))
        xud4 = self.top_down_3(torch.cat([xud3, xu1], dim=1))

        xd1 = self.bot_down_1(x1)
        xd2 = self.bot_down_2(xd1)
        xd3 = self.bot_down_3(xd2)
        xdu1 = self.bot_down_4(xd3)

        xdu2 = self.bot_up_1(torch.cat([xdu1, xd3], dim=1))
        xdu3 = self.bot_up_2(torch.cat([xdu2, xd2], dim=1))
        xdu4 = self.bot_up_3(torch.cat([xdu3, xd1], dim=1))

        middle_out = self.out_mid(torch.cat([xdu4, xud4], dim=1))

        logits = self.outc(middle_out)
    

        return logits
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    


class ResNetFeatures(nn.Module):
    def __init__(self, output_size, saved_model_path):
        super(ResNetFeatures, self).__init__()
        resnet = models.resnet50(weights=None)
        resnet.fc = torch.nn.Linear(2048,19)
#         resnet.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Load your pretrained weights here if you have them
        checkpoint = torch.load(saved_model_path)

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        #print(state_dict.keys())
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                #pdb.set_trace()
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        
        '''
        # remove prefix
        state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
        '''
        #args.start_epoch = 0
        resnet.load_state_dict(state_dict, strict=False)

        # Remove the fully connected layer and the average pooling layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class FusionNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super(FusionNet, self).__init__()
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1)  # Reduce to 1 channel
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class RGB_DEM_to_SO(nn.Module):
    def __init__(self, resnet_output_size, 
                 fusion_output_size, 
                 model_choice, 
                 resnet_saved_model_path,
                 input_choice='RD', 
                 dropout_rate=0.5,
                 number_of_in_channels=2):
        
        super(RGB_DEM_to_SO, self).__init__()
        self.resnet = ResNetFeatures(output_size=resnet_output_size, saved_model_path=resnet_saved_model_path)
        self.fusion_net = FusionNet(input_channels=6*2048, output_size=fusion_output_size)
        self.unet = UNet_1(n_channels=number_of_in_channels, n_classes=9, dropout_rate=dropout_rate)
        self.unet_light = UNet_light(n_channels=number_of_in_channels, n_classes=9, dropout_rate=dropout_rate)
        self.onet = BothNet(in_channels=number_of_in_channels, out_channels=9)
        self.model_choice = model_choice
        self.input_choice = input_choice

    def forward(self, dem, rgbs):

        # rgbs is a list of RGB images
        features = [self.resnet(rgb) for rgb in rgbs]
        features = torch.cat(features, dim=1)  # Concatenate features along the channel dimension
        fused = self.fusion_net(features)

        if self.input_choice == 'RD':
            
            # Concatenate DEM and fused features
            combined_input = torch.cat((dem, fused), dim=1)
            if self.model_choice == "Unet_1":
                so_output = self.unet(combined_input)
            if self.model_choice == "Unet_light":
                so_output = self.unet_light(combined_input)
            if self.model_choice == "Onet":
                so_output = self.onet(combined_input)
        
        if self.input_choice == 'D':

            # DEM
            combined_input = dem
            if self.model_choice == "Unet_1":
                so_output = self.unet(combined_input)
            if self.model_choice == "Unet_light":
                so_output = self.unet_light(combined_input)
            if self.model_choice == "Onet":
                so_output = self.onet(combined_input)
        
        if self.input_choice == 'R':

            # RGB
            combined_input = fused
            if self.model_choice == "Unet_1":
                so_output = self.unet(combined_input)
            if self.model_choice == "Unet_light":
                so_output = self.unet_light(combined_input)
            if self.model_choice == "Onet":
                so_output = self.onet(combined_input)

        return so_output


class LightweightUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightUnet, self).__init__()
        # Define a simple U-Net-like structure
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = torch.relu(self.down(x))
        x = torch.relu(self.up(x))
        x = self.out_conv(x)


class ChainedUnets(nn.Module):
    def __init__(self, 
                 resnet_output_size, 
                 fusion_output_size, 
                 resnet_saved_model_path,
                 num_unets, 
                 in_channels=3, 
                 out_channels=1):
        
        super(ChainedUnets, self).__init__()
        self.resnet = ResNetFeatures(output_size=resnet_output_size, saved_model_path=resnet_saved_model_path)
        self.fusion_net = FusionNet(input_channels=6*2048, output_size=fusion_output_size)
        self.initial_unet = UNet_light(in_channels - 1, out_channels)
        self.unets = nn.ModuleList([UNet_light(in_channels, out_channels) for _ in range(num_unets-1)])
        self.out_channels = out_channels
    
    def forward(self, dem, rgbs):

        features = [self.resnet(rgb) for rgb in rgbs]
        features = torch.cat(features, dim=1)  # Concatenate features along the channel dimension
        fused = self.fusion_net(features)
        combined_input = torch.cat((dem, fused), dim=1)

        outputs = []
        input_x = combined_input
        x = self.initial_unet(input_x)
        outputs.append(x)
        for i, unet in enumerate(self.unets):
            # Detach x to cut off gradients flowing into earlier layers
            x = unet(torch.cat([x.detach(), combined_input], dim=1))
            outputs.append(x)
        return outputs
