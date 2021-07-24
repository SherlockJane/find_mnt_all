import torch
import torch.nn as nn
from torch.nn import init
# import cv2 as cv
import numpy as np
from PIL import Image
import math
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(1, 1), bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = out * self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = out * self.relu(out)
        return out

class BasicBlock_sig(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_sig, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.sigmoid = nn.Sigmoid()
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = out * self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.sigmoid(out)
        return out


def make_layer(block, inplanes, planes, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    return nn.Sequential(*layers)


def make_block(block):
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])  # out_channels 表示给几个核
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)  # 因为输入是list，所以要用*来引


def get_model(out_channel):
    blocks = {}
    models = {}
    block0 = [{'conv1_1': [1, 48, 3, 2, 0]}]
    models['block0'] = make_block(block0)
    models['block1_1'] = make_layer(BasicBlock, 48, 96, 2)
    models['block1_2'] = make_layer(BasicBlock, 96, 128, 1)
    models['block1_3'] = make_layer(BasicBlock, 128, 256, 1)
    models['block1_4'] = make_layer(BasicBlock, 256, 512, 1)
    models['block1_5'] = make_layer(BasicBlock, 512, 256, 1)
    models['block1_6'] = make_layer(BasicBlock, 256, 128, 1)
    models['block1_7'] = make_layer(BasicBlock, 128, 64, 1)
    models['block1_8'] = make_layer(BasicBlock_sig, 64, out_channel, 1)

    class mnt_model(nn.Module):
        def __init__(self, model_dict):
            super(mnt_model, self).__init__()
            self.model0 = model_dict['block0']
            self.model1_1 = model_dict['block1_1']
            self.model1_2 = model_dict['block1_2']
            self.model1_3 = model_dict['block1_3']
            self.model1_4 = model_dict['block1_4']
            self.model1_5 = model_dict['block1_5']
            self.model1_6 = model_dict['block1_6']
            self.model1_7 = model_dict['block1_7']
            self.model1_8 = model_dict['block1_8']
            self._initialize_weights_norm()

        def forward(self, x):
            x = self.model0(x)
            # print("0", x.shape)
            x = self.model1_1(x)
            # print("1", x.shape)
            x = self.model1_2(x)
            # print("2", x.shape)
            x = self.model1_3(x)
            # print("3", x.shape)
            x = self.model1_4(x)
            # print("4", x.shape)
            x = self.model1_5(x)
            # print("5", x.shape)
            x = self.model1_6(x)
            # print("6", x.shape)
            x = self.model1_7(x)
            x = self.model1_8(x)

            # x = self.avgpool(x)
            # x = x.view(x.size(0), -1)
            # x = self.fc(x)
            return x

        def _initialize_weights_norm(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):  # 判断数据类型是否相同
                    # init.normal_(m.weight, mean=0.01,std=0.0001)  # 参数初始化（正态分布） 对卷积层操作
                    torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant_(m.bias, 0.0)  # 初始化偏置（常值分布）
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 0.0001)
                    nn.init.constant_(m.bias, 0.001)

    model = mnt_model(models)
    return model


if __name__ == "__main__":
    model = get_model()
    print(model)
    # for p in model.parameters():
    #     print(p)
    # imgpath = "F:/DYSY/fingerprint_paper/HRNet/mnt_test/data/train/DB1_A_2002/BMP/2002DB1A1_1.bmp"
    # img = Image.open(imgpath)
    # origin_img = np.array(img)
    # (h, w) = origin_img.shape[:2]
    # d0 = max(h, w)
    # if h > w:
    #     tran_img = cv.copyMakeBorder(origin_img, 0, 0, int(np.floor((h - w) / 2.0)), int(np.ceil((h - w) / 2.0)),
    #                                  cv.BORDER_CONSTANT, value=255)
    # else:
    #     tran_img = cv.copyMakeBorder(origin_img, int(np.floor((w - h) / 2.0)), int(np.ceil((w - h) / 2.0)), 0, 0,
    #                                  cv.BORDER_CONSTANT, value=255)
    # # print(tran_img.shape)
    # resize_img = cv.resize(tran_img, (384, 384))
    #
    # tran = transforms.ToTensor()  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
    # img_tensor = tran(resize_img)
    # data = {'image': resize_img, 'label': 1}
    #
    # # out = model(img_tensor)
    # # print(out.shape)
