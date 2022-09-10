from pickle import NONE
from re import T
import string
import torch
import torch.nn.functional as f
from torch import nn
from torch import Tensor
from typing import Optional
from brevitas.export import FINNManager
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from layer_templates_pytorch import *

import brevitas

import numpy as np

"""
edit this class to modify your neural network, try mixing Conv2d and Linear layers. See how the training time, loss and 
test error changes. Remember to add an activation layer after each CONV or FC layer. There are two ways to write a set 
of layers: the sequential statement or an explicit assignment. In the sequential assignment you just write a sequence 
of layers which are then wrapped up and called in order exactly as you have written them. This is useful for small 
sequences or simple structures, but prevents you from branching, merging or selectively extract features from the 
layers. The explicit assignment on the other hand allows  to write more complex structures, but requires you to 
instantiate the layers in the __init__ function and connect them inside the forward function."""

###exercize_1 model
class default_model(nn.Module):
    def __init__(self):
        super(default_model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 10)
        self.act = nn.ReLU()

        # TODO: add explicit weight initialization to overwrite default pytorch method, here there is a commented
        #       placeholder to show the syntax. Research the initialization methods online, choose one and justify why
        #       you decided to use that in particular.
        
        '''
        torch.nn.init.normal(self.linear1.weight)
        torch.nn.init.normal(self.linear2.weight)
        torch.nn.init.normal(self.linear3.weight)
        '''

        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        torch.nn.init.xavier_normal_(self.linear3.weight)

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        return out


# TODO: create new classes while doing the next exercises and choose a meaningful name

###exercize_2 model
class micro_resnet(nn.Module):
    def __init__(self):
        super(micro_resnet, self).__init__()
        self.conv2D_1 = nn.Conv2d(1,4,kernel_size=(3,3), stride=(2,2), padding=1)
        self.conv2D_2 = nn.Conv2d(4,16,kernel_size=(3,3), stride=(2,2), padding=1)
        self.skip1 = nn.MaxPool2d(kernel_size=(4,4), stride=(4,4))
        self.conv2D_3 = nn.Conv2d(16,4, kernel_size=(1,1), stride=(1,1))
        self.conv2D_4 = nn.Conv2d(4,4,kernel_size=(3,3), stride=(2,2), padding=1)
        self.conv2D_5 = nn.Conv2d(4,16,kernel_size=(1,1), stride=(1,1))
        self.skip2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.act = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64,10)

        torch.nn.init.xavier_normal_(self.conv2D_1.weight)
        torch.nn.init.xavier_normal_(self.conv2D_2.weight)
        torch.nn.init.xavier_normal_(self.conv2D_3.weight)
        torch.nn.init.xavier_normal_(self.conv2D_4.weight)
        torch.nn.init.xavier_normal_(self.conv2D_5.weight)
        torch.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        #downsample + res connection (2 layer)
        x_skip1 = self.skip1(x)
        out = self.conv2D_1(x)
        out = self.act(out)
        out = self.conv2D_2(out)
        out = out + x_skip1
        out = self.act(out)
        #bottleneck res connection (3 layer)
        x_skip2 = self.skip2(out)
        out = self.conv2D_3(out)
        out = self.act(out)
        out = self.conv2D_4(out)
        out = self.act(out)
        out = self.conv2D_5(out)
        out = out + x_skip2
        out = self.act(out)
        #flatten + dropout + fully connected (1 layer)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

class nano_resnet(nn.Module):
    def __init__(self):
        super(nano_resnet, self).__init__()
        self.conv2D_1 = nn.Conv2d(1,4,kernel_size=(3,3), stride=(2,2), padding=1)
        self.conv2D_2 = nn.Conv2d(4,8,kernel_size=(3,3), stride=(2,2), padding=1)
        self.skip1 = nn.MaxPool2d(kernel_size=(4,4), stride=(4,4))
        self.conv2D_3 = nn.Conv2d(8,2, kernel_size=(1,1), stride=(1,1))
        self.conv2D_4 = nn.Conv2d(2,2,kernel_size=(3,3), stride=(2,2), padding=1)
        self.conv2D_5 = nn.Conv2d(2,8,kernel_size=(1,1), stride=(1,1))
        self.skip2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.act = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32,10)

        torch.nn.init.xavier_normal_(self.conv2D_1.weight)
        torch.nn.init.xavier_normal_(self.conv2D_2.weight)
        torch.nn.init.xavier_normal_(self.conv2D_3.weight)
        torch.nn.init.xavier_normal_(self.conv2D_4.weight)
        torch.nn.init.xavier_normal_(self.conv2D_5.weight)
        torch.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        #downsample + res connection (2 layer)
        x_skip1 = self.skip1(x)
        out = self.conv2D_1(x)
        out = self.act(out)
        out = self.conv2D_2(out)
        out = out + x_skip1
        out = self.act(out)
        #bottleneck res connection (3 layer)
        x_skip2 = self.skip2(out)
        out = self.conv2D_3(out)
        out = self.act(out)
        out = self.conv2D_4(out)
        out = self.act(out)
        out = self.conv2D_5(out)
        out = out + x_skip2
        out = self.act(out)
        #flatten + dropout + fully connected (1 layer)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

###exercize_2bis model
class mini_resnet(nn.Module):
    def __init__(self):
        super(mini_resnet, self).__init__()
        ### 28
        self.conv2D_1 = nn.Conv2d(1,32,kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
        ### 14
        self.conv2D_2 = nn.Conv2d(32,64,kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
        ### 7
        self.skip1 = nn.MaxPool2d(kernel_size=(5,5), stride=(4,4), padding=2)
        self.conv2D_3 = nn.Conv2d(64,16, kernel_size=(1,1), stride=(1,1), bias=True)
        self.conv2D_4 = nn.Conv2d(16,16,kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
        ### 4
        self.conv2D_5 = nn.Conv2d(16,64,kernel_size=(1,1), stride=(1,1), bias=True)
        self.skip2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        self.act = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        ### 1
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256,10)

        torch.nn.init.kaiming_normal_(self.conv2D_1.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_2.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_3.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_4.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_5.weight)
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        #downsample + res connection (2 layer)
        x_skip1 = self.skip1(x)
        out = self.conv2D_1(x)
        out = self.act(out)
        out = self.conv2D_2(out)
        out = out + x_skip1
        out = self.act(out)
        #bottleneck res connection (3 layer)
        x_skip2 = self.skip2(out)
        out = self.conv2D_3(out)
        out = self.act(out)
        out = self.conv2D_4(out)
        out = self.act(out)
        out = self.conv2D_5(out)
        out = out + x_skip2
        out = self.act(out)
        #downsample
        out = self.avgpool(out)
        #flatten + dropout + fully connected (1 layer)
        out = self.flatten(out)
        out = self.linear(out)
        return out

class inv_resnet(nn.Module):
    def __init__(self):
        super(inv_resnet, self).__init__()
        ### 28
        self.conv2D_1 = nn.Conv2d(1,32,kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
        ### 14
        self.conv2D_2 = nn.Conv2d(32,16, kernel_size=(1,1), stride=(1,1), bias=True)
        self.conv2D_3 = nn.Conv2d(16,16,kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
        ### 7
        self.conv2D_4 = nn.Conv2d(16,32,kernel_size=(1,1), stride=(1,1), bias=True)
        self.skip1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        self.conv2D_5 = nn.Conv2d(32,64,kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
        ### 4
        self.act = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        ### 1
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256,10)

        torch.nn.init.kaiming_normal_(self.conv2D_1.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_2.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_3.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_4.weight)
        torch.nn.init.kaiming_normal_(self.conv2D_5.weight)
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        #downsample
        out = self.conv2D_1(x)
        out = self.act(out)
        #bottleneck res connection (3 layer)
        x_skip1 = self.skip1(out)
        out = self.conv2D_2(out)
        out = self.act(out)
        out = self.conv2D_3(out)
        out = self.act(out)
        out = self.conv2D_4(out)
        out = out + x_skip1
        out = self.act(out)
        out = self.conv2D_5(out)
        out = self.act(out)

        #downsample
        out = self.avgpool(out)
        #flatten + dropout + fully connected (1 layer)
        out = self.flatten(out)
        out = self.linear(out)
        return out

###exercize_3 model for only-batch-norm variation

### taken from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 0) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

### modulo che duplica un tensor lungo la dimensione del canale..
class concatLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat((x, x), 1)
        return out

### modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class Bottleneck(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        ### strozza il bottleneck dimezzando i canali per il layer conv3x3
        width = int(inplanes / 2)
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, padding=1)
        self.bn2 = norm_layer(width)
        ### l'output riadatta al numero di canali ricevuto in input
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

### modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class ResNet(nn.Module):
    def __init__(self, blocks_conn, layers, drop=0, init=None, sparse_ammount=0):
        super().__init__()

        res_conn_type = []
        for t in blocks_conn:
            if t:
                res_conn_type.append(self._make_layer_conv_res)
            else:
                res_conn_type.append(self._make_layer_reduced_res)
        
        ### 32 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        ### 32 -> 16
        self.layer1 = res_conn_type[0](32, 32, layers[0], stride=2)
        ### 16 -> 8
        self.layer2 = res_conn_type[1](32, 64, layers[1], stride=2)
        ### 8 -> 4
        self.layer3 = res_conn_type[2](64, 128, layers[2], stride=2)
        '''
        ### 8 -> 2
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=3, padding=0)
        '''
        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        '''
        self.linear = nn.Linear(512, 10)
        '''
        self.dropout = nn.Dropout(drop)
        self.linear = nn.Linear(128, 10)

        ### initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init is not None:
                    if init == "normal":
                        nn.init.normal_(m.weight, std=1.0)
                    elif init == "sparse":
                      #PYTORCH SPARSE INIT GIVES PROBLEMS IN TRAINING SOMEHOW..
                        #for i in range(m.weight.shape[0]):
                        #    for j in range(m.weight.shape[1]):
                        #        nn.init.sparse_(m.weight[i, j, :, :], sparsity=0.2, std=1.0)
                        #        m.weight = nn.parameter.Parameter(m.weight, requires_grad=True)
                      #'SPARSIFICATION' IS DONE BY HAND..
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        prova = torch.from_numpy(np.random.binomial(1, 1-sparse_ammount, m.weight.shape))
                        m.weight = nn.parameter.Parameter(m.weight * prova, requires_grad=True)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                if init is not None:
                        nn.init.normal_(m.weight, std=1.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a=0, b=1)
                nn.init.constant_(m.bias, 0)
            
    def _make_layer_conv_res(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                    conv1x1(inplanes, planes, stride),
                    nn.BatchNorm2d(planes),
                )
        
        layers = []
        ### il primo blocco fa il downsample
        layers.append(Bottleneck(inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(Bottleneck(planes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

### resnet model with bottleneck:
#   - 13 conv layers
#   - 13 bn layers
#   - 1 fc
def isaResNet_14():
    model = ResNet([True, True, True], [1, 1, 2])
    return model, "isaResNet_14"

### resnet model with bottleneck:
#   - 37 conv layers
#   - 37 bn layers
#   - 1 fc
def isaResNet_38():
    model = ResNet([True, True, True], [3, 3, 6])
    return model, "isaResNet_38"

### resnet model with bottleneck:
#   - 109 conv layers
#   - 109 bn layers
#   - 1 fc
  # standard version
def isaResNet_110():
    model = ResNet([True, True, True], [12, 12, 12])
    return model, "isaResNet_110"
  # wide normal distr initialization version
def isaResNet_110_normal():
    model = ResNet([True, True, True], [12, 12, 12], init="normal")
    return model, "isaResNet_110_normal"
  # sparse matrix version
def isaResNet_110_sparse():
    model = ResNet([True, True, True], [12, 12, 12], init="sparse", sparse_ammount=0.5)
    return model, "isaResNet_110_sparse"
  # dropout version
def isaResNet_110_dropout():
    model = ResNet([True, True, True], [12, 12, 12], drop=0.5)
    return model, "isaResNet_110_dropout"

### resnet model with bottleneck:
#   - 289 conv layers
#   - 289 bn layers
#   - 1 fc
def isaResNet_290():
    model = ResNet([True, False, False], [32, 32, 32], init="normal")
    return model, "isaResNet_290"

###exercize_3 model for the standard exercize

class ex3ResNet_small(nn.Module):
    def __init__(self):
        super().__init__()
        ### 32 -> 32
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        ### 32 -> 16
        self.res1_1 = ResidualBlock(in_channels=8, out_channels=8, bias=False, block_type="Residual33")
        self.res1_2 = ResidualBlock(in_channels=8, out_channels=16, halve_resolution=True, bias=False, block_type="Residual33")

        ### 16 -> 8
        self.res2_1 = ResidualBlock(in_channels=16, int_channels=8, out_channels=16, bias=False, block_type="Residual131")
        self.res2_2 = ResidualBlock(in_channels=16, int_channels=8, out_channels=32, halve_resolution=True, bias=False, block_type="Residual131")

        ### 8 -> 4
        self.res3_1 = ResidualBlock(in_channels=32, out_channels=32, bias=False, block_type="SeparableConv2d", SP_replicas=2)
        self.res3_2 = ResidualBlock(in_channels=32, out_channels=64, halve_resolution=True, bias=False, block_type="SeparableConv2d", SP_replicas=1)
        self.res3_3 = ResidualBlock(in_channels=64, out_channels=64, bias=False, block_type="SeparableConv2d", SP_replicas=1)

        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

class ex3ResNet_small_SE(nn.Module):
    def __init__(self):
        super().__init__()
        ### 32 -> 32
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        ### 32 -> 16
        self.res1_1 = ResidualBlock(in_channels=8, out_channels=8, bias=False, block_type="Residual33", squeeze_and_excite=True, SE_ratio=1)
        self.res1_2 = ResidualBlock(in_channels=8, out_channels=16, halve_resolution=True, bias=False, block_type="Residual33", squeeze_and_excite=True, SE_ratio=1)

        ### 16 -> 8
        self.res2_1 = ResidualBlock(in_channels=16, int_channels=8, out_channels=16, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=2)
        self.res2_2 = ResidualBlock(in_channels=16, int_channels=8, out_channels=32, halve_resolution=True, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=2)

        ### 8 -> 4
        self.res3_1 = ResidualBlock(in_channels=32, out_channels=32, bias=False, block_type="SeparableConv2d", SP_replicas=2, squeeze_and_excite=True, SE_ratio=2)
        self.res3_2 = ResidualBlock(in_channels=32, out_channels=64, halve_resolution=True, bias=False, block_type="SeparableConv2d", SP_replicas=1)
        self.res3_3 = ResidualBlock(in_channels=64, out_channels=64, bias=False, block_type="SeparableConv2d", SP_replicas=1, squeeze_and_excite=True, SE_ratio=2)

        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

class ex3ResNet_medium(nn.Module):
    def __init__(self):
        super().__init__()
        ### 32 -> 32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        ### 32 -> 16
        self.res1_1 = ResidualBlock(in_channels=16, out_channels=16, bias=False, block_type="Residual33")
        self.res1_2 = ResidualBlock(in_channels=16, out_channels=32, halve_resolution=True, bias=False, block_type="Residual33")

        ### 16 -> 8
        self.res2_1 = ResidualBlock(in_channels=32, int_channels=16, out_channels=32, bias=False, block_type="Residual131")
        self.res2_2 = ResidualBlock(in_channels=32, int_channels=16, out_channels=64, halve_resolution=True, bias=False, block_type="Residual131")

        ### 8 -> 8
        self.res3_1 = ResidualBlock(in_channels=64, int_channels=32, out_channels=64, bias=False, block_type="Residual131")
        self.res3_2 = ResidualBlock(in_channels=64, int_channels=32, out_channels=128, bias=False, block_type="Residual131")

        ### 8 -> 4
        self.res4_1 = ResidualBlock(in_channels=128, int_channels=64, out_channels=128, bias=False, block_type="Residual131")
        self.res4_2 = ResidualBlock(in_channels=128, int_channels=64, out_channels=256, halve_resolution=True, bias=False, block_type="Residual131")

        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

class ex3ResNet_medium_SE(nn.Module):
    def __init__(self):
        super().__init__()
        ### 32 -> 32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        ### 32 -> 16
        self.res1_1 = ResidualBlock(in_channels=16, out_channels=16, bias=False, block_type="Residual33", squeeze_and_excite=True, SE_ratio=1)
        self.res1_2 = ResidualBlock(in_channels=16, out_channels=32, halve_resolution=True, bias=False, block_type="Residual33", squeeze_and_excite=True, SE_ratio=1)

        ### 16 -> 8
        self.res2_1 = ResidualBlock(in_channels=32, int_channels=16, out_channels=32, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=1)
        self.res2_2 = ResidualBlock(in_channels=32, int_channels=16, out_channels=64, halve_resolution=True, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=1)

        ### 8 -> 8
        self.res3_1 = ResidualBlock(in_channels=64, int_channels=32, out_channels=64, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=2)
        self.res3_2 = ResidualBlock(in_channels=64, int_channels=32, out_channels=128, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=2)

        ### 8 -> 4
        self.res4_1 = ResidualBlock(in_channels=128, int_channels=64, out_channels=128, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=2)
        self.res4_2 = ResidualBlock(in_channels=128, int_channels=64, out_channels=256, halve_resolution=True, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=2)

        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

class ex3ResNet_large(nn.Module):
    def __init__(self):
        super().__init__()
        ### 32 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        ### 32 -> 16
        self.res1_1 = ResidualBlock(in_channels=32, out_channels=32, bias=False, block_type="Residual33")
        self.res1_2 = ResidualBlock(in_channels=32, out_channels=32, bias=False, block_type="Residual33")
        self.res1_3 = ResidualBlock(in_channels=32, out_channels=32, bias=False, block_type="Residual33")
        self.res1_4 = ResidualBlock(in_channels=32, out_channels=64, halve_resolution=True, bias=False, block_type="Residual33")

        ### 16 -> 8
        self.res2_1 = ResidualBlock(in_channels=64, int_channels=32, out_channels=64, bias=False, block_type="Residual131")
        self.res2_2 = ResidualBlock(in_channels=64, int_channels=32, out_channels=64, bias=False, block_type="Residual131")
        self.res2_3 = ResidualBlock(in_channels=64, int_channels=32, out_channels=64, bias=False, block_type="Residual131")
        self.res2_4 = ResidualBlock(in_channels=64, int_channels=64, out_channels=128, halve_resolution=True, bias=False, block_type="Residual131")

        ### 8 -> 8
        self.res3_1 = ResidualBlock(in_channels=128, int_channels=32, out_channels=128, bias=False, block_type="Residual131")
        self.res3_2 = ResidualBlock(in_channels=128, int_channels=32, out_channels=128, bias=False, block_type="Residual131")
        self.res3_3 = ResidualBlock(in_channels=128, int_channels=32, out_channels=128, bias=False, block_type="Residual131")
        self.res3_4 = ResidualBlock(in_channels=128, int_channels=64, out_channels=256, bias=False, block_type="Residual131")

        ### 8 -> 4
        self.res4_1 = ResidualBlock(in_channels=256, int_channels=64, out_channels=256, bias=False, block_type="Residual131")
        self.res4_2 = ResidualBlock(in_channels=256, int_channels=64, out_channels=256, bias=False, block_type="Residual131")
        self.res4_3 = ResidualBlock(in_channels=256, int_channels=64, out_channels=256, bias=False, block_type="Residual131")
        self.res4_4 = ResidualBlock(in_channels=256, int_channels=128, out_channels=512, halve_resolution=True, bias=False, block_type="Residual131")

        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        x = self.res1_4(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        x = self.res2_4(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
        x = self.res3_4(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        x = self.res4_4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x