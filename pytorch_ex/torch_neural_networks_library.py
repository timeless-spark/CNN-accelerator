from pickle import NONE
import torch
import torch.nn.functional as f
from torch import nn
from torch import Tensor
from typing import Optional
from brevitas.export import FINNManager
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from layer_templates_pytorch import *

import brevitas



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
        self.act = nn.ReLU6()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256,10)

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
        self.act = nn.ReLU6()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128,10)

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
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        ### 1
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64,10)

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
        dropout: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        ### strozza il bottleneck dimezzando i canali per il layer conv3x3
        width = int(inplanes / 4)
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, padding=1)
        self.bn2 = norm_layer(width)
        ### l'output riadatta al numero di canali ricevuto in input
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop = dropout

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.drop is not None:
            out = self.drop(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ReducedBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d

        self.concat = None
        self.maxpool = None
        if inplanes != planes:
            self.concat_check = True
            self.concat = concatLayer()
        else:
            self.concat_check = False
        if stride != 1:
            self.stride_check = True
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.stride_check = False

        width = int(inplanes / 4)
        
        self.conv1 = conv1x1(planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x1(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = dropout

    def forward(self, x: Tensor) -> Tensor:
        if self.stride_check:
            x = self.maxpool(x)
        if self.concat_check:
            x = self.concat(x)
        
        identity = x

        out = self.conv1(x)
        if self.drop is not None:
            out = self.drop(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

### modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class ResNet(nn.Module):
    def __init__(self, blocks, layers, default_res, drop=[0, 0, 0]):
        super().__init__()

        if default_res:
            self._make_layer = self._make_layer_conv_res
        else:
            self._make_layer = self._make_layer_direct_res
        
        ### 32 -> 16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        ### 16 -> 8
        self.layer1 = self._make_layer(blocks[0], 64, 64, layers[0], stride=2, drop=drop[0])
        ### 8 -> 8
        self.layer2 = self._make_layer(blocks[1], 64, 128, layers[1], drop=drop[1])
        ### 8 -> 4
        self.layer3 = self._make_layer(blocks[2], 128, 128, layers[2], stride=2, drop=drop[2])
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
        self.linear = nn.Linear(128, 10)

        ### initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ### siccome il training è solo dei batchnorm layer provare anche altre inizializzazioni..
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def _make_layer_conv_res(self, block, inplanes, planes, blocks, stride=1, drop=0):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                    conv1x1(inplanes, planes, stride),
                    nn.BatchNorm2d(planes),
                )
        elif inplanes != planes:
            downsample = nn.Sequential(
                    conv1x1(inplanes, planes),
                    nn.BatchNorm2d(planes),
                )
        
        dropout = None
        if drop != 0:
            dropout = nn.Dropout2d(drop)

        layers = []
        ### il primo blocco fa il downsample
        layers.append(block(inplanes, planes, stride, downsample, dropout=dropout))
        for i in range(1, blocks):
            layers.append(block(planes, planes, dropout=dropout))

        return nn.Sequential(*layers)

    def _make_layer_direct_res(self, block, inplanes, planes, blocks, stride=1, drop=False):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
            if inplanes != planes:
                downsample.append(concatLayer())
        elif inplanes != planes:
            downsample = concatLayer()

        dropout = None
        if drop != 0:
            dropout = nn.Dropout2d(drop)

        layers = []
        ### il primo blocco fa il downsample
        layers.append(block(inplanes, planes, stride, downsample, dropout=dropout))
        ### si potrebbe pensare ad una crescita dei canali graduale nei vari blocchi...
        for i in range(1, blocks):
            layers.append(block(planes, planes, dropout=dropout))

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
        x = self.linear(x)

        return x

### resnet model with bottleneck:
#   - 13 conv layers
#   - 13 bn layers
#   - 1 fc
def isaResnet_14():
    model = ResNet([Bottleneck, Bottleneck, ReducedBlock], [1, 1, 2], default_res=True)
    return model



### resnet model with bottleneck:
#   - 25 conv layers
#   - 25 bn layers
#   - 1 fc
def isaResnet_26():
    model = ResNet([Bottleneck, Bottleneck, ReducedBlock], [2, 2, 4], default_res=True, drop=[0.2, 0, 0])
    return model

### resnet model with bottleneck:
#   - 49 conv layers
#   - 49 bn layers
#   - 1 fc
def isaResnet_50():
    model = ResNet([Bottleneck, ReducedBlock, ReducedBlock], [4, 4, 8], default_res=False, drop=[0.2, 0.1, 0])
    return model

### resnet model with bottleneck:
#   - 97 conv layers
#   - 97 bn layers
#   - 1 fc
def isaResnet_98():
    model = ResNet([Bottleneck, ReducedBlock, ReducedBlock], [8, 8, 16], default_res=False, drop=[0.4, 0.2, 0.1])
    return model

###exercize_3 model for the standard exercize

class ex3ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        ### 32 -> 16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        ### 16 -> 8
        self.res1_1 = ResidualBlock(in_channels=32, out_channels=32, bias=False, block_type="Residual33")
        self.res1_2 = ResidualBlock(in_channels=32, out_channels=64, halve_resolution=True, bias=False, block_type="Residual33", squeeze_and_excite=True, SE_ratio=2)

        ### 8 -> 8
        self.res2_1 = ResidualBlock(in_channels=64, int_channels=16, out_channels=64, bias=False, block_type="Residual131")
        self.res2_2 = ResidualBlock(in_channels=64, int_channels=16, out_channels=128, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=2)

        ### 8 -> 4
        self.res3_1 = ResidualBlock(in_channels=128, int_channels=32, out_channels=128, bias=False, block_type="Residual131")
        self.res3_2 = ResidualBlock(in_channels=128, int_channels=32, out_channels=256, halve_resolution=True, bias=False, block_type="Residual131", squeeze_and_excite=True, SE_ratio=4)

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
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
