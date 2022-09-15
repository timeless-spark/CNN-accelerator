from torch import nn
from torch.nn import functional as F

"""
    The residual block can be composed in several ways: 
    1- a pointwise downscaling layer that optionally can downscale the number of input features, then a convolutional 
       layer with 3x3 kernels, finally a pointwise layer, which upscale the features if the first layer downscaled them.
    2- two convolutional layers with 3x3 kernels, optionally can upscale or downscale the features
    3- depthwise separable convolution
    4- squeeze and excite module
    Then, if the number of input features is the same as the input features the output is added to the input, otherwise
    the input features are scaled to match the output and then added.

    More information can be found in the paper "Deep Residual Learning for Image Recognition"
    Link: https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
"""


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, int_channels=None, SP_kernel_size=(3, 3), padding=(1, 1),
                 bias=True, block_type="Residual33", halve_resolution=False, squeeze_and_excite=False,
                 SP_replicas=1, SE_ratio=2):
        super(ResidualBlock, self).__init__()
        self.squeeze_and_excite = squeeze_and_excite
        self.halve_resolution = halve_resolution
        stride = (2, 2) if self.halve_resolution else (1, 1)
        # Set up the main sequence of layers
        if block_type == "Residual33":
            self.res_block = Residual33(in_channels, out_channels, stride, bias)
        elif block_type == "Residual131":
            self.res_block = Residual131(in_channels, int_channels, out_channels, stride, bias)
        elif block_type == "SeparableConv2d":
            self.res_block = SeparableConv2d(in_channels, out_channels, SP_kernel_size, padding, halve_resolution, bias, SP_replicas)
        else:
            exit("invalid block")
        # Configure the residual path
        self.upscale = True if in_channels != out_channels else False  # add layer to match channel size
        if self.upscale:
            if self.halve_resolution:
                self.up = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=(3, 3), bias=bias, padding=(1,1))
            else:
                self.up = nn.Conv2d(in_channels, out_channels, stride=(1, 1), kernel_size=(1, 1), bias=bias)
            self.bn_up = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        if self.halve_resolution and not self.upscale:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if self.squeeze_and_excite:
            self.se = SqueezeAndExcite(out_channels, SE_ratio, bias)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.res_block(x)
        res = self.bn_up(self.up(x)) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        ###collegamento epr squeeze and excite..
        res = self.se(res) if self.squeeze_and_excite else res
        out = self.act(out + res)
        return out


class Residual131(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, stride=(1, 1), bias=True):
        super(Residual131, self).__init__()
        self.l1 = nn.Conv2d(in_channels, int_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=int_channels, momentum=0.01, eps=1e-3)
        self.act1 = nn.ReLU()
        self.l2 = nn.Conv2d(int_channels, int_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=int_channels, momentum=0.01, eps=1e-3)
        self.act2 = nn.ReLU()
        self.l3 = nn.Conv2d(int_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.l3(out)
        out = self.bn3(out)
        return out


class Residual33(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=True):
        super(Residual33, self).__init__()
        self.l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        self.act1 = nn.ReLU()
        self.l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.l2(out)
        out = self.bn2(out)
        return out


""" 
    The depthwise separable block is composed by a depthwise convolution followed by a pointwise convolution. The number
    of parameter is equal to kernel_size^2*in_channels + in_channels*out_channels.   
    More information can be found in the paper "Xception: Deep Learning with Depthwise Separable Convolutions"
    Link: https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf
"""


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(1, 1), halve_resolution=False, bias=True, replicas=1):
        super(SeparableConv2d, self).__init__()
        self.separable_conv_list = []
        # first replica is always present
        self.separable_conv_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=(1, 1),
                                             groups=in_channels, bias=bias, padding=padding))
        # self.separable_conv_list.append(nn.BatchNorm2d(num_features=in_channels, momentum=0.01, eps=1e-3))
        # self.separable_conv_list.append(nn.ReLU())
        self.separable_conv_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=bias))
        self.separable_conv_list.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3))
        if replicas == 1 and halve_resolution:
            self.separable_conv_list.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        if replicas > 1:
            self.separable_conv_list.append(nn.ReLU())
        for i in range(1, replicas):
            self.separable_conv_list.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1,1),
                                                 groups=out_channels, bias=bias, padding=padding))
            # self.separable_conv_list.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3))
            # self.separable_conv_list.append(nn.ReLU())
            self.separable_conv_list.append(nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=bias))
            self.separable_conv_list.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3))
            if i < (replicas - 1):
                self.separable_conv_list.append(nn.ReLU())
            if i == (replicas - 1) and halve_resolution:
                self.separable_conv_list.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        
        self.separable_conv = nn.Sequential(*self.separable_conv_list)

    def forward(self, x):
        out = self.separable_conv(x)
        return out


"""
The squeeze and excitation module is an evolution of the residual module presented in ResNet. The SE block is made of a 
residual block that feeds the exice path, composed by: a global pooling layer (squeezes the spatial dimensions to 1x1), 
a fully connected layer, a ReLU, another FC layer, a sigmoid activation and then the results are combined with a scale 
layer. Finally, the output of the scale and the input to the residual module are added. 
More information can be found in the paper "Squeeze-and-Excitation Networks"
Link: https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
"""

class SqueezeAndExcite(nn.Module): # TODO: complete SqueezeAndExcite
    def __init__(self, channels, ratio, bias=True):
        super(SqueezeAndExcite, self).__init__()
        # self.glob_pool = nn.AvgPool2d(kernel_size=(glob_avg_window, glob_avg_window))
        self.fc1 = nn.Linear(channels, int(float(channels)/float(ratio)), bias=bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(float(channels)/float(ratio)), channels, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.reshape((out.shape[0], out.shape[1], 1, 1))

        return out * x

