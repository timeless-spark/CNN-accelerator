from torch import nn
from torch.nn import functional as F

class Quant_ResidualBlock_custom(nn.Module):
    def __init__(self, in_channels, out_channels, int_channels=None, SP_kernel_size=(3, 3), padding=(1, 1),
                 bias=True, block_type="Residual33", halve_resolution=False, squeeze_and_excite=False,
                 SP_replicas=1, SE_ratio=2):
        super(Quant_ResidualBlock_custom, self).__init__()
        self.squeeze_and_excite = squeeze_and_excite
        self.halve_resolution = halve_resolution
        stride = (2, 2) if self.halve_resolution else (1, 1)
        # Set up the main sequence of layers
        if block_type == "Residual33":
            self.res_block = Quant_Residual33_custom(in_channels, out_channels, stride, bias)
        elif block_type == "Residual131":
            self.res_block = Quant_Residual131_custom(in_channels, int_channels, out_channels, stride, bias)
        elif block_type == "SeparableConv2d":
            self.res_block = Quant_SeparableConv2d_custom(in_channels, out_channels, SP_kernel_size, padding, halve_resolution, bias, SP_replicas)
        else:
            exit("invalid block")
        # Configure the residual path
        self.upscale = True if in_channels != out_channels else False  # add layer to match channel size
        if self.upscale:
            if self.halve_resolution:
                self.up = cs.Conv2d(in_channels, out_channels, stride=stride, kernel_size=(3, 3), bias=bias, padding=(1,1), act_bit=self.act_bit,
                                    weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
            else:
                self.up = cs.Conv2d(in_channels, out_channels, stride=(1, 1), kernel_size=(1, 1), bias=bias, act_bit=self.act_bit, weight_bit=self.weight_bit,
                                    bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
            self.bn_up = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        if self.halve_resolution and not self.upscale:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if self.squeeze_and_excite:
            self.se = Quant_SqueezeAndExcite_custom(out_channels, SE_ratio, bias)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.res_block(x)
        res = self.bn_up(self.up(x)) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        ###collegamento epr squeeze and excite..
        res = self.se(res) if self.squeeze_and_excite else res
        out = self.act(out + res)
        return out


class Quant_Residual131_custom(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, stride=(1, 1), bias=True):
        super(Quant_Residual131_custom, self).__init__()
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


class Quant_Residual33_custom(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=True):
        super(Quant_Residual33_custom, self).__init__()
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

class Quant_SeparableConv2d_custom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(1, 1), halve_resolution=False, bias=True, replicas=1):
        super(Quant_SeparableConv2d_custom, self).__init__()
        self.separable_conv = nn.Sequential()
        # first replica is always present
        self.separable_conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=(1, 1),
                                             groups=in_channels, bias=bias, padding=padding))
        # self.separable_conv.append(nn.BatchNorm2d(num_features=in_channels, momentum=0.01, eps=1e-3))
        # self.separable_conv.append(nn.ReLU())
        self.separable_conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=bias))
        self.separable_conv.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3))
        if replicas == 1 and halve_resolution:
            self.separable_conv.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        if replicas > 1:
            self.separable_conv.append(nn.ReLU())
        for i in range(1, replicas):
            self.separable_conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1,1),
                                                 groups=out_channels, bias=bias, padding=padding))
            # self.separable_conv.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3))
            # self.separable_conv.append(nn.ReLU())
            self.separable_conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=bias))
            self.separable_conv.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3))
            if i < (replicas - 1):
                self.separable_conv.append(nn.ReLU())
            if i == (replicas - 1) and halve_resolution:
                self.separable_conv.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

    def forward(self, x):
        out = self.separable_conv(x)
        return out

class Quant_SqueezeAndExcite_custom(nn.Module): # TODO: complete SqueezeAndExcite
    def __init__(self, channels, ratio, bias=True):
        super(Quant_SqueezeAndExcite_custom, self).__init__()
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

##########################

class Quant_ResidualBlock_brevitas(nn.Module):
    def __init__(self, in_channels, out_channels, int_channels=None, SP_kernel_size=(3, 3), padding=(1, 1),
                 bias=True, block_type="Residual33", halve_resolution=False, squeeze_and_excite=False,
                 SP_replicas=1, SE_ratio=2):
        super(Quant_ResidualBlock_brevitas, self).__init__()
        self.squeeze_and_excite = squeeze_and_excite
        self.halve_resolution = halve_resolution
        stride = (2, 2) if self.halve_resolution else (1, 1)
        # Set up the main sequence of layers
        if block_type == "Residual33":
            self.res_block = Quant_Residual33(in_channels, out_channels, stride, bias)
        elif block_type == "Residual131":
            self.res_block = Quant_Residual131(in_channels, int_channels, out_channels, stride, bias)
        elif block_type == "SeparableConv2d":
            self.res_block = Quant_SeparableConv2d(in_channels, out_channels, SP_kernel_size, padding, halve_resolution, bias, SP_replicas)
        else:
            exit("invalid block")
        # Configure the residual path
        self.upscale = True if in_channels != out_channels else False  # add layer to match channel size
        if self.upscale:
            if self.halve_resolution:
                self.up = qnn.QuantConv2d(in_channels, out_channels, stride=stride, kernel_size=(3, 3), bias=bias, padding=(1,1), act_bit=self.act_bit,
                                    weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
            else:
                self.up = qnn.QuantConv2d(in_channels, out_channels, stride=(1, 1), kernel_size=(1, 1), bias=bias, act_bit=self.act_bit, weight_bit=self.weight_bit,
                                    bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
            self.bn_up = qnn.QuantBatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        if self.halve_resolution and not self.upscale:
            self.pool = qnn.QuantMaxPool2d(kernel_size=(2, 2), stride=2)

        if self.squeeze_and_excite:
            self.se = Quant_SqueezeAndExcite(out_channels, SE_ratio, bias)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.res_block(x)
        res = self.bn_up(self.up(x)) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        ###collegamento epr squeeze and excite..
        res = self.se(res) if self.squeeze_and_excite else res
        out = self.act(out + res)
        return out
