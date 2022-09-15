import torch
from torch import nn
from torch.nn import functional as F
import quantization as cs
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8Bias, Int16Bias, Int32Bias, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat

### CUSTOM LAYERS

class Quant_ResidualBlock_custom(nn.Module):
    def __init__(self, in_channels, out_channels, int_channels=None, bias=True, block_type="Residual33", 
                 halve_resolution=False, squeeze_and_excite=False, SE_ratio=2, quantization=True,
                 weight_bit=8, act_bit=8, bias_bit=16, quant_method='scale', alpha_coeff=10.0):
        super(Quant_ResidualBlock_custom, self).__init__()

        self.bias = bias
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.alpha_coeff = alpha_coeff

        self.squeeze_and_excite = squeeze_and_excite
        self.halve_resolution = halve_resolution
        stride = (2, 2) if self.halve_resolution else (1, 1)
        # Set up the main sequence of layers
        if block_type == "Residual33":
            self.res_block = Quant_Residual33_custom(in_channels, out_channels, stride, bias=self.bias, weight_bit=self.weight_bit, act_bit=self.act_bit, bias_bit=self.bias_bit, quant_method=self.quant_method, alpha_coeff=self.alpha_coeff, quantization=self.quantization)
        elif block_type == "Residual131":
            self.res_block = Quant_Residual131_custom(in_channels, int_channels, out_channels, stride, bias=self.bias, weight_bit=self.weight_bit, act_bit=self.act_bit, bias_bit=self.bias_bit, quant_method=self.quant_method, alpha_coeff=self.alpha_coeff, quantization=self.quantization)
        else:
            exit("invalid block")
        # Configure the residual path
        self.upscale = True if in_channels != out_channels else False  # add layer to match channel size
        if self.upscale:
            if self.halve_resolution:
                self.up = cs.Conv2d(in_channels, out_channels, stride=stride, kernel_size=(3, 3), bias=self.bias, padding=(1,1), act_bit=self.act_bit,
                                    weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
            else:
                self.up = cs.Conv2d(in_channels, out_channels, stride=(1, 1), kernel_size=(1, 1), bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit,
                                    bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        if self.halve_resolution and not self.upscale:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if self.squeeze_and_excite:
            self.se = Quant_SqueezeAndExcite_custom(out_channels, SE_ratio, bias=self.bias)
        self.act = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)

    def forward(self, x):
        out = self.res_block(x)
        res = self.up(x) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        ###collegamento epr squeeze and excite..
        res = self.se(res) if self.squeeze_and_excite else res
        out = self.act(out + res)
        return out

class Quant_Residual131_custom(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, stride=(1, 1), bias=True, quantization=True,
                 weight_bit=8, act_bit=8, bias_bit=16, quant_method='scale', alpha_coeff=10.0):
        super(Quant_Residual131_custom, self).__init__()

        self.bias = bias
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.alpha_coeff = alpha_coeff

        self.l1 = cs.Conv2d(in_channels, int_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.act1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l2 = cs.Conv2d(int_channels, int_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.act2 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l3 = cs.Conv2d(int_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

    def forward(self, x):
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        out = self.act2(out)
        out = self.l3(out)
        return out

class Quant_Residual33_custom(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=True, quantization=True,
                 weight_bit=8, act_bit=8, bias_bit=16, quant_method='scale', alpha_coeff=10.0):
        super(Quant_Residual33_custom, self).__init__()

        self.bias = bias
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.alpha_coeff = alpha_coeff

        self.l1 = cs.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.act1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l2 = cs.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

    def forward(self, x):
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        return out

class Quant_SqueezeAndExcite_custom(nn.Module): # TODO: complete SqueezeAndExcite
    def __init__(self, channels, ratio, bias=True):
        super(Quant_SqueezeAndExcite_custom, self).__init__()

        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 16
        self.quant_method = 'scale'
        self.alpha_coeff = 10.0

        # self.glob_pool = nn.AvgPool2d(kernel_size=(glob_avg_window, glob_avg_window))
        self.fc1 = cs.Linear(channels, int(float(channels)/float(ratio)), bias=self.bias)
        self.relu = cs.ReLU(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=True)
        self.fc2 = cs.Linear(int(float(channels)/float(ratio)), channels, bias=self.bias)
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

### CUSTOM FOLDED LAYERS SUPPORT

# tolto il supporto a separable conv e squeeze and excite
class Quant_ResidualBlock_custom_folded(nn.Module):
    def __init__(self, in_channels, out_channels, int_channels=None, bias=True, quantization=True,
                 block_type="Residual33", halve_resolution=False, bn_eps=1e-3, bn_momentum=0.01,
                 weight_bit=8, act_bit=8, bias_bit=16, quant_method='scale', alpha_coeff=10.0):
        super(Quant_ResidualBlock_custom_folded, self).__init__()

        self.bias = bias
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.alpha_coeff = alpha_coeff
        self.bn_eps= bn_eps
        self.bn_momentum = bn_momentum

        self.halve_resolution = halve_resolution
        stride = (2, 2) if self.halve_resolution else (1, 1)
        # Set up the main sequence of layers
        if block_type == "Residual33":
            self.res_block = Quant_Residual33_custom_folded(in_channels, out_channels, stride, bias=self.bias, bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, weight_bit=self.weight_bit, act_bit=self.act_bit, bias_bit=self.bias_bit, quant_method=self.quant_method, alpha_coeff=self.alpha_coeff, quantization=self.quantization)
        elif block_type == "Residual131":
            self.res_block = Quant_Residual131_custom_folded(in_channels, int_channels, out_channels, stride, bias=self.bias, bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, weight_bit=self.weight_bit, act_bit=self.act_bit, bias_bit=self.bias_bit, quant_method=self.quant_method, alpha_coeff=self.alpha_coeff, quantization=self.quantization)
        else:
            exit("invalid block")
        # Configure the residual path
        self.upscale = True if in_channels != out_channels else False  # add layer to match channel size
        if self.upscale:
            if self.halve_resolution:
                self.up = cs.Conv2d_folded(in_channels, out_channels, stride=stride, kernel_size=(3, 3), bias=self.bias, padding=(1,1), act_bit=self.act_bit,
                                    bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
            else:
                self.up = cs.Conv2d_folded(in_channels, out_channels, stride=(1, 1), kernel_size=(1, 1), bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit,
                                    bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        if self.halve_resolution and not self.upscale:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.act = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)

    def forward(self, x):
        out = self.res_block(x)
        res = self.up(x) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        out = self.act(out + res)
        return out

class Quant_Residual131_custom_folded(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, stride=(1, 1), bias=True, bn_eps=1e-3, bn_momentum=0.01,
                 weight_bit=8, act_bit=8, bias_bit=16, quant_method='scale', alpha_coeff=10.0, quantization=True):
        super(Quant_Residual131_custom_folded, self).__init__()

        self.bias = bias
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.alpha_coeff = alpha_coeff
        self.bn_eps= bn_eps
        self.bn_momentum = bn_momentum

        self.l1 = cs.Conv2d_folded(in_channels, int_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.act1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l2 = cs.Conv2d_folded(int_channels, int_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.act2 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l3 = cs.Conv2d_folded(int_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

    def forward(self, x):
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        out = self.act2(out)
        out = self.l3(out)
        return out

class Quant_Residual33_custom_folded(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=True, bn_eps=1e-3, bn_momentum=0.01,
                 weight_bit=8, act_bit=8, bias_bit=16, quant_method='scale', alpha_coeff=10.0, quantization=True):
        super(Quant_Residual33_custom_folded, self).__init__()

        self.bias = bias
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.alpha_coeff = alpha_coeff
        self.bn_eps= bn_eps
        self.bn_momentum = bn_momentum

        self.l1 = cs.Conv2d_folded(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.act1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l2 = cs.Conv2d_folded(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bn_eps=self.bn_eps, bn_momentum=self.bn_momentum, bias=self.bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

    def forward(self, x):
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        return out

### BREVITAS LAYERS

### Brevitas support for PACT ReLU function
class PACT_QuantReLU(nn.Module):
    def __init__(self, alpha, act_quant):
        super(PACT_QuantReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
        self.relu = qnn.QuantReLU(return_quant_tensor=True, act_quant=act_quant)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            out = torch.clamp(x, min=0, max=self.alpha.item())
        else:
            out = torch.clamp(x.value, min=0, max=self.alpha.item())
        out = self.relu(out)
        return out

class Quant_ResidualBlock_brevitas(nn.Module):
    def __init__(self, in_channels, out_channels, int_channels=None, SP_kernel_size=(3, 3), padding=(1, 1),
                 bias=True, block_type="Residual33", halve_resolution=False, squeeze_and_excite=False,
                 SP_replicas=1, SE_ratio=2, alpha=123.0):
        super(Quant_ResidualBlock_brevitas, self).__init__()

        self.bias = bias
        self.alpha_coeff = alpha

        self.identity = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Uint8ActPerTensorFloat)        

        self.squeeze_and_excite = squeeze_and_excite
        self.halve_resolution = halve_resolution
        stride = (2, 2) if self.halve_resolution else (1, 1)
        # Set up the main sequence of layers
        if block_type == "Residual33":
            self.res_block = Quant_Residual33_brevitas(in_channels, out_channels, stride, self.bias, alpha=self.alpha_coeff)
        elif block_type == "Residual131":
            self.res_block = Quant_Residual131_brevitas(in_channels, int_channels, out_channels, stride, self.bias, alpha=self.alpha_coeff)
        elif block_type == "SeparableConv2d":
            self.res_block = Quant_SeparableConv2d_brevitas(in_channels, out_channels, SP_kernel_size, padding, halve_resolution, self.bias, SP_replicas, alpha=self.alpha_coeff)
        else:
            exit("invalid block")
        # Configure the residual path
        self.upscale = True if in_channels != out_channels else False  # add layer to match channel size
        if self.upscale:
            if self.halve_resolution:
                self.up = qnn.QuantConv2d(in_channels, out_channels, stride=stride, kernel_size=(3, 3), padding=(1,1), weight_quant=Int8WeightPerTensorFloat, bias=self.bias, bias_quant=Int16Bias, return_quant_tensor=True)
            else:
                self.up = qnn.QuantConv2d(in_channels, out_channels, stride=(1, 1), kernel_size=(1, 1), weight_quant=Int8WeightPerTensorFloat, bias=self.bias, bias_quant=Int16Bias, return_quant_tensor=True)
        
            self.bn_up = qnn.BatchNorm2dToQuantScaleBias(num_features=out_channels, momentum=0.01, eps=1e-3, return_quant_tensor=True)#, input_quant=Uint8ActPerTensorFloat)
        if self.halve_resolution and not self.upscale:
            self.pool = qnn.QuantMaxPool2d(kernel_size=(2, 2), stride=2)

        if self.squeeze_and_excite:
            self.se = Quant_SqueezeAndExcite_brevitas(out_channels, SE_ratio, self.bias)
        self.act = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)

        self.quant_ID_2  = qnn.QuantIdentity(return_quant_tensor=True, act_quant=self.res_block.output_quant)

    def forward(self, x):
        out = self.identity(x)
        out = self.res_block(x)
        #out = self.quant_ID_1(out)
        res = self.bn_up(self.up(x)) if self.upscale else x
        #res = self.up(x) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        ###collegamento epr squeeze and excite..
        res = self.se(res) if self.squeeze_and_excite else res
        res = self.quant_ID_2(res)
        out = self.act(out + res)
        return out

class Quant_Residual131_brevitas(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, stride=(1, 1), bias=False, alpha=123.0):
        super(Quant_Residual131_brevitas, self).__init__()
        
        self.bias = bias
        self.alpha_coeff = alpha
    
        self.l1 = qnn.QuantConv2d(in_channels, int_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), weight_quant=Int8WeightPerTensorFloat, bias=self.bias, bias_quant=Int16Bias, return_quant_tensor=True)
        self.bn1 = qnn.BatchNorm2dToQuantScaleBias(num_features=int_channels, return_quant_tensor=True, eps=1e-3)#, input_quant=self.act_bit)
        self.act1 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.l2 = qnn.QuantConv2d(int_channels, int_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, weight_quant=Int8WeightPerTensorFloat, bias=self.bias, bias_quant=Int16Bias, return_quant_tensor=True)
        self.bn2 = qnn.BatchNorm2dToQuantScaleBias(num_features=int_channels, return_quant_tensor=True, eps=1e-3)#, input_quant=self.act_bit)
        self.act2 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.l3 = qnn.QuantConv2d(int_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), weight_quant=Int8WeightPerTensorFloat, bias=self.bias, bias_quant=Int16Bias, return_quant_tensor=True)
        self.bn3 = qnn.BatchNorm2dToQuantScaleBias(num_features=out_channels, return_quant_tensor=True, eps=1e-3, output_quant=Uint8ActPerTensorFloat)#, input_quant=self.act_bit)
        
        self.output_quant = self.bn3.output_quant

    def forward(self, x):
        #VEDERE SE x DA QUANTIZZARE
        out = self.l1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.l3(out)
        out = self.bn3(out)
        return out


class Quant_Residual33_brevitas(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=False, alpha=123.0):
        super(Quant_Residual33_brevitas, self).__init__()

        self.bias = bias
        self.alpha_coeff = alpha

        self.l1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, weight_quant=Int8WeightPerTensorFloat, bias=self.bias, bias_quant=Int16Bias, return_quant_tensor=True)
        self.bn1 = qnn.BatchNorm2dToQuantScaleBias(num_features=out_channels,return_quant_tensor=True, eps=1e-3)#, input_quant=self.act_bit)
        self.act1 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.l2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), weight_quant=Int8WeightPerTensorFloat, bias=self.bias, bias_quant=Int16Bias, return_quant_tensor=True)
        self.bn2 = qnn.BatchNorm2dToQuantScaleBias(num_features=out_channels, return_quant_tensor=True, eps=1e-3, output_quant=Uint8ActPerTensorFloat)#, input_quant=self.act_bit)
        
        self.output_quant = self.bn2.output_quant

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.l2(out)
        out = self.bn2(out)
        return out

class Quant_SeparableConv2d_brevitas(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(1, 1), halve_resolution=False, bias=True, replicas=1):
        super(Quant_SeparableConv2d_brevitas, self).__init__()
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

class Quant_SqueezeAndExcite_brevitas(nn.Module): # TODO: complete SqueezeAndExcite
    def __init__(self, channels, ratio, bias=True):
        super(Quant_SqueezeAndExcite_brevitas, self).__init__()

        #NICOLA : spostare i seguenti parametri su classi superiori
        self.weigh_bit = Int8WeightPerTensorFloat 
        self.act_bit = Uint8ActPerTensorFloat
        self.bias_bit = Int16Bias
        self.quant_method = 'scale'
        self.alpha_coeff = 123.0 # controlla alpha value   

        # self.glob_pool = nn.AvgPool2d(kernel_size=(glob_avg_window, glob_avg_window))
        self.fc1 = qnn.QuantLinear(channels, int(float(channels)/float(ratio)), bias=bias, return_quant_tensor=True)
        self.relu = qnn.QuantReLU(alpha=self.alpha_coeff)
        self.fc2 = qnn.QuantLinear(int(float(channels)/float(ratio)), channels, bias=bias, return_quant_tensor=True)
        self.sigmoid = qnn.QuantSigmoid(return_quant_tensor=True)
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