from torch import nn
from torch.nn import functional as F
import quantization as cs
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8Bias, Int16Bias, Int32Bias, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat


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
            #self.bn_up = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        if self.halve_resolution and not self.upscale:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if self.squeeze_and_excite:
            self.se = Quant_SqueezeAndExcite_custom(out_channels, SE_ratio, bias)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.res_block(x)
        #res = self.bn_up(self.up(x)) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        ###collegamento epr squeeze and excite..
        res = self.se(res) if self.squeeze_and_excite else res
        out = self.act(out + res)
        return out

class Quant_Residual131_custom(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, stride=(1, 1), bias=True):
        super(Quant_Residual131_custom, self).__init__()

        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 16
        self.quant_method = 'scale'
        self.quantization = True
        self.alpha_coeff = 10.0

        self.l1 = cs.Conv2d(in_channels, int_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        #self.bn1 = nn.BatchNorm2d(num_features=int_channels, momentum=0.01, eps=1e-3)
        self.act1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l2 = cs.Conv2d(int_channels, int_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        #self.bn2 = nn.BatchNorm2d(num_features=int_channels, momentum=0.01, eps=1e-3)
        self.act2 = cs.ReLU(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l3 = cs.Conv2d(int_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        #self.bn3 = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        #NICOLA : vedi se devi quantizzare x
        out = self.l1(x)
        #out = self.bn1(out)
        out = self.act1(out)
        out = self.l2(out)
        #out = self.bn2(out)
        out = self.act2(out)
        out = self.l3(out)
        #out = self.bn3(out)
        return out

class Quant_Residual33_custom(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=True):
        super(Quant_Residual33_custom, self).__init__()

        #NICOLA : questa roba metila sul residual block, che la passa poi ai sottoblocks
        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 16
        self.quant_method = 'scale'
        self.quantization = True
        self.alpha_coeff = 10.0

        self.l1 = cs.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        #self.bn1 = cs.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        self.act1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.l2 = cs.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=bias, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        #self.bn2 = cs.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        #NICOLA : vedi se devi quantizzare x o no
        out = self.l1(x)
        #out = self.bn1(out)
        out = self.act1(out)
        out = self.l2(out)
        #out = self.bn2(out)
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

        self.weight_bit = 8
        self.act_bit = 8
        self.bias_bit = 16
        self.quant_method = 'scale'
        self.quantization = True
        self.alpha_coeff = 10.0

        # self.glob_pool = nn.AvgPool2d(kernel_size=(glob_avg_window, glob_avg_window))
        self.fc1 = cs.Linear(channels, int(float(channels)/float(ratio)), bias=bias)
        self.relu = cs.ReLU(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.fc2 = cs.Linear(int(float(channels)/float(ratio)), channels, bias=bias)
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
            self.res_block = Quant_Residual33_brevitas(in_channels, out_channels, stride, bias)
        elif block_type == "Residual131":
            self.res_block = Quant_Residual131_brevitas(in_channels, int_channels, out_channels, stride, bias)
        elif block_type == "SeparableConv2d":
            self.res_block = Quant_SeparableConv2d_brevitas(in_channels, out_channels, SP_kernel_size, padding, halve_resolution, bias, SP_replicas)
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
            self.se = Quant_SqueezeAndExcite_brevitas(out_channels, SE_ratio, bias)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.res_block(x)
        res = self.bn_up(self.up(x)) if self.upscale else x
        res = self.pool(res) if (self.halve_resolution and not self.upscale) else res
        ###collegamento epr squeeze and excite..
        res = self.se(res) if self.squeeze_and_excite else res
        out = self.act(out + res)
        return out

class Quant_Residual131_brevitas(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, stride=(1, 1), bias=True):
        super(Quant_Residual131_brevitas, self).__init__()
        
        #NICOLA : spostare i seguenti parametri su classi superiori
        self.weigh_bit = Int8WeightPerTensorFloat 
        self.act_bit = Uint8ActPerTensorFloat
        self.bias_bit = Int16Bias
        self.quant_method = 'scale'
        self.alpha_coeff = 123.0 # controlla alpha value

    
        self.l1 = qnn.QuantConv2d(in_channels, int_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias, 
                                act_bit=self.act_bit, weight_bit=self.weigh_bit, bias_quant=self.bias_bit, 
                                quantization = self.quantization, quant_method = self.quant_method, return_quant_tensor=True)
        self.bn1 = qnn.BatchNorm2dToQuantScaleBias(num_features=int_channels, bias_quant=self.bias_bit, 
                                weight_quant=self.weigh_bit, output_quant=self.act_bit, return_quant_tensor=True, eps=1e-3)
        self.act1 = qnn.QuantReLU(alpha=self.alpha_coeff)
        self.l2 = qnn.QuantConv2d(int_channels, int_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=bias, 
                                act_bit=self.act_bit, weight_bit=self.weigh_bit,bias_quant=self.bias_bit, 
                                quantization= self.quantization, quant_method = self.quant_method, output_quant=self.act_bit, 
                                return_quant_tensor=True)
        self.bn2 = qnn.BatchNorm2dToQuantScaleBias(num_features=int_channels, bias_quant=self.bias_bit, 
                                weight_quant=self.weigh_bit,  output_quant=self.act_bit, return_quant_tensor=True, eps=1e-3)
        self.act2 = qnn.QuantReLU(alpha=self.alpha_coeff)
        self.l3 = qnn.QuantConv2d(int_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias, 
                                act_bit=self.act_bit, weight_bit=self.weigh_bit, bias_quant=self.bias_bit, 
                                quantization= self.quantization, quant_method=self.quant_method,output_quant=self.act_bit, 
                                return_quant_tensor=True)
        self.bn3 = qnn.BatchNorm2dToQuantScaleBias(num_features=out_channels, bias_quant=self.bias_bit, 
                                weight_quant=self.weigh_bit, output_quant=self.act_bit, return_quant_tensor=True, eps=1e-3)

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
    def __init__(self, in_channels, out_channels, stride, bias=True):
        super(Quant_Residual33_brevitas, self).__init__()

        #NICOLA : spostare i seguenti parametri su classi superiori
        self.weigh_bit = Int8WeightPerTensorFloat 
        self.act_bit = Uint8ActPerTensorFloat
        self.bias_bit = Int16Bias
        self.quant_method = 'scale'
        self.alpha_coeff = 123.0 # controlla alpha value     

   
        self.l1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride, bias=bias, 
                                act_bit=self.act_bit, weight_bit=self.weigh_bit, bias_quant=self.bias_bit, 
                                quantization = self.quantization, quant_method = self.quant_method, output_quant=self.act_bit, 
                                return_quant_tensor=True)
        self.bn1 = qnn.BatchNorm2dToQuantScaleBias(num_features=out_channels, bias_quant=self.bias_bit, 
                                weight_quant=self.weigh_bit, output_quant=self.act_bit, return_quant_tensor=True, eps=1e-3)
        self.act1 = qnn.QuantReLU(alpha=self.alpha_coeff)
        self.l2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=bias, 
                                act_bit=self.act_bit, weight_bit=self.weigh_bit, bias_quant=self.bias_bit, 
                                quantization = self.quantization, quant_method = self.quant_method, output_quant=self.act_bit,
                                return_quant_tensor=True)
        self.bn2 = qnn.BatchNorm2dToQuantScaleBias(num_features=out_channels, bias_quant=self.bias_bit, 
                                weight_quant=self.weigh_bit, output_quant=self.act_bit, return_quant_tensor=True, eps=1e-3)

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
        self.fc1 = qnn.QuantLinear(channels, int(float(channels)/float(ratio)), bias=bias, weight_quant=self.weigh_bit, bias_quant=self.bias_bit, 
                                    output_quant=self.act_bit, return_quant_tensor=True)
        self.relu = qnn.QuantReLU(alpha=self.alpha_coeff)
        self.fc2 = qnn.QuantLinear(int(float(channels)/float(ratio)), channels, bias=bias, weight_quant=self.weigh_bit, bias_quant=self.bias_bit, 
                                    output_quant=self.act_bit, return_quant_tensor=True)
        self.sigmoid = qnn.QuantSigmoid(act_quant=self.act_bit, input_quant=self.act_bit, return_quant_tensor=True)
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