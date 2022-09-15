"""

Before you execute this exercise, you should read the quantization.py script and check the papers mentioned in there
In this exercise, you should re-write the models for Fashion MNIST and CIFAR10 using only quantized operators.
If you used normalization layers, have a look at the following paper https://arxiv.org/pdf/1712.05877.pdf
In deployment scenarios, where your CNN model has to be executed on a resource constrained device, floating points units
are not feasible and usually not available. To avoid using FP32 arithmetic, you should fold the normalization layer
as described in the aforementioned paper, in order to fuse the FP32 operation into the quantized convolutional layer
during the training phase. Check this link for more information on the formulas used to fold the batch norm into the
convolution weight and bias. Notice that the convolution bias is included in this formula, but it is not useful as the
conv layer is followed by the normalization. Thus, set the bias term to zero.
https://nathanhubens.github.io/fasterai/bn_folding.html

You should also check the brevitas repository and understand the syntax, how to load and save models from pytorch and
export them to a FINN format. In order to export the model, load the weights and assign them to the same neural network
written using brevitas classes, then test again the model to check that you get the same accuracy as the one trained
using the custom classes inside quantization.py.

Save and load torch models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
Brevitas page: https://github.com/Xilinx/brevitas
Brevitas gitter: https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
FINN page: https://finn.readthedocs.io/en/latest/index.html

Assignment:
- Check the TODO highlighted in the code, answer the questions, read the online material linked in the comments
- Rewrite the two CNNs you used previously for fashion MNIST and CIFAR10 using the custom layers provided in
  quantization.py and retrain them. Try to minimize the model size as much as possible and maximize task accuracy.
- If you decided to include batch normalization layers, fold them in the convolution operation
- Rewrite the two CNNs using Brevitas classes, import the weights and validate the task accuracy.
- Export the fashion MNIST model to FINN to be used in exercise 5, use the following code to export the model and
  follow the instructions below if you encounter any errors while executing it.

    # if FINN_save:
    #     if dataset_type == 'FashionMNIST':
    #         in_tensor = (1, 1, 28, 28)
    #     elif dataset_type == 'CIFAR10':
    #         in_tensor = (1, 3, 32, 32)
    #     else:
    #         exit("invalid dataset")
    #
    #     '''
    #     Due to some bugs inside the onnx manager of brevitas, a small edit is necessary to export models
    #     trained on a GPU-accelerated environment.
    #     Copy and paste the following lines inside brevitas\export\onnx\manager.py at line 89:
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         input_t = torch.empty(input_shape, dtype=torch.float).to(device)
    #     instead of the following line:
    #         input_t = torch.empty(input_shape, dtype=torch.float)
    #     The two functions below are supposed to save the model in a FINN-friendly format, however, are both not well
    #     documented and might fail for various reasons. If this happens, you should use the gitter page linked on the
    #     Brevitas repo to find the solution to your specific problem.
    #     Therefore, save the same model twice and we will try them both with the FINN notebooks.
    #     '''
    #     FINNManager.export(model.to("cpu"), input_shape=in_tensor, export_path=model_directory + model_name + tag + 'finn.onnx')
    #     BrevitasONNXManager.export(model.cpu(), input_shape=in_tensor, export_path=model_directory + model_name + tag + 'brevitas.onnx')
    #


Rules:
- You are free to do everything that you want, as long as the final CNN model is deployed fully quantized.
- The goal is still to write a model that has the best tradeoff between accuracy, model parameters and model size.
- You cannot use any FP32 operation within your model, all inputs and outputs must be quantized
- You can use batch normalization, but must be quantized as well and folded into the convolution
"""

"""here is an example CNN written using the custom functions. The training loop is the same as the one you used
   previously, so you only need to work on the CNN class and the custom functions."""
import torch
from torch import nn
import quantization as cs
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8Bias, Int16Bias, Int32Bias, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat
from torch.autograd import Function
import torch.nn.functional as F
from brevitas.core.scaling import ScalingImplType
from quantized_layer_templates_pytorch import *

### CUSTOM LAYERS NETWORKS

# FashionMNIST models

class quant_custom_mini_resnet(nn.Module):
    def __init__(self, weight_bit, act_bit, bias_bit, quant_method):
        super(quant_custom_mini_resnet, self).__init__()
         # TODO: you can either use the same quantization for all layers and datatypes or use layer-wise mixed-precision
        #  i.e., each datatype of each layer has its own precision
        self.weight_bit = weight_bit  # TODO: select an appropriate quantization for the weights
        self.act_bit = act_bit  # TODO: select an appropriate quantization for the activations
        self.bias_bit = bias_bit  # TODO: select an appropriate quantization for the bias
        self.quant_method = quant_method  # TODO: use either  'scale' or  'affine' and justify it
        
        self.min_v_w = - 2 ** (self.weight_bit - 1) + 1
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.min_v_b = - 2 ** (self.bias_bit - 1) + 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1
        
        # TODO: what is the alpha_coeff used for? What is a good initial value? How does the value change during training?
        self.alpha_coeff = 10.0

        self.quantization = True  # set to True to enable quantization, set to False to train with FP32

        self.conv2D_1 = cs.Conv2d(1,32,kernel_size=(3,3), stride=(2,2), padding=1, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=False, quant_method=self.quant_method)
        self.conv2D_2 = cs.Conv2d(32,64,kernel_size=(3,3), stride=(2,2), padding=1, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.skip1 = nn.MaxPool2d(kernel_size=(5,5), stride=(4,4), padding=2)
        self.conv2D_3 = cs.Conv2d(64,16, kernel_size=(1,1), stride=(1,1), bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.conv2D_4 = cs.Conv2d(16,16,kernel_size=(3,3), stride=(2,2), padding=1, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.conv2D_5 = cs.Conv2d(16,64,kernel_size=(1,1), stride=(1,1), bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.skip2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        ### different ReLU for different aplha
        self.act_1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_2 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_3 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_4 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_5 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.flatten = nn.Flatten()
        self.linear = cs.Linear(256, 10, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=False, quant_method=self.quant_method)
        
        nn.init.kaiming_normal_(self.conv2D_1.weight)
        nn.init.kaiming_normal_(self.conv2D_2.weight)
        nn.init.kaiming_normal_(self.conv2D_3.weight)
        nn.init.kaiming_normal_(self.conv2D_4.weight)
        nn.init.kaiming_normal_(self.conv2D_5.weight)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        if self.quantization:
            x = cs.quantization_method[self.quant_method](x, -2 ** (self.act_bit-1) + 1, 2 ** (self.act_bit-1) - 1)
        x_skip1 = self.skip1(x)
        out = self.conv2D_1(x)
        out = self.act_1(out)
        out = self.conv2D_2(out)
        out = out + x_skip1
        out = self.act_2(out)
        #bottleneck res connection (3 layer)
        x_skip2 = self.skip2(out)
        out = self.conv2D_3(out)
        out = self.act_3(out)
        out = self.conv2D_4(out)
        out = self.act_4(out)
        out = self.conv2D_5(out)
        out = out + x_skip2
        out = self.act_5(out)
        #downsample
        out = self.avgpool(out)
        #flatten + fully connected (1 layer)
        out = self.flatten(out)
        out = self.linear(out)
        return out

# folded versions still have BN weights, modify the state_dict updating Conv2d weights last time
# and load it in the same network topology w/o folded layers
class quant_custom_mini_resnet_folded(nn.Module):
    def __init__(self, weight_bit, act_bit, bias_bit, quant_method, quantization=True):
        super(quant_custom_mini_resnet_folded, self).__init__()
         # TODO: you can either use the same quantization for all layers and datatypes or use layer-wise mixed-precision
        #  i.e., each datatype of each layer has its own precision
        self.weight_bit = weight_bit  # TODO: select an appropriate quantization for the weights
        self.act_bit = act_bit  # TODO: select an appropriate quantization for the activations
        self.bias_bit = bias_bit  # TODO: select an appropriate quantization for the bias
        self.quant_method = quant_method  # TODO: use either  'scale' or  'affine' and justify it
        
        self.min_v_w = - 2 ** (self.weight_bit - 1) + 1
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.min_v_b = - 2 ** (self.bias_bit - 1) + 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1
        
        # TODO: what is the alpha_coeff used for? What is a good initial value? How does the value change during training?
        self.alpha_coeff = 50.0

        self.quantization = quantization  # set to True to enable quantization, set to False to train with FP32

        self.conv2D_1 = cs.Conv2d_folded(1,32,kernel_size=(3,3), stride=(2,2), padding=1, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=False, quant_method=self.quant_method)
        self.conv2D_2 = cs.Conv2d_folded(32,64,kernel_size=(3,3), stride=(2,2), padding=1, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.skip1 = nn.MaxPool2d(kernel_size=(5,5), stride=(4,4), padding=2)
        self.conv2D_3 = cs.Conv2d_folded(64,16, kernel_size=(1,1), stride=(1,1), bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.conv2D_4 = cs.Conv2d_folded(16,16,kernel_size=(3,3), stride=(2,2), padding=1, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.conv2D_5 = cs.Conv2d_folded(16,64,kernel_size=(1,1), stride=(1,1), bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)
        self.skip2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        ### different ReLU for different aplha
        self.act_1 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_2 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_3 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_4 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.act_5 = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=self.quantization)
        self.flatten = nn.Flatten()
        self.linear = cs.Linear(256, 10, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=False, quant_method=self.quant_method)
        
        nn.init.kaiming_normal_(self.conv2D_1.weight)
        nn.init.kaiming_normal_(self.conv2D_2.weight)
        nn.init.kaiming_normal_(self.conv2D_3.weight)
        nn.init.kaiming_normal_(self.conv2D_4.weight)
        nn.init.kaiming_normal_(self.conv2D_5.weight)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        if self.quantization:
            x = cs.quantization_method[self.quant_method](x, -2 ** (self.act_bit-1) + 1, 2 ** (self.act_bit-1) - 1)
        x_skip1 = self.skip1(x)
        out = self.conv2D_1(x)
        out = self.act_1(out)
        out = self.conv2D_2(out)
        out = out + x_skip1
        out = self.act_2(out)
        #bottleneck res connection (3 layer)
        x_skip2 = self.skip2(out)
        out = self.conv2D_3(out)
        out = self.act_3(out)
        out = self.conv2D_4(out)
        out = self.act_4(out)
        out = self.conv2D_5(out)
        out = out + x_skip2
        out = self.act_5(out)
        #downsample
        out = self.avgpool(out)
        #flatten + fully connected (1 layer)
        out = self.flatten(out)
        out = self.linear(out)
        return out

# CIFAR10 models

class quant_custom_ex3ResNet_medium(nn.Module):
    def __init__(self, weight_bit, act_bit, bias_bit, quant_method, quantization):
        super().__init__()

        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.alpha_coeff = 10.0

        self.min_v_w = - 2 ** (self.weight_bit - 1) + 1
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.min_v_b = - 2 ** (self.bias_bit - 1) + 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1

        ### 32 -> 32
        self.conv1 = cs.Conv2d(3, 16, kernel_size=3, padding=1, bias=True, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

        ### 32 -> 16
        self.res1_1 = Quant_ResidualBlock_custom(in_channels=16, out_channels=16, bias=True, block_type="Residual33", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res1_2 = Quant_ResidualBlock_custom(in_channels=16, out_channels=32, halve_resolution=True, bias=True, block_type="Residual33", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 16 -> 8
        self.res2_1 = Quant_ResidualBlock_custom(in_channels=32, int_channels=16, out_channels=32, bias=True, block_type="Residual131", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res2_2 = Quant_ResidualBlock_custom(in_channels=32, int_channels=16, out_channels=64, halve_resolution=True, bias=True, block_type="Residual131", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 8 -> 8
        self.res3_1 = Quant_ResidualBlock_custom(in_channels=64, int_channels=32, out_channels=64, bias=True, block_type="Residual131", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res3_2 = Quant_ResidualBlock_custom(in_channels=64, int_channels=32, out_channels=128, bias=True, block_type="Residual131", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 8 -> 4
        self.res4_1 = Quant_ResidualBlock_custom(in_channels=128, int_channels=64, out_channels=128, bias=True, block_type="Residual131", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res4_2 = Quant_ResidualBlock_custom(in_channels=128, int_channels=64, out_channels=256, halve_resolution=True, bias=True, block_type="Residual131", act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=True)
        self.flatten = nn.Flatten()
        self.linear = cs.Linear(256, 10, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.quantization, quant_method=self.quant_method)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        x = self.conv1(x)
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

class quant_custom_ex3ResNet_medium_folded(nn.Module):
    def __init__(self, weight_bit, act_bit, bias_bit, quant_method, quantization):
        super().__init__()

        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.quant_method = quant_method
        self.quantization = quantization
        self.first_last_quant = True
        self.alpha_coeff = 10.0

        self.min_v_w = - 2 ** (self.weight_bit - 1) + 1
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.min_v_b = - 2 ** (self.bias_bit - 1) + 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1

        ### 32 -> 32
        self.conv1 = cs.Conv2d_folded(3, 16, kernel_size=3, padding=1, bias=True, bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.first_last_quant, quant_method=self.quant_method)

        ### 32 -> 16
        self.res1_1 = Quant_ResidualBlock_custom_folded(in_channels=16, out_channels=16, bias=True, block_type="Residual33", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res1_2 = Quant_ResidualBlock_custom_folded(in_channels=16, out_channels=32, halve_resolution=True, bias=True, block_type="Residual33", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 16 -> 8
        self.res2_1 = Quant_ResidualBlock_custom_folded(in_channels=32, int_channels=16, out_channels=32, bias=True, block_type="Residual131", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res2_2 = Quant_ResidualBlock_custom_folded(in_channels=32, int_channels=16, out_channels=64, halve_resolution=True, bias=True, block_type="Residual131", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 8 -> 8
        self.res3_1 = Quant_ResidualBlock_custom_folded(in_channels=64, int_channels=32, out_channels=64, bias=True, block_type="Residual131", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res3_2 = Quant_ResidualBlock_custom_folded(in_channels=64, int_channels=32, out_channels=128, bias=True, block_type="Residual131", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 8 -> 4
        self.res4_1 = Quant_ResidualBlock_custom_folded(in_channels=128, int_channels=64, out_channels=128, bias=True, block_type="Residual131", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)
        self.res4_2 = Quant_ResidualBlock_custom_folded(in_channels=128, int_channels=64, out_channels=256, halve_resolution=True, bias=True, block_type="Residual131", bn_eps=1e-3, bn_momentum=0.01, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, alpha_coeff=self.alpha_coeff, quant_method=self.quant_method, quantization=self.quantization)

        ### 4 -> 1
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        
        self.relu = cs.ReLu(alpha=self.alpha_coeff, act_bit=self.act_bit, quantization=True)
        self.flatten = nn.Flatten()
        self.linear = cs.Linear(256, 10, act_bit=self.act_bit, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quantization=self.first_last_quant, quant_method=self.quant_method)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        x = self.conv1(x)
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


### BREVITAS LAYERS

# FashionMNIST models

class quant_brevitas_mini_resnet(nn.Module):
    def __init__(self):
        super(quant_brevitas_mini_resnet, self).__init__()
        
        self.alpha_coeff = 123.0  ### controllare alpha values...

        self.identity = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Uint8ActPerTensorFloat)

        self.conv2D_1 = qnn.QuantConv2d(1, 32, kernel_size=(3,3), stride=(2,2), padding=1, weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)
        self.conv2D_2 = qnn.QuantConv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1, weight_quant=Int8WeightPerTensorFloat, output_quant=Uint8ActPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)
        self.skip1 = qnn.QuantMaxPool2d(kernel_size=(5,5), stride=(4,4), padding=2, return_quant_tensor=True)
        self.quant_ID_1 = qnn.QuantIdentity(return_quant_tensor=True, act_quant=self.conv2D_2.output_quant)
        self.conv2D_3 = qnn.QuantConv2d(64, 16, kernel_size=(1,1), stride=(1,1), bias=True, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int16Bias, return_quant_tensor=True)
        self.conv2D_4 = qnn.QuantConv2d(16, 16, kernel_size=(3,3), stride=(2,2), padding=1, weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)
        self.conv2D_5 = qnn.QuantConv2d(16, 64, kernel_size=(1,1), stride=(1,1), bias=True, weight_quant=Int8WeightPerTensorFloat, output_quant=Uint8ActPerTensorFloat, bias_quant=Int16Bias, return_quant_tensor=True)
        self.skip2 = qnn.QuantMaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1, return_quant_tensor=True)
        self.quant_ID_2 = qnn.QuantIdentity(return_quant_tensor=True, act_quant=self.conv2D_5.output_quant)
        self.act_1 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.act_2 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.act_3 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.act_4 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.act_5 = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.avgpool = qnn.QuantAvgPool2d(kernel_size=(4,4), stride=(1,1), padding=0, return_quant_tensor=True)
        self.flatten = nn.Flatten()
        self.linear = qnn.QuantLinear(64,10, weight_quant=Int8WeightPerTensorFloat, act_quant=Uint8ActPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)

        nn.init.kaiming_normal_(self.conv2D_1.weight)
        nn.init.kaiming_normal_(self.conv2D_2.weight)
        nn.init.kaiming_normal_(self.conv2D_3.weight)
        nn.init.kaiming_normal_(self.conv2D_4.weight)
        nn.init.kaiming_normal_(self.conv2D_5.weight)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x):
        out = self.identity(x)
        x_skip1 = self.skip1(out)
        x_skip1 = self.quant_ID_1(x_skip1)
        out = self.conv2D_1(out)
        out = self.act_1(out)
        out = self.conv2D_2(out)
        out = out + x_skip1
        out = self.act_2(out)
        #bottleneck res connection (3 layer)
        x_skip2 = self.skip2(out)
        x_skip2 = self.quant_ID_2(x_skip2)
        out = self.conv2D_3(out)
        out = self.act_3(out)
        out = self.conv2D_4(out)
        out = self.act_4(out)
        out = self.conv2D_5(out)
        out = out + x_skip2
        out = self.act_5(out)
        #downsample
        out = self.avgpool(out)
        #flatten + fully connected (1 layer)
        out = self.flatten(out)
        out = self.linear(out)
        return out

# CIFAR10 models

class quant_brevitas_ex3ResNet_medium(nn.Module):
    def __init__(self):
        super(quant_brevitas_ex3ResNet_medium, self).__init__()

        self.alpha_coeff = 123.0 # controlla alpha value

        self.identity = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Uint8ActPerTensorFloat)
        ### 32 -> 32

        self.conv1 = qnn.QuantConv2d(3, 16, kernel_size=3, padding=1, bias=False, return_quant_tensor=True)
        self.bn1 = qnn.BatchNorm2dToQuantScaleBias(16, return_quant_tensor=True)#, input_quant=self.act_bit)

        ### 32 -> 16
        self.res1_1 = Quant_ResidualBlock_brevitas(in_channels=16, out_channels=16, bias=False, block_type="Residual33", alpha=self.alpha_coeff)
        self.res1_2 = Quant_ResidualBlock_brevitas(in_channels=16, out_channels=32, halve_resolution=True, bias=False, block_type="Residual33", alpha=self.alpha_coeff)

        ### 16 -> 8
        self.res2_1 = Quant_ResidualBlock_brevitas(in_channels=32, int_channels=16, out_channels=32, bias=False, block_type="Residual131", alpha=self.alpha_coeff)
        self.res2_2 = Quant_ResidualBlock_brevitas(in_channels=32, int_channels=16, out_channels=64, halve_resolution=True, bias=False, block_type="Residual131", alpha=self.alpha_coeff)

        ### 8 -> 8
        self.res3_1 = Quant_ResidualBlock_brevitas(in_channels=64, int_channels=32, out_channels=64, bias=False, block_type="Residual131", alpha=self.alpha_coeff)
        self.res3_2 = Quant_ResidualBlock_brevitas(in_channels=64, int_channels=32, out_channels=128, bias=False, block_type="Residual131", alpha=self.alpha_coeff)

        ### 8 -> 4
        self.res4_1 = Quant_ResidualBlock_brevitas(in_channels=128, int_channels=64, out_channels=128, bias=False, block_type="Residual131", alpha=self.alpha_coeff)
        self.res4_2 = Quant_ResidualBlock_brevitas(in_channels=128, int_channels=64, out_channels=256, halve_resolution=True, bias=False, block_type="Residual131", alpha=self.alpha_coeff)

        ### 4 -> 1
        self.avgpool = qnn.QuantAvgPool2d(kernel_size=4, stride=1, padding=0, return_quant_tensor=True)
        
        self.relu = PACT_QuantReLU(alpha=self.alpha_coeff, act_quant=Uint8ActPerTensorFloat)
        self.flatten = nn.Flatten()
        self.linear = qnn.QuantLinear(256, 10, weight_quant=Int8WeightPerTensorFloat, act_quant=Uint8ActPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)

    def forward(self, x):
        x = self.identity(x)
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

# FINN models (FashionMNIST)

class FINN_quant_brevitas_mini_resnet(nn.Module):
    def __init__(self):
        super(FINN_quant_brevitas_mini_resnet, self).__init__()

        self.identity = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Uint8ActPerTensorFloat)

        #28
        self.conv2D_1 = qnn.QuantConv2d(1, 16, kernel_size=(3,3), stride=(2,2), padding=1, weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)
        self.act_1 = qnn.QuantReLU(return_quant_tensot=True, act_quant=Uint8ActPerTensorFloat)
        #14
        self.conv2D_2 = qnn.QuantConv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=2, weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)
        self.act_2 = qnn.QuantReLU(return_quant_tensot=True, act_quant=Uint8ActPerTensorFloat)
        self.maxpool_1 = qnn.QuantMaxPool2d(kernel_size=(2,2), stride=(2,2), return_quant_tensor=True)
        #8
        self.conv2D_3 = qnn.QuantConv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=1, bias=True, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int16Bias, return_quant_tensor=True)
        self.act_3 = qnn.QuantReLU(return_quant_tensot=True, act_quant=Uint8ActPerTensorFloat)
        self.conv2D_4 = qnn.QuantConv2d(64, 64, kernel_size=(3,3), stride=(2,2), padding=1, bias=True, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int16Bias, return_quant_tensor=True)
        self.act_4 = qnn.QuantReLU(return_quant_tensot=True, act_quant=Uint8ActPerTensorFloat)
        #4
        self.maxpool_2 = qnn.QuantMaxPool2d(kernel_size=(2,2), stride=(2,2), return_quant_tensor=True)
        #2
        self.flatten = nn.Flatten()
        self.avg_approx = qnn.QuantLinear(256, 64, weight_quant=Int8WeightPerTensorFloat, act_quant=Uint8ActPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)
        self.act_5 = qnn.QuantReLU(return_quant_tensot=True, act_quant=Uint8ActPerTensorFloat)
        self.linear = qnn.QuantLinear(64, 10, weight_quant=Int8WeightPerTensorFloat, act_quant=Uint8ActPerTensorFloat, bias=True, bias_quant=Int16Bias, return_quant_tensor=True)

    def forward(self, x):
        out = self.identity(x)
        out = self.conv2D_1(out)
        print(out.shape)
        out = self.act_1(out)
        out = self.conv2D_2(out)
        out = self.act_2(out)
        out = self.maxpool_1(out)
        print(out.shape)
        out = self.conv2D_3(out)
        out = self.act_3(out)
        out = self.conv2D_4(out)
        print(out.shape)
        out = self.act_4(out)
        out = self.maxpool_2(out)
        print(out.shape)
        out = self.flatten(out)
        out = self.avg_approx(out)
        out = self.act_5(out)
        out = self.linear(out)
        return out