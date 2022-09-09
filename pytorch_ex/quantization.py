import torch
from torch.nn import functional as F
from torch import nn
from torch.autograd import Function


def scale(in_tensor, max_range):
    t_max = torch.max(torch.abs(torch.min(in_tensor)), torch.abs(torch.max(in_tensor))).item()
    if t_max == 0.0:
        t_max = 1.0
    scaling_factor = max_range / t_max
    return scaling_factor, 0


def affine(in_tensor, max_range):
    t_max = torch.max(torch.abs(torch.min(in_tensor)), torch.abs(torch.max(in_tensor))).item()
    if t_max == 0.0:
        t_max = 1.0
    scaling_factor = max_range / t_max
    if torch.min(in_tensor) == 0.0:
        zero_point = max_range / 2.0
    else:
        zero_point = - torch.round(torch.min(in_tensor) * scaling_factor) - max_range
    return scaling_factor, zero_point


def scale_quantization(real_tensor, min_v, max_v):
    """
    Check the integer quantization survey to see if scale quantization is better for your use case, if you are using
    any normalization layer, scale quantization might be faster and more accurate https://arxiv.org/pdf/2004.09602.pdf
    @param real_tensor: input tensor in FP32 or any other format
    @param min_v, max_v: minimum and maximum values for quantized range
    @return: quantized input tensor with scale quantization, real zero is the same as quantized zero
    """
    scale_value, zero_value = scale(real_tensor, max_v)
    return torch.clamp(torch.round(scale_value * real_tensor), min=min_v, max=max_v) / scale_value


def affine_quantization(real_tensor, min_v, max_v):
    """
    Check the survey to see if affine quantization is better for your use case https://arxiv.org/pdf/2004.09602.pdf
    @param real_tensor: input tensor in FP32 or any other format
    @param min_v, max_v: minimum and maximum values for quantized range
    @return: quantized input tensor with affine quantization, quantized zero may be shifted with respect to 0 value
    """
    scale_value, zero = affine(real_tensor, max_v)
    return (torch.clamp(torch.round(scale_value * real_tensor + zero), min=min_v, max=max_v) - zero) / scale_value


class ParametrizedActivationClipping(Function):
    """
    Perform parametrized activation clipping as described in PACT https://arxiv.org/pdf/1805.06085.pdf
    Read the paper to understand what is a good initial value for alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha, scale, min_v, max_v):
        ctx.save_for_backward(x, alpha)
        y = torch.clamp(x, min=0, max=alpha.item())
        scale_alpha = scale / alpha
        y_q = torch.clamp(torch.round(y * scale_alpha), min=min_v, max=max_v) / scale_alpha
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha, = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        x_range = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None, None, None


class DoReFaWeightQuantization(Function):
    """
    DoReFa weight and bias quantization strategy described in https://arxiv.org/pdf/1606.06160.pdf
    """

    @staticmethod
    def forward(ctx, real_tensor, min_v, max_v, quant_method):
        ctx.save_for_backward(real_tensor)
        tanh = torch.tanh(real_tensor).float()
        float_quant_tensor = 2 * quantization_method[quant_method](
            tanh / (2 * torch.max(torch.abs(tanh)).detach()) + 0.5, min_v, max_v) - 1
        return float_quant_tensor

    @staticmethod
    def backward(ctx, grad_out_tensor):
        return grad_out_tensor, None, None, None

#ritornare qui anche zero_point e scaling_factor ?
def myDoReFaWeightQuantization(real_tensor, min_v, max_v, quant_method):
        tanh = torch.tanh(real_tensor).float()
        float_quant_tensor = 2 * quantization_method[quant_method](
            tanh / (2 * torch.max(torch.abs(tanh)).detach()) + 0.5, min_v, max_v) - 1
        return float_quant_tensor

class ReLu(nn.Module):
    def __init__(self, act_bit=16, alpha=10.0, quantization=False):
        super(ReLu, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        if quantization:
            self.relu = ParametrizedActivationClipping.apply
        else:
            self.alpha.requires_grad = False
            self.relu = nn.ReLU()
        self.act_bit = act_bit
        self.scale = 2 ** (self.act_bit - 1) - 1
        self.quantization = quantization
        self.min_v = - 2 ** (self.act_bit - 1) + 1
        self.max_v = 2 ** (self.act_bit - 1) - 1

    def forward(self, x):
        if self.quantization:
            out = self.relu(x, self.alpha, self.scale, self.min_v, self.max_v)
        else:
            out = self.relu(x)
        return out


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 act_bit=8, weight_bit=8, bias_bit=8, quantization=False, quant_method='scale'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, groups, dilation, bias)
        self.quantize = DoReFaWeightQuantization.apply
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit

        self.min_v_w = - 2 ** (self.weight_bit - 1) + 1
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.min_v_b = - 2 ** (self.bias_bit - 1) + 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1

        torch.nn.init.kaiming_normal_(self.weight)
        self.quantization = quantization
        self.quant_method = quant_method

    def forward(self, x):
        if self.quantization:
            wq = self.quantize(self.weight, self.min_v_w, self.max_v_w, self.quant_method)
            if self.bias is None:
                bq = None
            else:
                bq = self.quantize(self.bias, self.min_v_b, self.max_v_b, self.quant_method)
            y = F.conv2d(x, wq, bq, self.stride, self.padding, self.dilation, self.groups)
        else:
            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return y

### "folding-aware" quantized Conv2d layer..
class Conv2d_folded(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True,
                 act_bit=8, weight_bit=8, bias_bit=8, quantization=False, quant_method='scale'):
        super(Conv2d_folded, self).__init__(in_channels, out_channels, kernel_size, stride, padding, groups, dilation, bias)
        self.quantize = DoReFaWeightQuantization.apply
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.out_channels = out_channels

        self.batch_norm = nn.BatchNorm2d(self.out_channels)

        self.min_v_w = - 2 ** (self.weight_bit - 1) + 1
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.min_v_b = - 2 ** (self.bias_bit - 1) + 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1

        torch.nn.init.kaiming_normal_(self.weight)
        self.quantization = quantization
        self.quant_method = quant_method

    def forward(self, x):
        # fist apply convolution and internal batch normalization
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        y = self.batch_norm(y)
        
        # if it is quantized the Conv2d weight and bias are changed according to the current running 
        # mean and variance and quantized back, then the new output with quantized parameters is computed
        if self.quantization:
            bn_gamma = self.batch_norm.weight
            bn_beta = self.batch_norm.bias
            bn_mean = self.batch_norm.running_mean
            bn_var = self.batch_norm.running_var
            bn_eps = self.batch_norm.eps
            new_weight = self.weight * bn_gamma.reshape((self.out_channels,1,1,1)) / torch.sqrt(bn_var.reshape((self.out_channels,1,1,1)) + bn_eps)
            wq = self.quantize(new_weight, self.min_v_w, self.max_v_w, self.quant_method)
            if self.bias is not None:
                new_bias = ((self.bias - bn_mean) * bn_gamma / torch.sqrt(bn_var + bn_eps)) + bn_beta
                bq = self.quantize(new_bias, self.min_v_b, self.max_v_b, self.quant_method)
            else:
                new_bias = ((torch.zeros((self.out_channels)) - bn_mean) * bn_gamma / torch.sqrt(bn_var + bn_eps)) + bn_beta
                bq = self.quantize(new_bias, self.min_v_b, self.max_v_b, self.quant_method)
            y = F.conv2d(x, wq, bq, self.stride, self.padding, self.dilation, self.groups)

        return y


class Linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, act_bit=8, weight_bit=8, bias_bit=8,
                 quantization=False, quant_method='scale'):
        super(Linear, self).__init__(in_channels, out_channels, bias)
        self.quantize = DoReFaWeightQuantization.apply
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit

        self.min_v_w = - 2 ** (self.weight_bit - 1) + 1
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.min_v_b = - 2 ** (self.bias_bit - 1) + 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1

        torch.nn.init.kaiming_normal_(self.weight)
        self.quantization = quantization
        self.quant_method = quant_method

    def forward(self, x):
        if self.quantization:
            wq = self.quantize(self.weight, self.min_v_w, self.max_v_w, self.quant_method)
            if self.bias is None:
                bq = None
            else:
                bq = quantization_method[self.quant_method](self.bias, self.min_v_b, self.max_v_b)
            y = F.linear(x, wq, bq)
        else:
            y = F.linear(x, self.weight, self.bias)
        return y


scaling_method = {'scale': scale, 'affine': affine}
quantization_method = {'scale': scale_quantization, 'affine': affine_quantization}

