# -*- coding: utf-8 -*-
# @Time    : 2022/1/16 下午2:21
# @Author  : 小锋学长生活大爆炸
# @FileName: model_store_sxf.py
# @Software: PyCharm
# @Blog    : https://blog.csdn.net/sxf1061700625
import math
import cv2
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import spectral_norm
from car_hynet.utils.util import dist_th
import torch.nn.functional as F

# ---------------Common--------------------- #
eps_l2_norm = 1e-10

def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    # return (desc / torch.Tensor(desc).pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)).numpy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp = desc
    if not torch.is_tensor(desc):
        temp = torch.Tensor(desc).to(device)
    if temp.ndim == 1:
        temp = temp.unsqueeze(0)
    temp = temp / temp.pow(2).sum(dim=1, keepdim=True).add(1e-10).pow(0.5)
    if torch.is_tensor(temp):
        return temp
    return temp.cpu().numpy()

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        FRN layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        '''打印模块的自定义的额外信息，'''
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        # Compute the mean norm of activations per channel.
        # 计算每个通道的激活的平均值
        # FRN的操作是(H, W)维度上的，即对每个样例的每个channel单独进行归一化，
        # 这里 x 就是一个N维度（HxW）的向量，所以FRN没有BN层对batch依赖的问题。
        # BN层采用归一化方法是减去均值然后除以标准差，而FRN却不同，这里没有减去均值操作，
        # 公式中的 v^2是x 的二次范数的平均值。
        # 这种归一化方式类似BN可以用来消除中间操作（卷积和非线性激活）带来的尺度问题，有助于模型训练。
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        # 公式里的epsilon是一个很小的正常量，以防止除0。
        # FRN是在H,W两个维度上归一化，一般情况下网络的特征图大小N=HxW较大，
        # 但是有时候可能会出现1x1的情况，比如InceptionV3和VGG网络，此时epsilon就比较关键。
        # 当epsilon值较小时，归一化相当于一个符号函数（sign function），这时候梯度几乎为0，严重影响模型训练；
        # 当值较大时，曲线变得更圆滑，此时的梯度利于模型学习。
        # 对于这种情况，论文建议采用一个可学习的epsilon。
        # 对于不含有1x1特征的模型，论文中采用的是一个常量值1e-6。
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        # 归一化之后同样需要进行缩放和平移变换，这里的gamma和beta也是可学习的参数（参数大小为C）
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features):
        """
        TLU layer as in the paper
        FRN缺少去均值的操作，这可能使得归一化的结果任意地偏移0，
        如果FRN之后是ReLU激活层，可能产生很多0值，这对于模型训练和性能是不利的。
        为了解决这个问题，FRN之后采用的阈值化的ReLU，即TLU。
        这里的tau是一个可学习的参数。论文中发现FRN之后采用TLU对于提升性能是至关重要的。
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        #self.relu = nn.SiLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v: new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=None, keep_3x3=False):
        """
        x: 输入张量
        oup: 输出通道数
        stride: 步长
        expand_ratio: 扩张系数
        identity_tensor_multiplier: 区间在0-1的浮点数，用于部分通道残差连接，
                                    默认为1，即原始残差连接
        norm_layer: 若为False则不做BN，为True则过一层BN
        """
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier == 1.0 else True
        self.identity_tensor_channels = int(round(inp * identity_tensor_multiplier))
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        # dw
        if expand_ratio == 2 or inp == oup or keep_3x3:
            layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp, norm_layer=norm_layer))
        layers.append(CoordAtt(inp, inp))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                norm_layer(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, norm_layer=norm_layer),
        ])
        if expand_ratio == 2 or inp == oup or keep_3x3 or stride == 2:
            layers.extend([
                # dw-linear
                nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
                norm_layer(oup),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor = x[:, :self.identity_tensor_channels, :, :] + out[:, :self.identity_tensor_channels, :, :]
                out = torch.cat([identity_tensor, out[:, self.identity_tensor_channels:, :, :]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out

# ------------------------------------------- #

# ---------------BaseNet--------------------- #
class BaseOriSingle8Net(nn.Module):
    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        # WARNING: 我们需要.detach()输入，否则带有F.grid_sample的patches extractor
        # 产生的gradient是非常有噪声的，使得检测器训练完全不稳定。
        return (x - mean.detach()) / (std.detach() + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            # print(m.weights.sum())
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            x = layer(x)
        desc_raw = self.layer8(x).squeeze()
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc

class BaseOriMultiNet(nn.Module):
    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer2, self.layer3, self.layer4, self.layer4, self.layer5, self.layer6]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze()
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc

class BaseResNet(nn.Module):
    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        # WARNING: 我们需要.detach()输入，否则带有F.grid_sample的patches extractor
        # 产生的gradient是非常有噪声的，使得检测器训练完全不稳定。
        return (x - mean.detach()) / (std.detach() + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            # print(m.weights.sum())
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, x, mode='eval'):
        x1 = self.block1(self.input_norm(x))
        x2 = self.block2(x1)
        g1 = self.global_pool2(x2)
        g2 = self.global_pool1(x1)
        desc_raw = g1 + g2
        desc_raw = desc_raw.squeeze()
        desc = F.normalize(desc_raw, p=2, dim=1)
        # desc = desc_l2norm(desc_raw)
        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc
# ------------------------------------------- #

# ---------------Model Def------------------- #
class CAR_HyNet(nn.Module):
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(CAR_HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FRN(3, is_bias=is_bias_FRN),
            TLU(3),
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            CoordAtt(32, 32),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            CoordAtt(32, 32),
            TLU(32),
        )
        self.layer2_5 = SandGlass(32, 32, 1, 6)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer4_5 = SandGlass(64, 64, 1, 6)

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )


    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        # WARNING: 我们需要.detach()输入，否则带有F.grid_sample的patches extractor
        # 产生的gradient是非常有噪声的，使得检测器训练完全不稳定。
        return (x - mean.detach()) / (std.detach() + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, x, mode='eval'):
        #for layer in [self.layer1, self.layer2, self.layer2, self.layer3, self.layer4, self.layer4, self.layer5, self.layer6]:
        #    x = layer(x)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer2_5(x1)
        x3 = x1 + x2
        x = self.layer3(x3)
        x1 = self.layer4(x)
        x2 = self.layer4_5(x1)
        x3 = x1 + x2
        x = self.layer5(x3)
        x = self.layer6(x)

        desc_raw = self.layer7(x).squeeze()
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc

class HyNet(nn.Module):
    """HyNet model definition"""
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze()
        desc = desc_l2norm(desc_raw)
        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc


class L2Net(nn.Module):
    """L2Net model definition"""
    def __init__(self, is_bias=False, is_affine=False, dim_desc=128, drop_rate=0.3):
        super(L2Net, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.InstanceNorm2d(1, affine=is_affine),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

        return

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            x = layer(x)
        return desc_l2norm(x.squeeze())

class Loss_HyNet():
    def __init__(self, device, num_pt_per_batch, dim_desc, margin, alpha, is_sosr, knn_sos=8):
        self.device = device
        self.margin = margin
        self.alpha = alpha
        self.is_sosr = is_sosr
        self.num_pt_per_batch = num_pt_per_batch
        self.dim_desc = dim_desc
        self.knn_sos = knn_sos
        self.index_desc = torch.LongTensor(range(0, num_pt_per_batch))
        self.index_dim = torch.LongTensor(range(0, dim_desc))
        diagnal = torch.eye(num_pt_per_batch)
        self.mask_pos_pair = diagnal.eq(1).float().to(self.device)
        self.mask_neg_pair = diagnal.eq(0).float().to(self.device)

    def sort_distance(self):
        L = self.L.clone().detach()
        L = L + 2 * self.mask_pos_pair
        L = L + 2 * L.le(dist_th).float()
        R = self.R.clone().detach()
        R = R + 2 * self.mask_pos_pair
        R = R + 2 * R.le(dist_th).float()
        LR = self.LR.clone().detach()
        LR = LR + 2 * self.mask_pos_pair
        LR = LR + 2 * LR.le(dist_th).float()
        self.indice_L = torch.argsort(L, dim=1)
        self.indice_R = torch.argsort(R, dim=0)
        self.indice_LR = torch.argsort(LR, dim=1)
        self.indice_RL = torch.argsort(LR, dim=0)
        return

    def triplet_loss_hybrid(self):
        L = self.L
        R = self.R
        LR = self.LR
        indice_L = self.indice_L[:, 0]
        indice_R = self.indice_R[0, :]
        indice_LR = self.indice_LR[:, 0]
        indice_RL = self.indice_RL[0, :]
        index_desc = self.index_desc
        dist_pos = LR[self.mask_pos_pair.bool()]
        dist_neg_LL = L[index_desc, indice_L]
        dist_neg_RR = R[indice_R, index_desc]
        dist_neg_LR = LR[index_desc, indice_LR]
        dist_neg_RL = LR[indice_RL, index_desc]
        dist_neg = torch.cat((dist_neg_LL.unsqueeze(0),
                              dist_neg_RR.unsqueeze(0),
                              dist_neg_LR.unsqueeze(0),
                              dist_neg_RL.unsqueeze(0)), dim=0)
        dist_neg_hard, index_neg_hard = torch.sort(dist_neg, dim=0)
        dist_neg_hard = dist_neg_hard[0, :]
        # scipy.io.savemat('dist.mat', dict(dist_pos=dist_pos.cpu().detach().numpy(), dist_neg=dist_neg_hard.cpu().detach().numpy()))
        # 限定数据范围
        loss_triplet = torch.clamp(self.margin + (dist_pos + dist_pos.pow(2)/2*self.alpha) - (dist_neg_hard + dist_neg_hard.pow(2)/2*self.alpha), min=0.0)
        self.num_triplet_display = loss_triplet.gt(0).sum()
        self.loss = self.loss + loss_triplet.sum()
        self.dist_pos_display = dist_pos.detach().mean()
        self.dist_neg_display = dist_neg_hard.detach().mean()
        return

    def norm_loss_pos(self):
        diff_norm = self.norm_L - self.norm_R
        self.loss += diff_norm.pow(2).sum().mul(0.1)

    def sos_loss(self):
        L = self.L
        R = self.R
        knn = self.knn_sos
        indice_L = self.indice_L[:, 0:knn]
        indice_R = self.indice_R[0:knn, :]
        indice_LR = self.indice_LR[:, 0:knn]
        indice_RL = self.indice_RL[0:knn, :]
        index_desc = self.index_desc
        num_pt_per_batch = self.num_pt_per_batch
        index_row = index_desc.unsqueeze(1).expand(-1, knn)
        index_col = index_desc.unsqueeze(0).expand(knn, -1)
        A_L = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_R = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_LR = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_L[index_row, indice_L] = 1
        A_R[indice_R, index_col] = 1
        A_LR[index_row, indice_LR] = 1
        A_LR[indice_RL, index_col] = 1
        A_L = A_L + A_L.t()
        A_L = A_L.gt(0).float()
        A_R = A_R + A_R.t()
        A_R = A_R.gt(0).float()
        A_LR = A_LR + A_LR.t()
        A_LR = A_LR.gt(0).float()
        A = A_L + A_R + A_LR
        A = A.gt(0).float() * self.mask_neg_pair
        sturcture_dif = (L - R) * A
        self.loss = self.loss + sturcture_dif.pow(2).sum(dim=1).add(eps_sqrt).sqrt().sum()
        return

    def compute(self, desc_L, desc_R, desc_raw_L, desc_raw_R):
        # num_pt_current_batch = desc_L.size(0)
        # # 动态创建掩码
        # diagonal = torch.eye(num_pt_current_batch, device=self.device)
        # self.mask_pos_pair = diagonal.eq(1).float()
        # self.mask_neg_pair = diagonal.eq(0).float()
        # self.index_desc = torch.LongTensor(range(0, num_pt_current_batch))
        self.desc_L = desc_L
        self.desc_R = desc_R
        self.desc_raw_L = desc_raw_L
        self.desc_raw_R = desc_raw_R
        self.norm_L = self.desc_raw_L.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.norm_R = self.desc_raw_R.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.L = cal_l2_distance_matrix(desc_L, desc_L)
        self.R = cal_l2_distance_matrix(desc_R, desc_R)
        self.LR = cal_l2_distance_matrix(desc_L, desc_R)
        self.loss = torch.Tensor([0]).to(self.device)
        self.sort_distance()
        self.triplet_loss_hybrid()
        self.norm_loss_pos()
        if self.is_sosr: self.sos_loss()
        return self.loss, self.dist_pos_display, self.dist_neg_display
# ------------------------------------------- #
