import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple
import torch.fft
import random
from math import sqrt
from functools import partial, reduce
from operator import mul
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def GASF(data,window_size=40,method='summation'):
    # transformer = PiecewiseAggregateApproximation(window_size)
    num_samples,num_features, num_time_steps = data.shape
    num_time_steps=int(num_time_steps/window_size)
    gaf_images = np.zeros((num_samples, num_features, num_time_steps, num_time_steps))

    for i in range(num_samples):
        gasf = GramianAngularField(image_size=num_time_steps, method=method)
        gaf_images[i] = gasf.fit_transform(data[i].detach().numpy())
    gaf_images = Tensor(gaf_images)
    return gaf_images

class Generator(nn.Module):
    def __init__(self,img_shape):
        super(Generator, self).__init__()
        self.img_shape=img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(64, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.img_shape[-1]),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class Mlp(nn.Module):
    '''
    多层感知机模型
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Tanh, drop=0.):
        '''

        :param in_features: 输入特征数量
        :param hidden_features: 隐藏层数量
        :param out_features: 输出特征数量
        :param act_layer: 激活层
        :param drop: Dropout丢弃率
        '''
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    '''
    全滤波器模型
    '''
    def __init__(self, dim, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 uncertainty_model=0, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0,lr=0.0001):
        '''

        :param dim: 维度数量
        :param h: 高
        :param w: 宽
        :param mask_radio: 掩码参数
        :param mask_alpha: 掩码值
        :param noise_mode: 噪声参数
        :param uncertainty_model: 不确定性模型
        :param perturb_prob: 扰动概率
        :param uncertainty_factor: 不确定性因子
        :param noise_layer_flag: 噪声标志
        :param gauss_or_uniform: 置信水平
        '''
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02) #可学习的滤波器参数
        self.w = w
        self.h = h

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha
        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform
        #
        self.img_shape = (int(h * sqrt(mask_radio)), int(w * sqrt(mask_radio)), dim)
        self.generator=Generator(self.img_shape)
        self.discriminator=Discriminator(self.img_shape)
        self.criterion=torch.nn.BCELoss()
        self.criterion=self.criterion.to(device)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
    def _reparameterize(self, mu, std, epsilon_norm):
        '''
        这是一个辅助方法，用于对给定的均值mu和标准差std以及epsilon的标准化值进行重新参数化。在这段代码中，它用于生成beta和gamma以对频谱进行加噪声处理
        :param mu: 均值
        :param std: 标准差
        :param epsilon_norm: 标准化的输入
        :return:
        '''
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t
    def loss_calculation_backward(self,imgs,gamma):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(gamma.size(0), 1).fill_(0.0), requires_grad=False)

        self.optimizer_G.zero_grad()
        g_loss = self.criterion(self.discriminator(gamma), valid)
        g_loss.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.criterion(self.discriminator(imgs), valid)
        fake_loss = self.criterion(self.discriminator(gamma.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()
        return g_loss,d_loss
    def spectrum_noise(self, img_fft, ratio=1.0, noise_mode=1,
                       uncertainty_model=0, gauss_or_uniform=0):
        '''
        这个方法用于在频谱上添加噪声。它首先获取输入图像的维度信息，并对输入图像进行傅立叶变换。
        然后根据给定的参数，通过对图像的幅度或相位或两者同时添加噪声。噪声模型和不确定性模型的选择将决定具体添加噪声的方式，
        例如是对批次进行建模还是对通道进行建模。添加的噪声可以是高斯分布或均匀分布。最后，它对噪声添加后的频谱进行逆傅立叶变换，并返回处理后的图像。
        :param img_fft:输入的fft图像
        :param ratio:给定参数 中心裁剪
        :param noise_mode:噪声模式
        :param uncertainty_model: 1 为 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling
        :param gauss_or_uniform:高斯或均匀噪声
        :return:
        '''
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
        if random.random() > self.p:
            return img_fft
        batch_size, h, w, c = img_fft.shape

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        img_abs = torch.fft.fftshift(img_abs, dim=(1))

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = 0

        img_abs_ = img_abs.clone()
        if noise_mode != 0:
            if uncertainty_model != 0:
                if uncertainty_model == 1:
                    # batch level modeling
                    miu = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),
                                     keepdim=True)
                    var = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),
                                    keepdim=True)
                    sig = (var + self.eps).sqrt()  # Bx1x1xC

                    var_of_miu = torch.var(miu, dim=0, keepdim=True)
                    var_of_sig = torch.var(sig, dim=0, keepdim=True)
                    sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
                    sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

                    if gauss_or_uniform == 0:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)

                        miu_mean = miu
                        sig_mean = sig

                        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    elif gauss_or_uniform == 1:
                        epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1.  # U(-1,1)
                        epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
                        beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    else:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)

                    # adjust statistics for each sample
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = gamma * (
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] - miu) / sig + beta

                elif uncertainty_model == 2:
                    # element level modeling
                    miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
                                             keepdim=True)
                    var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
                                            keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC

                    if gauss_or_uniform == 0:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = \
                        img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] + gamma

                elif uncertainty_model == 3:
                    var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
                                            keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC
                    if gauss_or_uniform == 0:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor
                    gamma=self.generator(gamma)
                    self.loss_calculation_backward(img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :],gamma)
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] =\
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] + gamma

        img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.training:
            if self.noise_mode != 0 and self.noise_layer_flag == 1:
                if self.uncertainty_model !=3:
                    x = self.spectrum_noise(x,ratio=self.mask_radio, noise_mode=self.noise_mode,
                                            uncertainty_model=self.uncertainty_model,
                                            gauss_or_uniform=self.gauss_or_uniform)
        weight = torch.view_as_complex(self.complex_weight) #可学习的滤波器
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x

class Block(nn.Module):
    def __init__(self, dim,mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 uncertainty_model=0, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 gauss_or_uniform=0,lr=0.0001 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w,
                                   mask_radio=mask_radio,
                                   mask_alpha=mask_alpha,
                                   noise_mode=noise_mode,
                                   uncertainty_model=uncertainty_model, perturb_prob=perturb_prob,
                                   uncertainty_factor=uncertainty_factor, noise_layer_flag=1,
                                   gauss_or_uniform=gauss_or_uniform,lr=lr )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cnn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.uncertainty_model=uncertainty_model
    def forward(self, input):
        x = input
        x = x + self.drop_path(self.cnn(self.norm2(self.filter(self.norm1(x)))))
        # x = x + self.drop_path(self.mlp(self.filter(x)))
        # Drop_path: In residual architecture, drop the current block for randomly seleted samples
        return x




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        img_size = to_2tuple(img_size) #(224,224)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def init_weights(self, stop_grad_conv1=0):
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.embed_dim))
        nn.init.uniform_(self.proj.weight, -val, val)
        nn.init.zeros_(self.proj.bias)

        if stop_grad_conv1:
            self.proj.weight.requires_grad = False
            self.proj.bias.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # BxCxHxW -> BxNxC , N=(224/4)^2=3136, C=64

        return x

class ALOFT(nn.Module):
    def __init__(self,args):

        '''
        :param dim: 维度数量
        :param img_size: 图像大小
        :param patch_size: 补丁大小
        :param in_chans: 输入通道数
        :param embed_dim: 输出维度 w
        :param block_num: block的层数
        :param mlp_ratio: MLP隐藏层因数 dim*mlp_ratio
        :param drop: drop剪支因子 >0 就是这个因子，否则为一个随机数
        :param drop_path: 剪支数
        :param act_layer: 激活函数
        :param norm_layer: 标准化层
        :param h: 高
        :param w: 宽
        :param mask_radio: 掩码开始值 为1则是直接整个图像 中心裁剪值
        :param mask_alpha:无关紧要的一个参数
        :param noise_mode: 噪声模式设置  1 amplitude幅值; 2: phase相位 3:both"""
        :param uncertainty_model: 1 为 batch-wise modeling像素级 2: channel-wise modeling统计级 3:token-wise modeling
        :param perturb_prob: 添加噪声的概率 小于perturb_prob则不添加噪声
        :param uncertainty_factor: 辅助方法中的因数
        :param gauss_or_uniform: 高斯或均匀噪声 0为高斯噪声
        '''
        super(ALOFT, self).__init__()
        self.args=args
        self.embed=PatchEmbed(img_size=args.img_size, patch_size=args.patch_size, in_chans=args.in_chans, embed_dim=args.embed_dim)
        self.img_shape=(int(args.h * sqrt(args.mask_radio)),int(args.w * sqrt(args.mask_radio)),args.embed_dim)
        self.blocklayer=nn.Sequential(*[Block(args.embed_dim, mlp_ratio=args.mlp_ratio, drop=args.drop, drop_path=args.drop_path, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=args.h, w=args.w,
                 mask_radio=args.mask_radio, mask_alpha=args.mask_alpha,
                 noise_mode=args.noise_mode,
                 uncertainty_model=args.uncertainty_model, perturb_prob=args.perturb_prob,
                 uncertainty_factor=args.uncertainty_factor,
                 gauss_or_uniform=args.gauss_or_uniform,lr=args.learning_rate ) for _ in range(args.block_num)],
                                      nn.Flatten())
        self.classfier=nn.Sequential(
                                     nn.Linear(self.args.embed_dim*self.args.h**2,self.args.h*args.embed_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.args.h*self.args.embed_dim,self.args.embed_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.args.embed_dim,1),)


    def forward(self,x):
        x = GASF(x.cpu(),window_size=self.args.window_size)
        return self.classfier(self.blocklayer(self.embed(x)))

