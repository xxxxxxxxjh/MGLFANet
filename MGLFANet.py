import types
import math
from torchvision.ops import DeformConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.io import savemat

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


######################################################################
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

###########  Network Modules ###############
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class UpsampleConvLayer(torch.nn.Module):
    """
    Transpose convolution layer to upsample the feature maps
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    """
    Residual convolutional block for feature enhancement in decoder
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class OverlapPatchEmbed(nn.Module):
    """
    Patch Embedding layer to donwsample spatial resolution before each stage.
    in_chans: number of channels of input features
    embed_dim: number of channels for output features
    patch_size: kernel size of the convolution
    stride: stride value of convolution
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return x, H, W

class LinearProj(nn.Module):
    """
    Linear projection used to reduce the number of channels and feature mixing.
    input_dim: number of channels of input features
    embed_dim: number of channels for output features
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.proj(x)
        return x

################## Encoder Modules ####################
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        #self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        #x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x



class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DeformableConv2d, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=True)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        return x


class GLFAM(nn.Module):
    """
    Graph-Enhanced Local Global Context Aggregation module with Deformable Convolution and Simplified Attention
    dim: number of channels of input
    heads: number of heads utilized in computing attention
    """

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dfconv = DeformableConv2d(dim // 2, dim // 2, kernel_size=3, padding=1)
        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * self.heads, 1, padding=0)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x1, x2 = torch.split(x, [C // 2, C // 2], dim=1)

        x1 = self.act(self.dfconv(x1))

        x2 = self.act(self.qkvl(x2))
        x2 = x2.reshape(B, self.heads, C // 4, H, W)

        query = x2[:, :-3, :, :, :].mean(dim=1)
        key = x2[:, -3, :, :, :]
        value = x2[:, -2, :, :, :]
        z1 = x2[:, -1, :, :, :]


        query = query.view(B, -1, H * W)
        key = key.view(B, -1, H * W)
        value = value.view(B, -1, H * W)

        attn = torch.bmm(query.transpose(1, 2), key) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)

        x2 = torch.bmm(value, attn.transpose(1, 2)).view(B, C // 4, H, W)


        x = torch.cat([x1, z1, x2], dim=1)

        return x


class EncoderBlock(nn.Module):
    """
    dim: number of channels of input features
    """
    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.attn = GLFAM(dim, heads=heads)
        
    def forward(self, x, H, W):
        B, C, H, W = x.shape

        inp_copy = x
              
        x = self.layer_norm1(inp_copy)
        x = self.attn(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        out = out + x
        return out

class DifferenceEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(DifferenceEnhancementModule, self).__init__()
        self.in_channels = in_channels
        # 特征差异计算层
        self.diff_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        # 差异增强注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 特征融合层
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=True)

        self.act = nn.GELU()

    def forward(self, inp_feats):
        # 特征提取
        f1 = inp_feats[0]
        f2 = inp_feats[1]
        # 计算差异特征
        diff = torch.abs(f1 - f2)
        diff = self.diff_conv(diff)
        # 差异增强
        enhanced_diff = self.attention(diff) * diff
        # 特征融合
        combined_features = torch.cat([f1, f2, enhanced_diff], dim=1)
        fused_features = self.fusion_conv(combined_features)
        # 应用GELU激活函数
        fused_features = self.act(fused_features)

        return fused_features


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


################## Encoder #########################
#Transormer Ecoder with x4, x8, x16, x32 scales
class Encoder(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], drop_path_rate=0., heads=[4, 4, 4, 4],
                 depths=[3, 3, 4, 3]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        self.embed_dims     = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        ############ Stage-1 (x1/4 scale)
        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        #cur = 0 

        self.block1 = nn.ModuleList()
        for i in range(depths[0]):
            self.block1.append(EncoderBlock(dim=embed_dims[0], mlp_ratio=mlp_ratios[0]))
        
        ############# Stage-2 (x1/8 scale)
        #cur += depths[0]

        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            self.block2.append(EncoderBlock(dim=embed_dims[1], mlp_ratio=mlp_ratios[1]))
       
       ############# Stage-3 (x1/16 scale)
        #cur += depths[1]
        
        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            self.block3.append(EncoderBlock(dim=embed_dims[2], mlp_ratio=mlp_ratios[2]))
        
        ############# Stage-4 (x1/32 scale)
        #cur += depths[2]

        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            self.block4.append(EncoderBlock(dim=embed_dims[3], mlp_ratio=mlp_ratios[3]))

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, x):
        B = x.shape[0]
        outs = []
    
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        outs.append(x1)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class Decoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, in_channels = [32, 64, 128, 256], embedding_dim=64, output_nc=2, align_corners=True):
        super(Decoder, self).__init__()
        
        #settings
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # Channel reduction of feature maps before merging
        self.linear_c4 = LinearProj(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = LinearProj(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = LinearProj(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = LinearProj(input_dim=c1_in_channels, embed_dim=self.embedding_dim)


        # linear fusion layer to combine mult-scale features of all stages
        self.linear_fuse = nn.Sequential(
           nn.Conv2d(in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim, kernel_size=1, padding=0, stride=1),
           nn.BatchNorm2d(self.embedding_dim)
        )

        self.diff_c1 = DifferenceEnhancementModule(in_channels=self.embedding_dim)
        self.diff_c2 = DifferenceEnhancementModule(in_channels=self.embedding_dim)
        self.diff_c3 = DifferenceEnhancementModule(in_channels=self.embedding_dim)
        self.diff_c4 = DifferenceEnhancementModule(in_channels=self.embedding_dim)

        # Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        # Final activation
        self.active             = nn.Sigmoid() 


    def forward(self, inputs1, inputs2):
        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = inputs1        # len=4, 1/4, 1/8, 1/16, 1/32
        c1_2, c2_2, c3_2, c4_2 = inputs2        # len=4, 1/4, 1/8, 1/16, 1/32

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1)
        _c4_2 = self.linear_c4(c4_2)
        _c4   = self.diff_c4([_c4_1, _c4_2])
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1)
        _c3_2 = self.linear_c3(c3_2)
        _c3   = self.diff_c3([_c3_1, _c3_2])
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1)
        _c2_2 = self.linear_c2(c2_2)
        _c2   = self.diff_c2([_c2_1, _c2_2])
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1)
        _c1_2 = self.linear_c1(c1_2)
        _c1   = self.diff_c1([_c1_1, _c1_2])

        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1],dim=1))

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        return outputs


class MGLFANet(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4],
                 enc_channels=[64, 96, 128, 256], decoder_softmax=False, dec_embed_dim=256):
        super(MGLFANet, self).__init__()

        self.embed_dims = enc_channels
        self.depths     = depths
        self.embedding_dim = dec_embed_dim
        self.drop_path_rate = 0.1 

        # shared encoder
        self.enc = Encoder(patch_size=7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                                         heads=heads, mlp_ratios=[4, 4, 4, 4], drop_path_rate=self.drop_path_rate, depths=self.depths)
        
        # decoder
        self.dec = Decoder(in_channels=self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                           align_corners=False)

    def forward(self, x1, x2):

        fx1, fx2 = [self.enc(x1), self.enc(x2)]
        change_map = self.dec(fx1, fx2)

        return change_map