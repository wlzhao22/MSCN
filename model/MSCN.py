import torch
from torch import nn
import torch.nn.functional as F
import math
from model.RevIN import RevIN
from model.layers.layer import series_decomp, Flatten_Head
from .ours_memory_module import MemoryModule


class LayerNorm(nn.Module):


    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):

        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(
            x)

        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self,x,pad_length_left,pad_length_right,pad_values=0):

        D_out,D_in,ks=x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left)
            pad_right = torch.zeros(D_out,D_in,pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left,x],dims=-1)
        x = torch.cat([x,pad_right],dims=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class Block(nn.Module):
    def __init__(self, large_size, small_size, patchnum, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = nn.Conv1d(
            nvars*patchnum, nvars*patchnum, kernel_size=small_size,
            stride=1, padding=small_size//2, groups=nvars*patchnum, bias=True,dilation=1,
        )
        self.norm = nn.BatchNorm1d(patchnum)
        self.norm1 = nn.BatchNorm1d(nvars*patchnum)
        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * patchnum
                                 , out_channels=nvars * patchnum, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=patchnum
                                 )
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * patchnum, out_channels=nvars * patchnum, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=patchnum)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffnpw1 = nn.Conv1d(in_channels=nvars * patchnum, out_channels=nvars * patchnum, kernel_size=1, stride=1,
                                 padding=0 ,bias=True,groups=nvars)
        self.ffnact = nn.GELU()
        self.ffnpw2 = nn.Conv1d(in_channels=nvars * patchnum, out_channels=nvars * patchnum, kernel_size=1, stride=1,
                                 padding=0 ,bias=True,groups=nvars)

        
        self.ffndrop1 = nn.Dropout(drop)
        self.ffndrop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dff
    def forward(self,x):

        input = x
        B, M, N, D = x.shape
        x = x.reshape(B,M*N,D)
        x = self.dw(x)
        x = self.norm1(x)
        x = x.reshape(B,M,N,D)
        x = x.reshape(B*M,N,D)
        x = self.norm(x)
        x = x.reshape(B, M, N,D)
        x = x.reshape(B, M * N,D)

        x = self.ffndrop1(self.ffnpw1(x))
        x = self.ffnact(x)
        x = self.ffndrop2(self.ffnpw2(x))
        x = x.reshape(B, M, N, D)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N * M, D)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, N, M, D)
        x = x.permute(0, 2, 1, 3)
        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars, patchnum,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        # d_ffn = patchnum * nvars
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, patchnum=patchnum, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class MSCN(nn.Module):
    def __init__(self,patch_size,patch_stride, stem_ratio, downsample_ratio, ffn_ratio, num_blocks, large_size, small_size, dims, dw_dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True, affine=True,
                 subtract_last=False, freq=None, seq_len=512, c_in=7, individual=False, target_window=96, class_drop=0.,class_num = 10,n_memory=10,phase_type=None,device=None,dataset_name=None,shrink_thres=0.05,memory_init_embedding=None,memory_initial = False,momentum=0.9):

        super(MSCN, self).__init__()
        self.class_drop = class_drop
        self.class_num = class_num
        self.memory_initial  = memory_initial
        self.seq_len = seq_len

        # RevIN
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine)

        # stem layer & down sampling layers
        self.downsample_layer = nn.Linear(patch_size,dims[0])
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # backbone
        patch_num = seq_len // patch_stride
        self.num_stage = len(num_blocks)
        self.stage = Stage(ffn_ratio, num_blocks[0], large_size[0], small_size[0], dmodel=dims[0],
                          dw_model=dw_dims[0], nvars=nvars, patchnum = patch_num, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
        # head
        self.n_vars = c_in
        self.individual = individual

        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
        else:
            if patch_num % pow(downsample_ratio,(self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio,(self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1))+1)


            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
        # self.head_moe = nn.Parameter(d_model, d_model)
        # self.head_dection1 = nn.Linear(d_model*2, self.patch_size)
        
        self.head_dection1 = nn.Linear(d_model * 2, self.patch_size)
        self.head_dropout = nn.Dropout(head_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model*patch_num, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name,momentum=momentum)
        self.mem_norm = nn.LayerNorm(n_memory)
        # if phase_type == 'test':
        #     self.downsample_layer.requires_grad_ = False
        #     self.stage.blocks[0].requires_grad_ = False
        #     self.head_dection1.requires_grad_ = False
            # self.stage.blocks[0].ffnpw2.requires_grad_ = False



    def forward_feature(self, x, te=None):

        B,M,L=x.shape
        x = x.unsqueeze(-2)
        B, M, D, N = x.shape
        x = x.reshape(B * M, D, N)
        if self.patch_size != self.patch_stride:
            # stem layer padding
            pad_len = self.patch_size - self.patch_stride
            pad = x[:,:,-1:].repeat(1,1,pad_len)
            x = torch.cat([x,pad],dim=-1)
        x = x.reshape(B,M,1,-1).squeeze(-2)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        x = self.downsample_layer(x)
        x = self.stage(x)
        return x

    def detection(self, x, update_flag):

        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        x = self.forward_feature(x, te=None)
        x = self.norm(x)
        b, v, n ,l = x.shape
        k_x = x.reshape(b, v, -1)
        outputs = self.mem_module(k_x, update_flag)
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']
        mem = self.mem_module.mem
        out = self.norm(out.reshape(b, v, 2, n, l)[:,:,1,:,:])         # 0: query 1: attn * memory

        x = torch.cat([x,out],dim=-1)
        x = self.head_dropout(self.head_dection1(x))
        B,M,_,_=x.shape
        x = x.reshape(B,M,-1)
        x = x[:,:,:self.seq_len]
        x = x.permute(0,2,1)

        if self.revin:
            x = self.revin_layer(x, 'denorm')
        if self.memory_initial:
            return {"out":out, "memory_item_embedding":None, "queries":k_x, "mem":mem}
        else:
            return {"out":x, "memory_item_embedding":memory_item_embedding, "queries":k_x, "mem":mem, "attn":attn}

    def forward(self, x, te=None, update_flag = False):

        x = self.detection(x, update_flag)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class Model(nn.Module):
    def __init__(self, configs, memory_initial = False ,memory_init_embedding=None,phase_type=None,dataset_name=None,shrink_thres=0.05):
        super(Model, self).__init__()
        # hyper param
        self.n_memory = configs.n_memory
        self.device = configs.device

        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.input_c
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.win_size
        self.c_in = self.nvars,
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.momentum = configs.momentum


        # decomp
        self.decomposition = configs.decomposition


        self.model = MSCN(patch_size=self.patch_size, patch_stride=self.patch_stride, stem_ratio=self.stem_ratio,
                           downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks,
                           large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                           nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                           backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                           use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                           subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in,
                           individual=self.individual, target_window=self.target_window,n_memory=self.n_memory, memory_initial = memory_initial,
                           shrink_thres=shrink_thres,memory_init_embedding=memory_init_embedding,phase_type=phase_type,dataset_name=dataset_name,momentum= self.momentum )

    def forward(self, x, update_flag = False, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        x = x.permute(0, 2, 1)
        te = None
        x = self.model(x, te ,update_flag)
        return x
