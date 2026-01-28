# -*- coding: utf-8 -*-
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


__all__ = ["resnet","MetaBasicBlock"]


def decom_conv(in_channels, out_channels, kernel_size=3, stride=1, bias=True):
    m = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias,
    )
    if kernel_size == 3:
        torch.nn.init.orthogonal_(m.weight)
    return m


class FactorizedConv(nn.Module):

    def __init__(self, in_channels, out_channels, n_basis,
                 stride=1, bias=False, de_conv=decom_conv):
        super(FactorizedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        modules = [nn.Conv2d(in_channels, n_basis, kernel_size=(1, 3),
            padding=(0, 1), stride=(1, stride), dilation=(1, 1), bias=bias,
        )]
        modules.append(nn.Conv2d(n_basis, out_channels, kernel_size=(3, 1),
            padding=(1, 0), stride=(stride, 1), dilation=(1, 1), bias=bias,
        ))
        self.conv = nn.Sequential(*modules)

    def recover(self):
        conv1 = self.conv[0] # (rank, inplanes, 1, 3)
        conv1.weight.data = conv1.weight.data.permute(1, 3, 2, 0)
        a, b, c, d = conv1.weight.shape
        dim1, dim2 = a * b, c * d
        VT = conv1.weight.data.reshape(dim1, dim2)
        conv2 = self.conv[1] # (outplanes, rank, 3, 1)
        conv2.weight.data = conv2.weight.data.permute(0, 2, 1, 3)
        a, b, c, d = conv2.weight.shape
        dim1, dim2 = a * b, c * d
        U = conv2.weight.data.reshape(dim1, dim2)
        W = torch.matmul(U, VT.T).reshape(self.out_channels, 3, self.in_channels, 3,).permute(0, 2, 1, 3)
        return W

    def frobenius_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]
        temp_VT = conv1.weight.permute(1, 3, 2, 0)
        a, b, c, d = temp_VT.data.shape
        dim1, dim2 = a * b, c * d
        VT = torch.reshape(temp_VT, (dim1, dim2))
        temp_UT = conv2.weight.permute(0, 2, 1, 3)
        a, b, c, d = temp_UT.data.shape
        dim1, dim2 = a * b, c * d
        U = torch.reshape(temp_UT, (dim1, dim2))
        loss = torch.norm(torch.matmul(U, torch.transpose(VT, 0, 1)), p='fro')**2
        return loss

    # def L2_loss(self):
    #     conv1 = self.conv[0]
    #     conv2 = self.conv[1]
    #     loss = torch.norm(conv1.weight, p='fro')**2 + torch.norm(conv2.weight, p='fro')**2
    #     return loss

    # 在 FactorizedConv 类中
    # def L2_loss(self):
    #     # 获取两个分解层的权重
    #     # conv[0] (V): shape [rank, in, 1, 3]
    #     # conv[1] (U): shape [out, rank, 3, 1]
    #     w_v = self.conv[0].weight 
    #     w_u = self.conv[1].weight
        
    #     # 1. 处理 V (右矩阵) -> 变为 [rank, In*3]
    #     # 展平: (rank, in, 1, 3) -> (rank, in * 3)
    #     mat_v = w_v.view(w_v.shape[0], -1) 
        
    #     # 2. 处理 U (左矩阵) -> 变为 [Out*3, rank]
    #     # 这里的变换必须是你 _decompose_layer 的逆过程
    #     # 原始生成: U_prime.view(out, 3, rank).permute(0, 2, 1).unsqueeze(-1)
    #     # 现在还原:
    #     # [out, rank, 3, 1] -> squeeze -> [out, rank, 3]
    #     # -> permute(0, 2, 1) -> [out, 3, rank]
    #     # -> view -> [out*3, rank]
    #     mat_u = w_u.squeeze(-1).permute(0, 2, 1).reshape(-1, w_u.shape[1])
        
    #     # 3. 虚拟矩阵乘法 (不改变模型结构，只计算值)
    #     # [Out*3, rank] @ [rank, In*3] -> [Out*3, In*3]
    #     w_rec = torch.matmul(mat_u, mat_v)
        
    #     # 4. 计算 L2 范数 (Frobenius Norm) 的平方
    #     loss = (w_rec ** 2).sum()
        
    #     return loss
    

    def L2_loss(self):

        # 1. Get weights

        # conv[0] (V): shape [rank, in, 1, 3]

        # conv[1] (U): shape [out, rank, 3, 1]

        w_v = self.conv[0].weight

        w_u = self.conv[1].weight



        # 2. Process V (Right Matrix) to match `recover` logic for VT

        # recover: conv1.weight.permute(1, 3, 2, 0) -> [in, 3, 1, rank]

        # recover: reshape(in*3, rank) -> VT is [in*3, rank]

        # recover: Uses VT.T -> [rank, in*3]

        

        # Mirroring 'recover' exactly:

        # [rank, in, 1, 3] -> [in, 3, 1, rank]

        v_permuted = w_v.permute(1, 3, 2, 0) 

        # [in, 3, 1, rank] -> [in*3, rank]

        vt_matrix = v_permuted.reshape(-1, w_v.shape[0]) 

        # Transpose to get V used in W = U @ V

        mat_v = vt_matrix.t() # Shape: [rank, in*3]



        # 3. Process U (Left Matrix) to match `recover` logic for U

        # recover: conv2.weight.permute(0, 2, 1, 3) -> [out, 3, rank, 1]

        # recover: reshape(out*3, rank) -> U is [out*3, rank]

        

        # Mirroring 'recover' exactly:

        # [out, rank, 3, 1] -> [out, 3, rank, 1]

        u_permuted = w_u.permute(0, 2, 1, 3)

        # [out, 3, rank, 1] -> [out*3, rank]

        mat_u = u_permuted.reshape(-1, w_u.shape[1]) # Shape: [out*3, rank]



        # 4. Virtual Matrix Multiplication (Reconstruction)

        # [out*3, rank] @ [rank, in*3] -> [out*3, in*3]

        w_rec = torch.matmul(mat_u, mat_v)



        # 5. Compute L2 Norm (Frobenius Norm) squared

        loss = (w_rec ** 2).sum()

        return loss

    def kronecker_loss(self):
        conv1 = self.conv[0]
        conv2 = self.conv[1]
        loss = (torch.norm(conv1.weight, p='fro')**2) * (torch.norm(conv2.weight, p='fro')**2)
        return loss



    def forward(self, x):
        return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def norm2d(group_norm_num_groups, planes,track_running_stats=True):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        group_nums = planes // group_norm_num_groups
        return nn.GroupNorm(group_nums, planes)
    else:
        return nn.BatchNorm2d(planes,track_running_stats=track_running_stats)


        # layers.append( #  block_fn 为两个3*3的卷积层的block块
        #     block_fn(
        #         in_planes=self.inplanes,
        #         out_planes=planes,
        #         stride=stride,
        #         downsample=downsample,
        #         group_norm_num_groups=group_norm_num_groups,
        #         track_running_stats=track_running_stats
        #     )

class MetaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, out_planes, stride=1, n_basis=1, downsample=None, group_norm_num_groups=None,track_running_stats=True,
                 dropout_rate=0, rate=1, track=None, cfg=None, ):
        super(MetaBasicBlock, self).__init__()


        planes = out_planes
        self.cfg = cfg
        self.inplanes = inplanes
        self.outplanes = planes
        self.rank = n_basis
        self.stride = stride

        # self.conv1 = DecomBlock(inplanes, planes, n_basis, stride=stride, bias=False)
        self.conv1 = FactorizedConv(inplanes, planes, n_basis, stride=stride, bias=False) # 分解卷积
        # self.bn1 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = DecomBlock(planes, planes, n_basis, stride=1, bias=False)
        self.conv2 = FactorizedConv(planes, planes, n_basis, stride=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes, momentum=0.0, track_running_stats=track)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)
        self.downsample = downsample

        self.dropout = nn.Dropout(p=dropout_rate)
        # if cfg['scale']: # 为了保证小模型的大模型对齐，可能需要对小模型地输出进行放大
        #     self.scaler = Scaler(rate)  # rate=1���S�RI�p
        # else:
        #     self.scaler = nn.Identity()
        self.scaler = nn.Identity()
    def _decompose_layer(self, layer, in_c, out_c, stride, rank):
            # 1. 获取原始权重 W (Out, In, 3, 3)
            if isinstance(layer, FactorizedConv):
                W_orig = layer.recover()
            else:
                W_orig = layer.weight.data

            # 2. 准备 SVD 矩阵
            # 我们需要将 spatial 维度分离并分别融合到 Channel 维度中
            # 原始: (Out, In, 3, 3)
            # 目标 SVD 输入: (Out*3, In*3)
            # 变换逻辑: 
            # (Out, In, 3, 3) -> permute(0, 2, 1, 3) -> (Out, 3, In, 3)
            # -> contiguous() -> view(Out*3, In*3)
            W_permuted = W_orig.permute(0, 2, 1, 3).contiguous()
            W_matrix = W_permuted.view(out_c * 3, in_c * 3)

            # 3. 执行 SVD 分解
            # U: (Out*3, Out*3), S: (min), V: (In*3, In*3)
            U, S, V = torch.svd(W_matrix)
            rank = min(rank, W_matrix.shape[0], W_matrix.shape[1])

            # 4. 截断与重构
            # 为了对称性，将 sqrt(S) 分别乘到 U 和 V 上
            sqrtS = torch.diag(torch.sqrt(S[:rank]))
            
            # U_prime: (Out*3, Rank) -> 对应 Conv2 (3x1卷积)
            U_prime = torch.matmul(U[:, :rank], sqrtS) 
            
            # V_prime: (Rank, In*3) -> 对应 Conv1 (1x3卷积)
            # 注意：torch.svd 返回的 V 是 V，重建公式是 U @ S @ V.T
            # 所以这里我们需要 (V @ sqrtS).T
            V_prime = torch.matmul(V[:, :rank], sqrtS).T 

            # 5. 创建新的 FactorizedConv 模块
            new_layer = FactorizedConv(in_c, out_c, rank, stride=stride, bias=False)

            # 6. 赋值权重
            
            # --- 赋值给第0层 (1x3 卷积) ---
            # 目标 shape: (Rank, In, 1, 3)
            # 来源 V_prime: (Rank, In*3)
            # 逻辑: 直接 view 分割 In 和 3，因为 W_matrix 构建时 In 在前 3 在后
            new_layer.conv[0].weight.data = V_prime.view(rank, in_c, 1, 3).contiguous()

            # --- 赋值给第1层 (3x1 卷积) ---
            # 目标 shape: (Out, Rank, 3, 1)
            # 来源 U_prime: (Out*3, Rank)
            # 逻辑: 
            # 1. view -> (Out, 3, Rank)
            # 2. permute(0, 2, 1) -> (Out, Rank, 3)
            # 3. unsqueeze(-1) -> (Out, Rank, 3, 1)
            new_layer.conv[1].weight.data = U_prime.view(out_c, 3, rank).permute(0, 2, 1).unsqueeze(-1).contiguous()

            return new_layer

    def decom(self, ratio_LR):
        # self.rank = round(ratio_LR * self.outplanes)
        # a, b, c, d = self.conv1.weight.shape  # (outplanes, inplanes, k, k)
        # dim1, dim2 = a * c, b * d
        # W = self.conv1.weight.data.reshape(dim1, dim2)
        # U, S, V = torch.svd(W)
        # sqrtS = torch.diag(torch.sqrt(S[:self.rank]))
        # new_U, new_V = torch.matmul(U[:, :self.rank], sqrtS), torch.matmul(V[:, :self.rank], sqrtS).T
        # self.conv1 = FactorizedConv(self.inplanes, self.outplanes, self.rank, stride=self.stride, bias=False)
        # self.conv1.conv[0].weight.data = new_V.reshape(self.inplanes, c, 1, self.rank).permute(3, 0, 2, 1)
        # self.conv1.conv[1].weight.data = new_U.reshape(self.outplanes, c, self.rank, 1).permute(0, 2, 1, 3)

        # a, b, c, d = self.conv2.weight.shape
        # dim1, dim2 = a * c, b * d
        # W = self.conv2.weight.data.reshape(dim1, dim2)
        # U, S, V = torch.svd(W)
        # sqrtS = torch.diag(torch.sqrt(S[:self.rank]))
        # new_U, new_V = torch.matmul(U[:, :self.rank], sqrtS), torch.matmul(V[:, :self.rank], sqrtS).T
        # self.conv2 = FactorizedConv(self.outplanes, self.outplanes, self.rank, stride=1, bias=False)
        # self.conv2.conv[0].weight.data = new_V.reshape(self.outplanes, c, 1, self.rank).permute(3, 0, 2, 1)
        # self.conv2.conv[1].weight.data = new_U.reshape(self.outplanes, c, self.rank, 1).permute(0, 2, 1, 3)
        # print("Done")


        self.rank = max(1, round(ratio_LR * self.outplanes))

        # 分解 conv1
        # 注意：需要传入正确的 stride
        self.conv1 = self._decompose_layer(
            self.conv1, 
            self.inplanes, 
            self.outplanes, 
            self.stride, 
            self.rank
        )

        # 分解 conv2
        # ResNet BasicBlock 中 conv2 的 stride 永远是 1，输入输出通道通常相等
        self.conv2 = self._decompose_layer(
            self.conv2, 
            self.outplanes, 
            self.outplanes, 
            1, 
            self.rank
        )

    def recover(self):
        W1 = self.conv1.recover()
        W2 = self.conv2.recover()
        self.conv1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.conv1.weight.data = W1
        self.conv2 = nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight.data = W2

    def frobenius_loss(self):
        loss1 = self.conv1.frobenius_loss()
        loss2 = self.conv2.frobenius_loss()
        return (loss1 + loss2)

    def kronecker_loss(self):
        loss1 = self.conv1.kronecker_loss()
        loss2 = self.conv2.kronecker_loss()
        return (loss1 + loss2)

    def L2_loss(self):
        loss = 0.0
        
        # 遍历两个主要的卷积层
        for layer in [self.conv1, self.conv2]:
            # 情况 A: 是分解层 (FactorizedConv)，调用它自定义的乘积 L2
            if hasattr(layer, 'L2_loss'):
                loss += layer.L2_loss()
            
            # 情况 B: 是普通层 (nn.Conv2d)，直接计算权重平方和
            elif isinstance(layer, nn.Conv2d):
                loss += (layer.weight ** 2).sum()
        
        return loss

    def forward(self, x):
        residual = x

        out = self.scaler(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.scaler(self.conv2(out))
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class HyperResNet(nn.Module): # resnet-18 可分为4个阶段，每个阶段有两个残差块 每个残差块包括两个3x3的卷积

    def __init__(self, data_shape, hidden_size, block, num_blocks, ratio_LR, decom_rule, num_classes=10,
                 rate=1, track=None, cfg=None, dropout_rate=0, group_norm_num_groups = None):
        super(HyperResNet, self).__init__()
        """
        decom_rule is a 2-tuple like (block_index, layer_index).
        For resnet18, block_index is selected from [0,1,2,3] and layer_index is selected from [0,1].
        Example: If we only want to decompose layers starting form the 8-th layer for resnet18, 
                 then we set decom_rule = (1, 1);
                 If we want to decompose all layer(expept head and tail layer), we can set 
                 decom_rule = (-1, 0);
                 If we don't want to decompose any layer, we can set 
                 decom_rule = (4, 0).
        """
        self.cfg = cfg
        self.dataset_name = cfg.data
        self.decom_rule = decom_rule # [0,0] 表示从第零个阶段的第零个残差块开始分解
        self.group_norm_num_groups = group_norm_num_groups
        self.inplanes = hidden_size[0] # 动态记录当前卷积输出通道，初始化为64
        self.hidden_size = hidden_size # 四个阶段的输出通道数量
        self.num_blocks = num_blocks    # 每个resnet阶段的残差块数量
        self.dropout_rate = dropout_rate # 随机失活率 这里为0
        self.feature_num = hidden_size[-1] # 最终层的输入 ？？？
        self.class_num = num_classes        # 10
        self.ratio_LR = ratio_LR            # 分解率

        self.head = nn.Sequential(
            nn.Conv2d(data_shape[0], self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64, momentum=None, track_running_stats=None),
            norm2d(group_norm_num_groups, planes=self.inplanes, track_running_stats=False),
            
        ) # 卷积层，输出对应上初始输出通道，这个应该初始层
        self.relu = nn.ReLU(inplace=True) # 一种内存优化手段

        # initialization the hybrid model
        strides = [1, 2, 2, 2]
        all_layers, common_layers, personalized_layers = [], [], []
        common_layers.append(self.head)
        for block_idx in range(len(hidden_size)):
            if block_idx < self.decom_rule[0]: # 看该阶段（块）是否需要分解
                layer = self._make_larger_layer(block=block, planes=hidden_size[block_idx],
                                                blocks=num_blocks[block_idx],
                                                stride=strides[block_idx], rate=rate, track=track)
                all_layers.append(layer)
                common_layers.append(layer)
            elif block_idx == self.decom_rule[0]: # 这个就是混合阶段（可能会第一块分解，第二块不分解，所以单独处理）
                config = round(hidden_size[block_idx] * self.ratio_LR)  # rank 第一块输出为64 ， 64*0.2=13
                layer = self._make_hybrid_layer(large_block=block, meta_block=MetaBasicBlock,
                                                planes=hidden_size[block_idx],
                                                blocks=num_blocks[block_idx], stride=strides[block_idx],
                                                start_decom_idx=self.decom_rule[1], config=config,
                                                rate=rate, track=track, )
                all_layers.append(layer)
                for layer_idx in range(self.decom_rule[1]): # 未分解的层视为公共层
                    common_layers.append(layer[layer_idx])
                for layer_idx in range(self.decom_rule[1], self.num_blocks[block_idx]): # 分解的层视为个性化层
                    personalized_layers.append(layer[layer_idx])

            elif block_idx > self.decom_rule[0]:
                config = round(hidden_size[block_idx] * self.ratio_LR)  # rank
                layer = self._make_meta_layer(block=MetaBasicBlock, planes=hidden_size[block_idx],
                                              blocks=num_blocks[block_idx],
                                              config=config, stride=strides[block_idx], rate=rate, track=track)
                all_layers.append(layer)
                personalized_layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 池化层，这里的全局池化替代了展平操作，即每个通道只取一个
        self.tail = nn.Linear(self.feature_num, num_classes) # 全连接层输出
        personalized_layers.append(self.tail) # 这个也算个性化层，但是未进行分解

        self.body = nn.Sequential(*all_layers) # 将列表变为网络层
        self.common = nn.Sequential(*common_layers)
        self.personalized = nn.Sequential(*personalized_layers) # 共六层，第一个阶段被拆为了两个层，后面三个阶段未拆，线性层不分解

        self.use_align = False
        self.feature_align_dim = 0

        if cfg.output_dim is not None:
            self.feature_align_dim = cfg.output_dim

        self.block_channels=[]
        
        self.use_align = False
        self.feature_align_dim = 0
        
        # 1. 判断是否开启对齐 (仅依赖 cfg.output_dim)
        if self.cfg.output_dim is not None:
            self.use_align = True
            self.feature_align_dim = self.cfg.output_dim
            
            # 2. 为每个阶段构建一个 Aligner
            # self.hidden_size 是 [64, 128, 256, 512]，正好对应4个阶段的输出通道
            self.aligners = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(channels, self.feature_align_dim),
                    # nn.ReLU(inplace=True) # 建议：如果做特征对齐，通常最后一层不加ReLU，以便允许负值，或者加归一化
                ) for channels in self.hidden_size
            ])
            
            print(f"Alignment enabled. 4 Aligners created for channels: {self.hidden_size} -> {self.feature_align_dim}")

            # self.fusion_weights = nn.Parameter(torch.ones(self.total_blocks_num) / self.total_blocks_num)  # 这里权重设为可学习权重




        # initialization for the hybrid model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): 
                m.weight.data.fill_(1) 
                m.bias.data.zero_()

    def _make_meta_layer(self, block, planes, blocks, config=None, stride=1, rate=1, track=None):
        cfg = self.cfg
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
                norm2d(self.group_norm_num_groups, planes=planes * block.expansion, track_running_stats=False),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride=stride, n_basis=config, downsample=downsample, group_norm_num_groups=self.group_norm_num_groups,
                            dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, n_basis=config, dropout_rate=self.dropout_rate, rate=rate,group_norm_num_groups=self.group_norm_num_groups,
                      track=track, cfg=cfg))
        return nn.Sequential(*layers)

    def _make_hybrid_layer(self, large_block, meta_block, planes, blocks, stride=1, rate=1, start_decom_idx=0,
                           config=1, track=None):
        """
        :param start_decom_idx: range from [0, blocks-1]
        """
        cfg = self.cfg
        downsample = None
        block = meta_block if start_decom_idx == 0 else large_block # 看第一块是否为要分解的块

        if stride != 1 or self.inplanes != planes * block.expansion:  # 残差对齐，按stride下采样保证大小相同，用1x1卷积对齐通道数，这里暂时还没用上
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
                norm2d(self.group_norm_num_groups, planes=planes * block.expansion, track_running_stats=False)
            )
        layers = []
        if start_decom_idx == 0:
            block = meta_block # 这又写重叠了,其中meta_block是分解后的残差块（即两层卷积的分解）
            layers.append(
                block(self.inplanes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate,group_norm_num_groups=self.group_norm_num_groups,
                      n_basis=config, track=track, cfg=cfg)) # 这里面要不是分解地两层卷积层，要不是未分解地
        else:
            block = large_block
            layers.append(
                block(self.inplanes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate,group_norm_num_groups=self.group_norm_num_groups,
                      track=track, cfg=cfg))
        self.inplanes = planes * block.expansion # 保证维度一致

        for idx in range(1, blocks): # 就剩下一个块，有必要写循环吗
            block = large_block if idx < start_decom_idx else meta_block # 和下面代码功能重复
            if idx < start_decom_idx:
                block = large_block
                layers.append(
                    block(self.inplanes, planes, dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg,group_norm_num_groups=self.group_norm_num_groups,))
            else:
                block = meta_block
                layers.append(
                    block(self.inplanes, planes, dropout_rate=self.dropout_rate, rate=rate, n_basis=config, track=track,group_norm_num_groups=self.group_norm_num_groups,
                          cfg=cfg))

        return nn.Sequential(*layers)


    def _make_larger_layer(self, block, planes, blocks, stride=1, rate=1, track=None):
        cfg = self.cfg
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion, momentum=0.0, track_running_stats=track)
                norm2d(self.group_norm_num_groups, planes=planes * block.expansion, track_running_stats=False),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample=downsample, dropout_rate=self.dropout_rate, rate=rate, group_norm_num_groups=self.group_norm_num_groups,
                  track=track, cfg=cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dropout_rate=self.dropout_rate, rate=rate, track=track, cfg=cfg, group_norm_num_groups=self.group_norm_num_groups,))

        return nn.Sequential(*layers)


    def recover_large_layer(self, ):
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    block.recover()
                else:
                    for j in range(len(block)):
                        block[j].recover()


    def decom_large_layer(self, ratio_LR=0.2):
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    block.decom(ratio_LR=ratio_LR)
                else:
                    for j in range(len(block)):
                        block[j].decom(ratio_LR=ratio_LR)


    def frobenius_decay(self):
        loss = torch.tensor(0.).to(self.cfg['device'])
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    loss += block.frobenius_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].frobenius_loss()
        return loss


    def kronecker_decay(self):
        loss = torch.tensor(0.).to(self.cfg['device'])
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    loss += block.kronecker_loss()
                else:
                    for j in range(len(block)):
                        loss += block[j].kronecker_loss()
        return loss


    # def L2_decay(self):
    #     loss = torch.tensor(0.).to(self.cfg.device)
    #     length = len(self.personalized)
    #     for idx, block in enumerate(self.personalized):
    #         # the last part of self.personalized is linear layer which is not decomposed
    #         if idx < length - 1:
    #             if isinstance(block, MetaBasicBlock):
    #                 loss += block.L2_loss()
    #             else:
    #                 for j in range(len(block)):
    #                     loss += block[j].L2_loss()
    #     return loss

    def L2_decay(self):
        loss = torch.tensor(0.).to(self.cfg.device if self.cfg else 'cuda')

        # 定义一个内部函数来递归处理各种层
        def add_l2_loss(module):
            local_loss = torch.tensor(0.).to(loss.device)
            
            # 情况1: 如果模块有自定义的 L2_loss 方法 (例如 MetaBasicBlock)，优先使用它
            if hasattr(module, 'L2_loss'):
                local_loss += module.L2_loss()
                
            # 情况2: 如果是基础的带权层 (Conv2d, Linear)，且没有被自定义方法处理过
            # 注意: MetaBasicBlock 也有 Conv2d，但上面的 hasattr 会拦截它，防止重复计算
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                # 只对 weight 做衰减，通常不对 bias 做 L2 (这也是 PyTorch optim 的默认行为之一，虽然 strict WD 会包含)
                if module.weight.requires_grad:
                    local_loss += torch.sum(module.weight ** 2)
                    
            # 情况3: 如果是容器 (Sequential, ModuleList, 或普通的 BasicBlock)，递归处理子模块
            else:
                for child in module.children():
                    local_loss += add_l2_loss(child)
                    
            return local_loss

        # ------------------------------------------------------
        # 遍历模型的两个主要部分：Common (前半截) 和 Personalized (后半截)
        # ------------------------------------------------------
        
        # 1. 处理 Common 层 (包含 Head 和 前期全秩 Blocks)
        # self.common 是一个 nn.Sequential
        loss += add_l2_loss(self.common)

        # 2. 处理 Personalized 层 (包含 Meta Blocks, 后期 Blocks 和 Tail)
        # self.personalized 也是一个 nn.Sequential
        loss += add_l2_loss(self.personalized)

        return loss


    def cal_smallest_svdvals(self): # 计算奇异值最小值，论证论文中的理论部分，分解后的奇异值有下界
        """
        calculate the smallest singular value of each residual block.
        For example, if the model is resnet18, then there are 8 residual blocks.
        """
        smallest_svdvals = []
        length = len(self.personalized)
        for idx, block in enumerate(self.personalized):
            # the last part of self.personalized is linear layer which is not decomposed
            if idx < length - 1:
                if isinstance(block, MetaBasicBlock):
                    smallest_svdvals.append(block.cal_smallest_svdvals())
                else:
                    for j in range(len(block)):
                        smallest_svdvals.append(block[j].cal_smallest_svdvals())
        return smallest_svdvals


    def calculate_stage_anchor_loss(self, anchors):
        """
        计算内部存储的 4 个阶段特征与传入的 4 个锚点的 MSE 损失。
        
        Args:
            anchors (list or tensor): 4 个锚点。
        """
        # 检查是否有特征 (防止 use_align=False 时调用报错)
        if not self.use_align or len(self.stage_features) == 0:
            return torch.tensor([0., 0., 0., 0.], device=self.parameters().__next__().device)

        losses = []
        
        for i in range(4):
            # [关键修改 4] 直接从 self 读取
            feat = self.stage_features[i] 
            anchor = anchors[i]
            
            # 1. Feature 归一化
            feat_norm = F.normalize(feat, p=2, dim=1)
            
            # 2. Anchor 归一化 (处理 tensor 类型)
            if isinstance(anchor, torch.Tensor):
                anchor = anchor.to(feat.device) 
                anchor_norm = F.normalize(anchor, p=2, dim=-1)
            else:
                # 如果 anchor 已经是 list 里的 tensor 且已归一化，直接用
                anchor_norm = anchor

            # 3. 计算 MSE
            loss = F.mse_loss(feat_norm, anchor_norm.detach())
            losses.append(loss)
            
        return torch.stack(losses)

    def forward(self, x):
        x = self.head(x)
        x = self.relu(x)
        self.stage_features = []
        # for idx, layer in enumerate(self.body):
        #     x = layer(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # feature = x
        # x = self.tail(x)
        # return feature,x

        for i, stage_layer in enumerate(self.body):
            x = stage_layer(x)
            
            # [关键修改 2] 如果开启对齐，计算特征并存入 self.stage_features
            if self.use_align:
                # x: [B, C, H, W] -> aligner -> [B, align_dim]
                aligned_feat = self.aligners[i](x)
                self.stage_features.append(aligned_feat)

        # 最后的分类头
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # pooled_feature = x # 如果外面需要用到池化后的特征，可以留着
        logits = self.tail(x)

        # [关键修改 3] 只返回 logits (或者 x, logits)，不再返回 feature 列表
        # 这样保持了接口的简洁性，外部调用不需要解包那么长
        return x, logits
        








class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats=True,
        dropout_rate = 0,
        rate=1, 
        track=None,
        cfg=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)

        self.downsample = downsample
        self.stride = stride

    def L2_loss(self):
        loss = 0.0
        
        loss += (self.conv1.weight ** 2).sum()
        
        loss += (self.conv2.weight ** 2).sum()
        
        if self.downsample is not None:
            for m in self.downsample:
                if isinstance(m, nn.Conv2d):
                    loss += (m.weight ** 2).sum()
                    
        return loss

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
        track_running_stats=True
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes,track_running_stats=track_running_stats)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes * 4,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * 4,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def decide_num_classes(dataset):
    if dataset == "cifar10" or dataset == "svhn":
        return 10
    elif dataset == "cifar100":
        return 100
    elif "tiny" in dataset:
        return 200
    elif "imagenet" in dataset:
        return 1000
    elif "femnist" == dataset:
        return 62

class ResNetBase(nn.Module):
    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif "tiny" in self.dataset:
            return 200
        elif "imagenet" in self.dataset:
            return 1000
        elif "femnist" == self.dataset:
            return 62

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0, std=0.01)
            #     m.bias.data.zero_()

    def _make_block(
        self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None,track_running_stats=True
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(group_norm_num_groups, planes=planes * block_fn.expansion,track_running_stats=track_running_stats),
            )

        layers = []
        layers.append( #  block_fn 为两个3*3的卷积层的block块
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
                track_running_stats=track_running_stats
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                    track_running_stats=track_running_stats
                )
            )
        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(ResNetBase, self).train(mode)

        # if self.freeze_bn:
        #     for m in self.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.eval()
        #             if self.freeze_bn_affine:
        #                 m.weight.requires_grad = False
        #                 m.bias.requires_grad = False


class ResNet_imagenet(ResNetBase):
    def __init__(
        self,
        dataset,
        resnet_size,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        projection=False,
        save_activations=False
    ):
        super(ResNet_imagenet, self).__init__()
        self.dataset = dataset
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        track_running_stats = not self.freeze_bn
        # define model param.
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[resnet_size]["block"]
        block_nums = model_params[resnet_size]["layers"]

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False,
        # )
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm2d(group_norm_num_groups, planes=64,track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=64,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats = track_running_stats
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        self.layer4 = self._make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = projection

        if self.projection:
            self.projection_layer = nn.Sequential(
                nn.Linear(512 * block_fn.expansion, 512 * block_fn.expansion),
                nn.ReLU(),
                nn.Linear(512 * block_fn.expansion, 256)
            )
            self.classifier = nn.Linear(
                in_features=256,
                out_features=self.num_classes,
            )
        else:
            self.classifier = nn.Linear(
                in_features=512 * block_fn.expansion, out_features=self.num_classes
            )
        self.save_activations = save_activations
        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward(self, x,start_layer_idx = 0):
        if start_layer_idx >= 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            #x = self.maxpool(x)
            x = self.layer1(x)
            activation1 = x
            x = self.layer2(x)
            activation2 = x
            x = self.layer3(x)
            activation3 = x
            x = self.layer4(x)
            activation4 = x
            x = self.avgpool(x)
            feature = x.view(x.size(0), -1)
            if self.projection:
                feature = self.projection_layer(feature)

            if self.save_activations:
                self.activations = [activation1, activation2, activation3,activation4]
        else:
            feature = x
        x = self.classifier(feature)
        return F.normalize(feature, dim=1),x

class CifarResNet(ResNetBase):
    def __init__(
        self,
        dataset,
        resnet_size,
        scaling=1,
        save_activations=False,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        projection=False,
        is_meta=False
    ):
        super(CifarResNet, self).__init__()

        self.dataset = dataset
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        track_running_stats = not self.freeze_bn

        # --- 1. 修改：层数配置逻辑 ---
        if resnet_size == 18:
            # ResNet-18 标准配置: 4个layer, 每个2个block
            self.layers_config = [2, 2, 2, 2]
            block_fn = BasicBlock
            self.has_layer4 = True
        else:
            # 原 CIFAR ResNet (20, 32, 44, 56, 110) 配置: 3个layer, 6n+2
            if resnet_size % 6 != 2:
                 raise ValueError("resnet_size must be 6n + 2 or 18:", resnet_size)
            n = (resnet_size - 2) // 6
            self.layers_config = [n, n, n]
            block_fn = Bottleneck if resnet_size >= 44 else BasicBlock
            self.has_layer4 = False

        self.num_classes = self._decide_num_classes()

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling), track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        # Layer 1
        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=self.layers_config[0],
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        # Layer 2
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=self.layers_config[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )
        # Layer 3
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=self.layers_config[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
            track_running_stats=track_running_stats
        )

        # --- 2. 修改：添加 Layer 4 (仅针对 ResNet-18) ---
        if self.has_layer4:
            self.layer4 = self._make_block(
                block_fn=block_fn,
                planes=int(128 * scaling), # 再次翻倍
                block_num=self.layers_config[3],
                stride=2,
                group_norm_num_groups=group_norm_num_groups,
                track_running_stats=track_running_stats
            )
            final_planes = int(128 * scaling)
            # ResNet-18 经过3次下采样(stride=2)，32x32 -> 4x4
            # 所以 avgpool kernel_size 应为 4
            pool_kernel = 4
        else:
            final_planes = int(64 * scaling)
            # ResNet-20等 经过2次下采样(layer2, layer3)，32x32 -> 8x8
            pool_kernel = 8

        self.avgpool = nn.AvgPool2d(kernel_size=pool_kernel)
        
        feature_dim = int(final_planes * block_fn.expansion)
        
        self.projection = projection
        if self.projection:
            self.projection_layer = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 256)
            )
            self.classifier = nn.Linear(256, self.num_classes)
        else:
            self.classifier = nn.Linear(feature_dim, self.num_classes)

        self._weight_initialization()
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx >= 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            
            # --- 3. 修改：前向传播包含 Layer 4 ---
            if self.has_layer4:
                x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            feature = x
            if self.projection:
                feature = self.projection_layer(feature)
        else:
            feature = x
        
        x = self.classifier(feature)

        return F.normalize(feature, dim=1), x

# class CifarResNet(ResNetBase):
#     def __init__(
#         self,
#         dataset,
#         resnet_size,
#         scaling=1,
#         save_activations=False,
#         group_norm_num_groups=None,
#         freeze_bn=False,
#         freeze_bn_affine=False,
#         projection = False,
#         is_meta = False
#     ):
#         super(CifarResNet, self).__init__()

#         self.dataset = dataset
#         self.freeze_bn = freeze_bn # 停止统计全局均值和方差：BN 层将不再维护和更新全局的 running_mean（滑动平均均值）和 running_var（滑动平均方差）。
#         self.freeze_bn_affine = freeze_bn_affine
#         track_running_stats = not self.freeze_bn # bn层的仿射变换是否有效

#         # define model.
#         if resnet_size % 6 != 2:
#             raise ValueError("resnet_size must be 6n + 2:", resnet_size)
#         block_nums = (resnet_size - 2) // 6
#         block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

#         # if is_meta:
#         #     block_fn = MetaBasicBlock

#         # decide the num of classes.
#         self.num_classes = self._decide_num_classes()

#         # define layers.
#         assert int(16 * scaling) > 0
#         self.inplanes = int(16 * scaling) # 第一个残差块的输入通道
#         self.conv1 = nn.Conv2d(
#             in_channels=3,
#             out_channels=(16 * scaling),
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False,
#         ) # 初始层
#         self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling),track_running_stats=track_running_stats) # 组归一化
#         self.relu = nn.ReLU(inplace=True)

#         self.layer1 = self._make_block( # 构建block_nums个残差块，每个块两层卷积
#             block_fn=block_fn,
#             planes=int(16 * scaling),
#             block_num=block_nums,
#             group_norm_num_groups=group_norm_num_groups,
#             track_running_stats=track_running_stats
#         )
#         self.layer2 = self._make_block(
#             block_fn=block_fn,
#             planes=int(32 * scaling),
#             block_num=block_nums,
#             stride=2,
#             group_norm_num_groups=group_norm_num_groups,
#             track_running_stats=track_running_stats
#         )
#         self.layer3 = self._make_block(
#             block_fn=block_fn,
#             planes=int(64 * scaling),
#             block_num=block_nums,
#             stride=2,
#             group_norm_num_groups=group_norm_num_groups,
#             track_running_stats=track_running_stats
#         )

#         self.avgpool = nn.AvgPool2d(kernel_size=8)
#         feature_dim = int(64 * scaling * block_fn.expansion)
#         self.projection = projection
#         if self.projection: # 通常用于特征对齐，提升性能

#             self.projection_layer = nn.Sequential(
#                 nn.Linear(feature_dim,feature_dim),
#                 nn.ReLU(),
#                 nn.Linear(feature_dim,256)
#             )
#             self.classifier = nn.Linear(
#                 in_features=256,
#                 out_features=self.num_classes,
#             )
#         else:
#             self.classifier = nn.Linear(
#                 in_features=feature_dim,
#                 out_features=self.num_classes,
#             )
#         # weight initialization based on layer type.
#         self._weight_initialization()

#         # a placeholder for activations in the intermediate layers.
#         self.save_activations = save_activations
#         self.activations = None

#     def forward(self, x, start_layer_idx = 0):
#         if start_layer_idx >= 0:
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)

#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             feature = x
#             if self.projection:
#                 feature = self.projection_layer(feature)
#         else:
#             feature = x
#         x = self.classifier(feature)

#         return F.normalize(feature, dim=1),x


def hybrid_resnet18(ratio_LR=1, decom_rule=[1, 1], track=False, cfg=None):

    data_shape = cfg.data_shape
    classes_size = cfg.classes_size
    hidden_size = cfg.resnet.hidden_size
    model = HyperResNet(data_shape, hidden_size, BasicBlock, [2, 2, 2, 2], ratio_LR=ratio_LR,group_norm_num_groups=cfg.group_norm_num_groups,
                        decom_rule=decom_rule, num_classes=classes_size, track=track, cfg=cfg)
    return model


def hybrid_resnet34(model_rate=1, ratio_LR=1, decom_rule=[1, 1], track=False, cfg=None):
    """
    :param model_rate:
    :param track:
    :param cfg:
    :return:
    """
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    scaler_rate = model_rate / cfg['global_model_rate']
    model = HyperResNet(data_shape, hidden_size, BasicBlock, [3, 4, 6, 3], ratio_LR=ratio_LR, decom_rule=decom_rule,group_norm_num_groups=cfg.group_norm_num_groups,
                        num_classes=classes_size, rate=scaler_rate, track=track, cfg=cfg)
    # model.apply(init_param)
    return model

def hybrid_resnet8(ratio_LR=1, decom_rule=[1, 1], track=False, cfg=None):
    data_shape = cfg.data_shape
    classes_size = cfg.classes_size
    hidden_size = cfg.resnet.hidden_size
    if cfg.freeze_bn:
        track = False
    model = HyperResNet( data_shape, hidden_size, BasicBlock, [1,1,1], ratio_LR=ratio_LR, group_norm_num_groups=cfg.group_norm_num_groups,
                        decom_rule=decom_rule, num_classes=classes_size, track=track, cfg=cfg)
    return model




def resnet(conf, arch=None):

    resnet_size = int((arch if arch is not None else conf.arch).replace("resnet", "")) # 层数
    dataset = conf.data
    save_activations = True if conf.AT_beta > 0 else False


    if "cifar" in conf.data or "svhn" in conf.data:
        # model = ResNet_cifar(
        #     dataset=dataset,
        #     resnet_size=resnet_size,
        #     freeze_bn=conf.freeze_bn,
        #     freeze_bn_affine=conf.freeze_bn_affine,
        #     group_norm_num_groups=conf.group_norm_num_groups,
        # )

            # model = HyperResNet(data_shape, hidden_size, BasicBlock, [3, 4, 6, 3], ratio_LR=ratio_LR, decom_rule=decom_rule,
            #             num_classes=classes_size, rate=scaler_rate, track=track, cfg=cfg)


        if conf.meta:
            if conf.arch == "resnet18":
                model = hybrid_resnet18(ratio_LR=conf.ratio_LR, decom_rule=conf.decom_rule, cfg=conf)
            elif conf.arch == "resnet8":
                model = hybrid_resnet8(ratio_LR=conf.ratio_LR, decom_rule=conf.decom_rule, cfg=conf)
            elif conf.arch == "resnet34":
                model.arch = hybrid_resnet34(ratio_LR=conf.ratio_LR, decom_rule=conf.decom_rule, cfg=conf)
        else:
            model = CifarResNet(
                dataset=dataset,
                resnet_size=resnet_size,
                freeze_bn=conf.freeze_bn,
                freeze_bn_affine=conf.freeze_bn_affine,
                group_norm_num_groups=conf.group_norm_num_groups,
                projection=conf.projection,
                save_activations = save_activations,
                is_meta=conf.meta,
                scaling=conf.resnet_scaling
            )
    elif "imagenet" in dataset:
        if dataset == "tiny-imagenet" or dataset == "imagenet":
            model = ResNet_imagenet(
                dataset=dataset,
                resnet_size=resnet_size,
                group_norm_num_groups=conf.group_norm_num_groups,
                freeze_bn=conf.freeze_bn,
                freeze_bn_affine=conf.freeze_bn_affine,
                projection=conf.projection,
                save_activations=save_activations
            )
        # if (
        #     "imagenet" in conf.data and len(conf.data) > 8
        # ):  # i.e., downsampled imagenet with different resolution.
        else:
            model = CifarResNet(
            dataset=dataset,
            resnet_size=resnet_size,
            freeze_bn=conf.freeze_bn,
            freeze_bn_affine=conf.freeze_bn_affine,
            group_norm_num_groups=conf.group_norm_num_groups,
            projection=conf.projection,
            scaling=4
        )
            # model = ResNet_cifar(
            #     dataset=dataset,
            #     resnet_size=resnet_size,
            #     scaling=4,
            #     group_norm_num_groups=conf.group_norm_num_groups,
            #     freeze_bn=conf.freeze_bn,
            #     freeze_bn_affine=conf.freeze_bn_affine,
            # )

    else:
        raise NotImplementedError
    return model


