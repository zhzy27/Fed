# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torchvision
import copy
import numpy as np
import string

__all__ = ["simple_cnn","MetaCNN"]


def _decide_num_classes(dataset):
    if dataset == "cifar10" or dataset == "svhn":
        return 10
    elif dataset == "cifar100":
        return 100
    elif "imagenet" in dataset:
        return 1000
    elif "mnist" == dataset:
        return 10
    elif "femnist" == dataset:
        return 62
    else:
        raise NotImplementedError(f"this dataset ({dataset}) is not supported yet.")


class CNNMnist(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True):
        super(CNNMnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=w_conv_bias)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=w_conv_bias)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50, bias=w_fc_bias)
        self.classifier = nn.Linear(50, self.num_classes, bias=w_fc_bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x


class CNNfemnist(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNfemnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        # define layers.
        self.conv1 = nn.Conv2d(1, 32, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(1024, 2048, bias=w_fc_bias)
        self.classifier = nn.Linear(2048, self.num_classes, bias=w_fc_bias)

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        activation1 = self.conv1(x)
        x = self.pool(F.relu(activation1))

        activation2 = self.conv2(x)

        x = self.pool(F.relu(activation2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2]
        return x


class CNNCifar(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNCifar, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        # define layers.
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 64)
        self.classifier = nn.Linear(64, self.num_classes, bias=w_fc_bias)

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)

        x = self.pool(F.relu(x))
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 1024)
        feat = F.relu(self.fc1(x))
        x = self.classifier(feat)

        return feat, x
    
# 新增5层卷积

class CNNCifar5Layer(nn.Module):
    def __init__(self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True):
        super(CNNCifar5Layer, self).__init__()

        # 确定类别数
        self.num_classes = _decide_num_classes(dataset)
        
        # 注意：如果使用 BatchNorm，卷积层的 bias 通常设为 False，因为 BN 自带 bias shift
        # 但为了兼容您的 conf配置，这里保留逻辑，但在下面 layer 定义时强制 bias=False (推荐)
        use_bias = False 

        # === 5层卷积结构 (VGG Style Block: Conv-BN-ReLU) ===
        
        # Layer 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=use_bias)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Layer 4: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=use_bias)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Layer 5: 256 -> 256 (保持通道数，提取高层语义)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias)
        self.bn5 = nn.BatchNorm2d(256)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout (防止过拟合)
        self.dropout = nn.Dropout(p=0.5)

        # 计算全连接层输入维度
        # 32x32 -> (Pool) -> 16x16 -> (Pool) -> 8x8 -> (Pool) -> 4x4
        # 最终特征图大小: 256 channels * 4 * 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512, bias=w_fc_bias)
        self.classifier = nn.Linear(512, self.num_classes, bias=w_fc_bias)

        self.save_activations = save_activations
        self.activations = None

    def forward(self, x,start_layer_idx=0):
        if start_layer_idx < 0:
            
            logits = self.classifier(x)
            return x, logits
        # Block 1
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)        # 32 -> 16
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(self.bn2(x)) 
        # 这里不加池化，保持特征更丰富，或者隔层池化
        
        # Block 3
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool(x)        # 16 -> 8
        
        # Block 4
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        # 不加池化
        
        # Block 5
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool(x)        # 8 -> 4

        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # FC Block
        x = F.relu(self.fc1(x))
        x = self.dropout(x)     # 关键：训练时随机丢弃神经元
        
        logits = self.classifier(x)

        return x, logits # 返回 feat, logits 保持与之前接口一致

   # 定义分解的cnn网络
class MetaCNN(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True, rank_rate=1.0):
        super(MetaCNN, self).__init__()
        
        self.num_classes = _decide_num_classes(dataset)
        self.rank_rate = rank_rate # 秩率
        self.w_conv_bias = w_conv_bias # 记录配置以便恢复时使用
        self.w_fc_bias = w_fc_bias

        # BN层通常不需要卷积偏置
        use_conv_bias = False 

        # === 前4层卷积 (保持固定，不进行分解) ===
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=use_conv_bias)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=use_conv_bias)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=use_conv_bias)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=use_conv_bias)
        self.bn4 = nn.BatchNorm2d(256)

        # === 第5层卷积 (可分解目标) ===
        # 输入: 256, 输出: 256
        if rank_rate >= 1.0:
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_conv_bias)
        else:
            self.conv5 = FactorizedConv(in_channels=256, out_channels=256, 
                                        rank_rate=rank_rate, kernel_size=3, 
                                        padding=1, stride=1, bias=use_conv_bias)
        
        self.bn5 = nn.BatchNorm2d(256)

        # 池化与Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

        # 计算Flatten后的维度: 256 * 4 * 4 = 4096
        self.flat_dim = 256 * 4 * 4
        
        # === FC1 层 (可分解目标) ===
        # 输入: 4096, 输出: 512
        if rank_rate >= 1.0:
            self.fc1 = nn.Linear(self.flat_dim, 512, bias=w_fc_bias)
        else:
            self.fc1 = FactorizedLinear(in_features=self.flat_dim, out_features=512, 
                                        rank_rate=rank_rate, bias=w_fc_bias)

        # === Classifier (保持固定) ===
        self.classifier = nn.Linear(512, self.num_classes, bias=w_fc_bias)

        # [ADD] 新增全局共享、可训练的 CLIP 适配层
        # 输入是 fc1 输出的 512 维特征，输出是对齐 CLIP 的 512 维特征
        self.clip_adapter = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512)
                )

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2 (No pool)
        x = F.relu(self.bn2(self.conv2(x)))
        # Layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Layer 4 (No pool)
        x = F.relu(self.bn4(self.conv4(x)))
        
        # === Layer 5 (可能已分解) ===
        # 注意: FactorizedConv 的 forward 已经处理了卷积逻辑
        x = self.conv5(x) 
        x = self.pool(F.relu(self.bn5(x)))

        # Flatten
        x = x.view(-1, self.flat_dim)

        # === FC1 (可能已分解) ===
        x = self.fc1(x)
        x = F.relu(x)
        
        # Dropout & Classifier
        x = self.dropout(x)
        logits = self.classifier(x)

        return x, logits

    # === 模型分解与恢复方法 ===

    def decom_model(self, rank_rate):
        """将 conv5 和 fc1 分解为低秩层"""
        if rank_rate >= 1.0:
            return
        
        print(f"Executing Decomposition (rank_rate={rank_rate})...")
        
        # 分解 Conv5
        if isinstance(self.conv5, nn.Conv2d):
            # 调用你提供的 Decom_COV
            self.conv5 = Decom_COV(self.conv5, rank_rate)
            print(" -> conv5 decomposed.")

        # 分解 FC1
        if isinstance(self.fc1, nn.Linear):
            # 调用你提供的 Decom_LINEAR
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
            print(" -> fc1 decomposed.")
            
        self.rank_rate = rank_rate

    def recover_model(self):
        """将 conv5 和 fc1 恢复为完整层"""
        if self.rank_rate >= 1.0:
            return

        print("Executing Recovery to Full Rank...")

        # 恢复 Conv5
        if isinstance(self.conv5, FactorizedConv):
            # 调用你提供的 Recover_COV
            self.conv5 = Recover_COV(self.conv5)
            print(" -> conv5 recovered.")

        # 恢复 FC1
        if isinstance(self.fc1, FactorizedLinear):
            # 调用你提供的 Recover_LINEAR
            self.fc1 = Recover_LINEAR(self.fc1)
            print(" -> fc1 recovered.")
            
        # self.rank_rate = 1.0

    # === 正则化 Loss 计算 (用于低秩训练阶段) ===
    
    def get_decomposed_loss(self, type='frobenius'):
        """计算分解层的正则化损失"""
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        
        if self.rank_rate >= 1.0:
            return loss

        # 获取 Conv5 的 loss
        if isinstance(self.conv5, FactorizedConv):
            if type == 'frobenius': loss += self.conv5.frobenius_loss()
            elif type == 'kronecker': loss += self.conv5.kronecker_loss()
            elif type == 'l2': loss += self.conv5.L2_loss()

        # 获取 FC1 的 loss
        if isinstance(self.fc1, FactorizedLinear):
            if type == 'frobenius': loss += self.fc1.frobenius_loss()
            elif type == 'kronecker': loss += self.fc1.kronecker_loss()
            elif type == 'l2': loss += self.fc1.L2_loss()
            
        return loss


        

def simple_cnn(conf):
    dataset = conf.data
#     获取秩 rank_rate
    rank_rate = conf.ratio_LR
    # rank_rate = 0.5

    if "cifar" in dataset or dataset == "svhn":
        if conf.meta: # 如果分解
            return MetaCNN(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias, rank_rate =rank_rate )
        # return CNNCifar(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
        return CNNCifar5Layer(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)

    elif "mnist" == dataset:
        return CNNMnist(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
    elif "femnist" == dataset:
        return CNNfemnist(
            dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias
        )
    else:
        raise NotImplementedError(f"not supported yet.")
    




# 分解后的卷积层 
class FactorizedConv(nn.Module):
    '''
    完整的 4维卷积核 转换为 两个较小的 2维矩阵 conv_u 和 conv_v。
    conv_v: 形状为 [rank, in_channels * kernel_size]。这是分解后的第一层权重（降维/特征变换）。
    conv_u: 形状为 [out_channels * kernel_size, rank]。这是分解后的第二层权重（升维/特征重组）。
    
    '''

    def __init__(self, in_channels, out_channels, rank_rate, padding=None, stride=1, kernel_size=3, bias=False):
        super(FactorizedConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding if padding is not None else 0
        self.groups = 1  # 分组卷积支持，默认为1

        # 计算低秩分解的秩
        # self.rank = max(1, round(rank_rate * min(in_channels, out_channels)))
        self.rank = max(1, round(rank_rate * min(out_channels * kernel_size, in_channels * kernel_size)))
        # 使用二维矩阵存储分解参数
        # 通用处理任意kernel_size
        self.dim1 = out_channels * kernel_size #对应卷积层分解成矩阵后的两个维度
        self.dim2 = in_channels * kernel_size

        # 低秩参数矩阵 (二维存储) 可学习 随机初始化分解形状的卷积层

        self.conv_v = nn.Parameter(torch.Tensor(self.rank, self.dim2)) #对应形状 [rank, in_channels * kernel_size]
        self.conv_u = nn.Parameter(torch.Tensor(self.dim1, self.rank)) #对应形状 [out_channels * kernel_size, rank]

        # 偏置参数 
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels)) # 偏执维度对应输出维度
        else:
            self.register_parameter('bias', None) # 注册为空 可以访问
        # 初始化模型参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_u, a=math.sqrt(0)) #使用 Kaiming 均匀分布初始化分解后的卷积参数
        nn.init.kaiming_uniform_(self.conv_v, a=math.sqrt(0))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 空间分解: 1xK + Kx1
        # 垂直卷积 (1xK)
        weight_v = self.conv_v.T.reshape(self.in_channels, self.kernel_size, 1, self.rank).permute(3, 0, 2, 1)
        #对应形状 [rank, in_channels * kernel_size]->转置 [in_channels * kernel_size, rank]
        # ->重塑 [in_channels, kernel_size,1, rank]->换轴 [rank, in_channels,1,kernel_size]
        out = F.conv2d(
            x, weight_v, None,
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, 1),
            groups=self.groups
        )

        # 水平卷积 (Kx1)  不显示reshpe权重
        weight_u = self.conv_u.reshape(self.out_channels, self.kernel_size, self.rank, 1).permute(0, 2, 1, 3)
         #[out_channels * kernel_size, rank] -> 重塑 [out_channels, kernel_size, rank, 1] ->
         #  换轴 [out_channels, rank, kernel_size, 1]

        out = F.conv2d(
            out, weight_u, self.bias,
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dilation=(1, 1),
            groups=self.groups
        )
        return out
# 正则损失
    def frobenius_loss(self):
        return torch.sum((self.conv_u @ self.conv_v) ** 2)


    def reconstruct_full_weight(self):
        """重建完整的卷积核权重 (用于聚合)"""
        # 直接使用矩阵乘法
        A = self.conv_u @ self.conv_v  # [out*K, in*K]
        W = A.reshape(self.out_channels, self.kernel_size, self.in_channels, self.kernel_size)
        W = W.permute(0, 2, 1, 3)  # [out, in, K, K]
        return W

    def L2_loss(self):
        """分解参数的L2范数平方和"""
        return torch.norm(self.conv_u, p='fro') ** 2 + torch.norm(self.conv_v, p='fro') ** 2

    def kronecker_loss(self):
        """Kronecker乘积损失"""
        return (torch.norm(self.conv_u, p='fro') ** 2) * (torch.norm(self.conv_v, p='fro') ** 2)
    
def Decom_COV(conv_model, ratio_LR=0.5):
    # 自动从卷积层获取参数
    in_planes = conv_model.in_channels
    out_planes = conv_model.out_channels
    kernel_size = conv_model.kernel_size[0]
    stride = conv_model.stride[0]
    padding = conv_model.padding[0]
    bias = conv_model.bias is not None

    # 创建分解层 (使用二维矩阵存储)
    factorized_cov = FactorizedConv(
        in_planes,
        out_planes,
        rank_rate=ratio_LR,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias
    )

    # 获取原始权重并重塑
    W = conv_model.weight.data

    # 重塑: [out, in, K, K] -> [out*K, in*K]
    A = W.permute(0, 2, 1, 3).reshape(out_planes * kernel_size, in_planes * kernel_size)

    # SVD分解
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # 计算截断秩
    rank = factorized_cov.rank
    S_sqrt = torch.sqrt(S[:rank]) # 截断秩 开平方 
    # 分配奇异值
    U_weight = U[:, :rank] @ torch.diag(S_sqrt) 
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]

    # 加载参数
    with torch.no_grad():
        factorized_cov.conv_u.copy_(U_weight)
        factorized_cov.conv_v.copy_(V_weight)

        # 复制偏置
        if bias:
            factorized_cov.bias.copy_(conv_model.bias.data)

    return factorized_cov


# 卷积层恢复函数
def Recover_COV(decom_conv):
    # 获取分解层参数
    in_planes = decom_conv.in_channels
    out_planes = decom_conv.out_channels
    kernel_size = decom_conv.kernel_size
    stride = decom_conv.stride
    padding = decom_conv.padding
    bias = decom_conv.bias is not None

    # 重建完整权重
    W = decom_conv.reconstruct_full_weight()

    # 创建原始卷积层
    recovered_conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias
    )

    # 加载权重
    with torch.no_grad():
        recovered_conv.weight.copy_(W)
        if bias:
            recovered_conv.bias.copy_(decom_conv.bias)

    return recovered_conv

# 分解后的全连接层 (二维矩阵存储)
class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_rate, bias=True):
        super(FactorizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 中间rank值
        self.rank = max(1, round(rank_rate * min(in_features, out_features)))

        # 二维矩阵参数
        # 第一个全连接层的参数（维度为 r*in）
        self.weight_v = nn.Parameter(torch.Tensor(self.rank, in_features))
        # 第二个全连接层的参数 (维度为 out*r)
        self.weight_u = nn.Parameter(torch.Tensor(out_features, self.rank))

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_u, a=math.sqrt(0))
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(0))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight_u.size(1))  # rank
            nn.init.uniform_(self.bias, -bound, bound)

    def reconstruct_full_weight(self):
        return self.weight_u @ self.weight_v

    def forward(self, x):
        """
        前向传播分为两步：
        1. 降维投影：x -> (batch,  rank)
        2. 升维投影：x -> (batch, out_features)
        """
        # 第一步：降维投影 (输入特征空间 -> 低秩空间)  传入 F.linear() 的权重存储形式与正常线性层 (nn.Linear) 完全一致必须是（out*in）
        x = F.linear(x, self.weight_v, None)  # 形状: (batch,  rank)

        # 第二步：升维投影 (低秩空间 -> 输出特征空间)
        x = F.linear(x, self.weight_u, self.bias)  # 形状: (batch, out_features)

        return x

    def frobenius_loss(self):
        W = self.weight_u @ self.weight_v
        return torch.sum(W ** 2)

    def L2_loss(self):
        return torch.norm(self.weight_v) ** 2 + torch.norm(self.weight_u) ** 2

    def kronecker_loss(self):
        return (torch.norm(self.weight_v) ** 2) * (torch.norm(self.weight_u) ** 2)


# 全连接层分解函数(将全连接权重  W（out*in）分解为 out*r（第二个全连接权重） r*in （第一个权全连接重）)
def Decom_LINEAR(linear_model, ratio_LR=0.5):
    in_features = linear_model.in_features
    out_features = linear_model.out_features
    has_bias = linear_model.bias is not None

    # 创建分解层
    factorized_linear = FactorizedLinear(in_features, out_features, ratio_LR, has_bias)

    # SVD分解（属注意与torch.svd函数的区别  主要是第三个矩阵）  Vh是V矩阵的转置（就是第三个矩阵）
    U, S, Vh = torch.linalg.svd(linear_model.weight.data, full_matrices=False)

    # 计算截断秩
    rank = factorized_linear.rank

    # 分配奇异值  第一个矩阵切列   第三个矩阵切行
    S_sqrt = torch.sqrt(S[:rank])
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)  # shape out*r
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]  # shape r*in

    # 加载参数
    with torch.no_grad():
        factorized_linear.weight_v.copy_(V_weight)
        factorized_linear.weight_u.copy_(U_weight)
        if has_bias:
            factorized_linear.bias.copy_(linear_model.bias.data)

    return factorized_linear


# 全连接层恢复函数
def Recover_LINEAR(factorized_linear):
    in_features = factorized_linear.in_features
    out_features = factorized_linear.out_features
    has_bias = factorized_linear.bias is not None

    # # 重建权重
    # weight = factorized_linear.weight_u @ factorized_linear.weight_v
    weight = factorized_linear.reconstruct_full_weight()

    # 创建原始线性层
    recovered_linear = nn.Linear(in_features, out_features, bias=has_bias)

    # 加载参数
    with torch.no_grad():
        recovered_linear.weight.copy_(weight)
        if has_bias:
            recovered_linear.bias.copy_(factorized_linear.bias)

    return recovered_linear
