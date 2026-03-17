# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import pcode.local_training.random_reinit as random_reinit
import pcode.datasets.mixup_data as mixup
import pcode.create_model as create_model
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.create_metrics as create_metrics
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat
from pcode.utils.timer import Timer
from pcode.utils.stat_tracker import RuntimeTracker
import  time
import copy
import clip
import os


class WorkerFedOur(object):
    def __init__(self, conf):
        
        self.conf = conf
        # some initializations.
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")
        conf.device = self.device
        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=conf.logger.log_metric,
        )
        
        # create dataset (as well as the potential data_partitioner) for training.
        self.anchor_weight = torch.tensor([0.125, 0.25, 0.5, 1.0], device=self.device)
        self.text_model, _ = self.load_clip_text_model()
        # self.anchor = self.generate_text_anchors()
        self.output_dim = self.text_model.text_projection.shape[1]
        
        
        conf.output_dim = self.output_dim 
        self.conf = conf

        
        print("anchor:{self.anchor}")

        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data, agg_data_ratio=conf.agg_data_ratio)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        )

        conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} initialized the local training data with Master."
        )

        # define the criterion.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} initialized dataset/criterion.\n"
        )   # 打印当前的进程编号

    def extra_init(self):
        # 1. 冻结 CLIP 文本模型的自身参数
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # 2. 初始化可学习的 Prompt (Context Tokens)
        n_ctx = 16  # 你可以调节提示词的长度 (如 4, 8, 16)
        ctx_dim = self.text_model.token_embedding.weight.shape[1] # 通常是 512
        
        # 将参数挂载到 self.model 上，这样联邦通信和优化器都能自动接管
        self.model.prompt_ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim).to(self.device))
        nn.init.normal_(self.model.prompt_ctx, std=0.02)
        
        # 3. 预先处理好类名的 Token (不带 "a photo of a")
        # 假设你原来的 data_classes_registry 依然可用
        dataset_name = self.conf.data.lower()
        if dataset_name == "cifar10":
            classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        elif dataset_name == "cifar100":
            classes = ['apple', 'aquarium fish', 'baby', 'bear', 'beaver'] # 填入你的 cifar100 完整列表
        else:
            raise ValueError("Dataset not supported.")
            
        prompts = [f"{c}" for c in classes] # 只保留类别名
        self.tokenized_prompts = clip.tokenize(prompts).to(self.device)
        
        # 预先提取类别名的基础 Embedding（节省每次前向传播的时间）
        with torch.no_grad():
            self.class_embedding = self.text_model.token_embedding(self.tokenized_prompts).type(self.text_model.dtype)


    def load_clip_text_model(self, model_name="ViT-B/32", save_dir="./model_state/"):
        """
        仅加载 CLIP 的文本提取部分。
        
        1. 自动检查本地/下载 (逻辑不变)。
        2. 使用 jit=False 加载，以便我们可以修改模型结构。
        3. 删除 visual 部分以节省显存。
        """
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Loading CLIP Text Backbone: {model_name}...")
        
        # [关键修改 1] jit=False: 允许我们将模型作为普通的 PyTorch nn.Module 加载，
        # 这样我们才能执行 del model.visual 操作。
        # model 的 preprocess (针对图像) 我们直接忽略，用不到。
        model, _ = clip.load(model_name, device=device, download_root=save_dir, jit=False)
        # [关键修改 2] 删除视觉编码器部分
        # 这会释放掉 ViT 或 ResNet 部分占用的显存
        if hasattr(model, 'visual'):
            # 1. 先保存当前的数据类型 (通常是 float16 或 float32)
            # 注意：如果 visual 已经被删了，这里会报错，所以要小心
            try:
                stored_dtype = model.visual.conv1.weight.dtype
            except:
                stored_dtype = torch.float16 # 默认备选
                
            # 2. 删除真正的视觉编码器 (释放显存)
            del model.visual
            if device == "cuda":
                torch.cuda.empty_cache()
                
            # 3. [核心] 塞入一个假的 visual，只为了让 model.dtype 属性不报错
            model.visual = DummyVisual(stored_dtype)
                
        print("Text-only model loaded. Visual encoder removed.")
        
        # 将模型设为评估模式 (对于文本部分这通常不影响，但好习惯)
        model.eval()
        model.to("cuda")

        return model, device
    
    def get_dynamic_anchors(self):
        """
        每次迭代动态生成带有梯度的 Anchors。
        流程: [SOS] + Learnable_Context + [Class_Name] + [EOS]
        """
        # 注意：这里一定不能用 torch.no_grad()，因为我们需要 prompt_ctx 的梯度！
        self.text_model.eval() 
        
        batch_size = len(self.class_embedding)
        n_ctx = self.model.prompt_ctx.shape[0]



        prompt_ctx_casted = self.model.prompt_ctx.to(self.text_model.dtype)     
        # 使用转换后的 prompt_ctx_casted 进行扩展
        ctx = prompt_ctx_casted.unsqueeze(0).expand(batch_size, -1, -1)
        
        # CLIP Token 结构分离
        # prefix: [SOS] token (索引 0)
        # suffix: 类别名称 + [EOS] + 填充 (索引 1 开始)
        prefix = self.class_embedding[:, :1, :] 
        suffix = self.class_embedding[:, 1:, :] 
        
        # 拼接并在 77 处截断以符合 CLIP 的输入限制
        x = torch.cat([prefix, ctx, suffix], dim=1)[:, :77, :]
        
        # 添加位置编码
        x = x + self.text_model.positional_embedding.type(self.text_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        layers = self.text_model.transformer.resblocks
        total_layers = len(layers)
        indices = [
            int(total_layers * 1 / 4) - 1,
            int(total_layers * 2 / 4) - 1,
            int(total_layers * 3 / 4) - 1,
            total_layers - 1
        ]
        
        anchors_list = []
        for i, layer in enumerate(layers):
            x = layer(x)
            if i in indices:
                x_temp = x.permute(1, 0, 2)
                
                # 由于插入了 n_ctx 个 token，EOS 的位置向后移动了 n_ctx
                eos_indices = self.tokenized_prompts.argmax(dim=-1) + n_ctx
                eos_indices = torch.clamp(eos_indices, max=76) # 防止越界
                
                sent_emb = x_temp[torch.arange(x_temp.shape[0]), eos_indices]
                
                if i == total_layers - 1:
                    sent_emb = self.text_model.ln_final(sent_emb).type(self.text_model.dtype)
                    sent_emb = sent_emb @ self.text_model.text_projection
                    
                sent_emb = sent_emb / sent_emb.norm(dim=-1, keepdim=True)
                anchors_list.append(sent_emb.float())
                
        return anchors_list

    def generate_text_anchors(self):
        """
        根据数据集名称生成文本锚点 (Text Anchors)。
        
        流程:
        1. 获取类别名称列表 (classes).
        2. 构造 Prompts (templates).
        3. Tokenize -> Text Encoder -> Normalize.
        """
        
        # 1. 获取当前设备
        # 假设 self.text_model 已经在正确的 device 上 (比如 cpu 或 cuda)
        device = next(self.text_model.parameters()).device
        
        # 2. 定义数据集与其对应的类别列表
        # 这里列出了 CIFAR-10 和 CIFAR-100 的标准类别
        data_classes_registry = {
            "cifar10": [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ],
            "cifar100": [
                'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ],
            # 你可以在这里添加 tinyimagenet 或其他数据集
        }

        dataset_name = self.conf.data.lower() # 确保大小写匹配
        
        if dataset_name not in data_classes_registry:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry. Please add class names to generate_text_anchors.")
        
        classes = data_classes_registry[dataset_name]
        
        # 3. 构造提示词 (Prompt Template)
        # CLIP 官方推荐使用 "a photo of a {label}"
        prompts = [f"a photo of a {c}" for c in classes]
        

        # 4. Tokenize
        # clip.tokenize 会自动截断或填充到 77 token 长度
        text_inputs = clip.tokenize(prompts).to(device)
        
        # 5. 提取特征 (Inference)
        # 确保不计算梯度，确保模型处于 eval 模式
        self.text_model.eval()
        with torch.no_grad():
            # encode_text 是 CLIP 模型提取文本特征的标准方法
            x = self.text_model.token_embedding(text_inputs).type(self.text_model.dtype)
            x = x + self.text_model.positional_embedding.type(self.text_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            # 2. 获取 Transformer 的层数
            # ViT-B/32 的 layers=12
            layers = self.text_model.transformer.resblocks
            total_layers = len(layers)
            
            # 3. 定义我们要截取的节点 (均匀分布)
            # 例如 12层 -> [2, 5, 8, 11] (索引从0开始)
            # 这里的逻辑是：ResNet Stage1 对齐 Layer3，Stage4 对齐 Layer12
            indices = [
                int(total_layers * 1 / 4) - 1,
                int(total_layers * 2 / 4) - 1,
                int(total_layers * 3 / 4) - 1,
                total_layers - 1
            ]
            
            anchors_list = []
            
            # 4. 前向传播并截取
            for i, layer in enumerate(layers):
                x = layer(x)
                
                if i in indices:
                    # 取出 [EOS] token 的特征 (就像 CLIP 标准做法一样)
                    # x 形状: [L, Batch, Dim] -> permute -> [Batch, L, Dim]
                    x_temp = x.permute(1, 0, 2)
                    
                    # text_inputs.argmax(dim=-1) 找到 EOS 的位置
                    # 提取特征
                    sent_emb = x_temp[torch.arange(x_temp.shape[0]), text_inputs.argmax(dim=-1)]
                    
                    # [关键]
                    # 只有最后一层 (i == total_layers - 1) 需要通过 text_projection 映射到联合空间
                    # 前面的层保持原样，或者也通过 LayerNorm
                    if i == total_layers - 1:
                        sent_emb = self.text_model.ln_final(sent_emb).type(self.text_model.dtype)
                        sent_emb = sent_emb @ self.text_model.text_projection
                    else:
                        # 中间层通常不需要 ln_final，或者你可以选择加上
                        pass 

                    # 归一化 (一定要做!)
                    sent_emb = sent_emb / sent_emb.norm(dim=-1, keepdim=True)
                    anchors_list.append(sent_emb.float()) # 转回 float32

            return anchors_list # 返回包含 4 个 Tensor 的列表

    def recv_extra_info_from_master(self):
        pass

    def send_extra_info_to_master(self):
        pass

    def get_params(self, model):
        weight_collector = None
        for param in model.parameters():
            if not isinstance(weight_collector, torch.Tensor):
                weight_collector = param.reshape(-1)
            else:
                weight_collector = torch.cat((weight_collector, param.reshape(-1)), 0)
        return weight_collector

    def run(self):
        while True:
            self.listen_to_master() # 每次可能分配不同的客户端

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self.recv_extra_info_from_master()

            self._recv_model_from_master() # 接受模型参数
            self._train()

            self.send_extra_info_to_master()

            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return

    def listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((3, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)

        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        ) # 取出属于自己的消息
        
        if self.conf.rank_list is not None:
            self.ratio_LR = self.conf.rank_list[self.conf.graph.client_id - 1]  # 取出属于自己的秩


        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        ) # 获取模型


        self.extra_init() # 将可训练提示词部分挂载到模型上

        self.model.to("cpu")

        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values())) # 将参数打包为一维参数，方便传输

        

        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            localdata_id=self.conf.graph.client_id - 1,
            is_train=True,
            data_partitioner=self.data_partitioner,
        )   # 这里的data_partitioner已经统一,这里传入的参时真正的客户端id

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values()) # 将接受的参数填入模型字典

        # if not self.conf.freeze_bn:
        self.model.load_state_dict(self.model_state_dict)   # 更新模型参数
        random_reinit.random_reinit_model(self.conf, self.model)    # 若为true，这里可能会随机丢弃参数
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device)) # 获得一个关闭了梯度的一摸一样的模型，常用于蒸馏

        # self.aggregation = Aggregation(self.model.classifier.in_features).cuda()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )

        dist.barrier()

    def _prepare_train(self):
        self.model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)
            self.text_model = self.text_model.to(self.device)

        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )   # 建立优化器，同时加入L2正则
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names) # 创建追踪器，记录损失和准确率
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )

    def prepare_train(self):
        self._prepare_train()

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature=None, target=None, prototypes = None):
        # 1. 基础损失
        self.anchor = self.get_dynamic_anchors()
        total_loss = loss + self.conf.meta_L2 * self.model.L2_decay()

        # 2. 计算层级锚点损失
        # 检查是否开启对齐 (可以通过 output 里的 stage_features 是否为空，或者 check use_align)
        # 注意：这里需要确保你已经正确解包了 output，例如 x, logits = output
        # 或者你现在的 output 就是 logits，而 stage_features 存在 self.model.stage_features 里
        if "resnet" in self.conf.arch:
            if self.model.use_align and hasattr(self, 'anchor'):
                
                # [关键步骤 A] 获取当前 Batch 的标签
                if "target" in data_batch:
                    current_target = data_batch["target"]
                elif target is not None:
                    current_target = target.to(self.device)
                else:
                    raise ValueError("local_training_with_extra_calculate") # 没标签没法算，直接返回

                # [关键步骤 B] 根据标签挑选锚点
                # self.anchor 是 list: [Tensor(10, 512), Tensor(10, 512)...]
                # 我们需要构造 batch_anchors: [Tensor(B, 512), Tensor(B, 512)...]
                current_batch_anchors = []
                for layer_idx in range(4):
                    # 利用高级索引，选出当前 target 对应的锚点行
                    # layer_anchor shape: [Batch_Size, Dim]
                    layer_anchor = self.anchor[layer_idx][current_target]
                    current_batch_anchors.append(layer_anchor)

                # [关键步骤 C] 传入处理好的 Batch 锚点
                # 返回的是 shape 为 [4] 的 tensor，包含 4 个阶段的 loss
                loss_vec = self.model.calculate_stage_anchor_loss(current_batch_anchors)
                
                # [关键步骤 D] 求和并加权
                # 因为 loss_vec 是 [L1, L2, L3, L4]，你需要把它变成一个标量才能加到 total_loss
                weight_tensor = torch.tensor(self.anchor_weight, device=loss_vec.device, dtype=loss_vec.dtype)
                anchor_loss_scalar = torch.sum(loss_vec * weight_tensor)
                
                total_loss += self.conf.anchor_loss * anchor_loss_scalar
        elif "cnn" in self.conf.arch:
            if "target" in data_batch:
                current_target = data_batch["target"]
            elif target is not None:
                current_target = target.to(self.device)
            else:
                raise ValueError("local_training_with_extra_calculate")
            sleleted_anchor = self.anchor[-1][current_target]
            aligned_feature = self.model.clip_adapter(feature)
            aligned_feature = F.normalize(aligned_feature, p=2,dim=1)
            mse_loss = F.mse_loss(aligned_feature, sleleted_anchor)
            total_loss = total_loss + 0.75*mse_loss

        if prototypes is not None and len(prototypes) > 1:
            if "target" in data_batch:
                current_target = data_batch["target"]
            else:
                current_target = target.to(self.device)

            # a. 提取所有的原型键，并排序保证顺序一致
            proto_keys = sorted(list(prototypes.keys()))
            proto_tensor = torch.stack([prototypes[k] for k in proto_keys]).to(self.device)

            # b. 特征与原型做 L2 归一化
            norm_feature = F.normalize(feature, p=2, dim=1)
            norm_proto = F.normalize(proto_tensor, p=2, dim=1)

            # c. 计算余弦相似度矩阵
            temperature = getattr(self.conf, 'proto_temp', 0.5) 
            sim_matrix = torch.matmul(norm_feature, norm_proto.T) / temperature

            # d. 制作目标标签映射 (因为交叉熵的 target 必须是 0 到 C-1 的连续索引)
            target_to_idx = {cls_name: idx for idx, cls_name in enumerate(proto_keys)}
            
            # 过滤掉万一没计算出原型的异常类 (稳健性设计)
            valid_mask = torch.tensor([t.item() in target_to_idx for t in current_target], dtype=torch.bool, device=self.device)
            
            if valid_mask.any():
                valid_sim_matrix = sim_matrix[valid_mask]
                valid_targets = current_target[valid_mask]
                
                mapped_targets = torch.tensor(
                    [target_to_idx[t.item()] for t in valid_targets], 
                    device=self.device
                )

                # e. 计算交叉熵损失 (拉近同类，推远异类)
                contrastive_loss = F.cross_entropy(valid_sim_matrix, mapped_targets)

                # f. 累加总损失
                proto_weight = getattr(self.conf, 'proto_contrastive_weight', 1.0)
                proto_weight = 0
                total_loss += proto_weight * contrastive_loss

        return total_loss

    def add_grad(self):
        pass

    def get_proto(self):
        """
        遍历所有批次，提取特征并计算每个类别的原型。
        """
        # 建议在提取特征时开启 eval 模式，避免影响 BatchNorm 的统计量
        self.model.eval() 
        class_features = {}
        
        with torch.no_grad():
            for _input, _target in self.train_loader:
                # 复用你的 load_data_batch 逻辑，这里 is_training 设为 False
                data_batch = create_dataset.load_data_batch(
                    self.conf, _input, _target, is_training=False, device=self.device
                )
                
                # 获取特征
                _, _, _, feature = self._inference(data_batch)
                current_target = data_batch.get("target", _target).to(self.device)
                
                # 按类别收集特征
                for i in range(len(current_target)):
                    label = current_target[i].item()
                    if label not in class_features:
                        class_features[label] = []
                    class_features[label].append(feature[i].detach())
                    
        prototypes = {}
        # 计算每个类别的均值作为原型
        for label, feats in class_features.items():
            # feats 列表转换为 Tensor: [N, feature_dim]
            prototypes[label] = torch.stack(feats).mean(dim=0)
            
        # 恢复训练模式
        self.model.train()
        return prototypes


    def _train(self):
        self.prepare_train() # 初始化指标追踪器，优化器等
        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:
            prototypes = self.get_proto()
            for _input, _target in self.train_loader:

                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_): # 数据统计
                    data_batch = create_dataset.load_data_batch(    # 这里主要是将数据搬到cuda上
                        self.conf, _input, _target, is_training=True, device=self.device
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    loss, performance, output, feature = self._inference(data_batch)    # 前向传播

                with self.timer("extra_forward_pass", epoch=self.scheduler.epoch_):
                    loss = self.local_training_with_extra_calculate(loss, output, data_batch, feature=feature,target = _target, prototypes = prototypes)  # 计算L2损失
                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10) # 梯度裁剪，max_norm为允许的最大梯度

                    self.optimizer.step()
                    self.scheduler.step()

                # check divergence.检查loss是否过大，防止将有毒参数上传
                if loss > 1e4 or torch.isnan(loss):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )   # 报告哪个客户端出错

                    self._terminate_comm_round()    # 终止训练
                    return

                # check stopping condition.
                if self._is_finished_one_comm_round(): # 检查轮数，如果够20轮直接返回

                    self.conf.logger.log(self.timer.summary())

                    self._terminate_comm_round()
                    return

            # # display tracking time.
            # if (
            #         self.conf.display_tracked_time
            #         and self.scheduler.local_index % self.conf.summary_freq == 0
            # ):


            # display the logging info.
            display_training_stat(self.conf, self.scheduler, self.tracker) # 打印这一轮的成功
            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        feature,output = self.model(data_batch["input"]) 

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            bsz = data_batch["target"].size(0)
            self.tracker.update_local_metrics(
                loss.item(), 0, n_samples=bsz
            )
            for idx in range(1, 1 + len(performance)):
                self.tracker.update_local_metrics(
                    performance[idx - 1], idx, n_samples=bsz
                )
        return loss, performance, output, feature

    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        return model

    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))    # 打包数据
        dist.send(tensor=flatten_model.buffer, dst=0)   # 发送
        dist.barrier()      # 等待其他客户端

    def _terminate_comm_round(self):
        self.model = self.model.cpu()

        self.scheduler.clean()
        self.conf.logger.save_json()
        torch.cuda.empty_cache()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.conf.local_n_epochs else False


class DummyLayer:
    def __init__(self, dtype):
        self.weight = type('DummyTensor', (), {'dtype': dtype})()

class DummyVisual:
    def __init__(self, dtype):
        self.conv1 = DummyLayer(dtype)