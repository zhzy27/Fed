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

        self.text_model, _ = self.load_clip_text_model()
        self.anchor = self.generate_text_anchors()
        self.output_dim = self.anchor[0].shape[-1] 
        
        print(f"Anchors generated. Count: {len(self.anchor)}, Dim: {self.output_dim}")
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
    

    def generate_text_anchors(self):
        """
        根据数据集名称生成文本锚点 (Text Anchors)。
        
        改进策略: Hierarchical Prompts (层级提示词)
        不再截取 Transformer 中间层，而是使用 4 组不同抽象层级的 Prompt，
        分别对应 ResNet 的 4 个 Stage。
        """
        
        # 1. 获取当前设备
        # 假设 self.text_model 已经在正确的 device 上
        device = next(self.text_model.parameters()).device
        
        # 2. 定义数据集与其对应的类别列表
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
        }

        dataset_name = self.conf.data.lower()
        if dataset_name not in data_classes_registry:
            raise ValueError(f"Dataset '{dataset_name}' not found. Please add classes.")
        
        classes = data_classes_registry[dataset_name]
        
        # ==============================================================================
        # [核心修改] 定义 4 个阶段的层级化提示词模板
        # 对应 ResNet 的 Stage 1 (底层) -> Stage 4 (高层)
        # ==============================================================================
        hierarchical_templates = [
            # Stage 1: 关注底层视觉特征 (纹理、边缘、颜色)
            "elementary visual features, edges, and textures of a {}",
            
            # Stage 2: 关注中层特征 (部件、模式)
            "visual patterns, parts, and components of a {}",
            
            # Stage 3: 关注高层几何特征 (形状、结构)
            "the visual shape, geometry, and structure of a {}",
            
            # Stage 4: 关注完整语义 (标准的分类描述)
            "a photo of a {}"
        ]

        print(f"Generating hierarchical text anchors for {dataset_name} using 4-stage prompts...")
        
        anchors_list = []
        
        # 确保模型处于评估模式
        self.text_model.eval()
        
        with torch.no_grad():
            # 遍历 4 个模板，生成 4 组锚点
            for i, template in enumerate(hierarchical_templates):
                # 3. 构造当前阶段的 Prompt
                prompts = [template.format(c) for c in classes]
                
                # 4. Tokenize
                text_inputs = clip.tokenize(prompts).to(device)
                
                # 5. 提取特征 (使用完整的 encode_text)
                # 注意：这里我们不需要手动通过 layers 了，直接用 encode_text 获取最终对齐特征
                # 因为我们是通过 Prompt 的含义来区分层级的，而不是通过网络深度
                
                # 为了防止之前提到的 'dtype' 报错 (如果 visual 删的不干净)，这里手动模拟 encode_text 流程最稳妥
                # 流程: Embedding -> Transformer -> LayerNorm -> Projection
                
                # A. Embedding
                x = self.text_model.token_embedding(text_inputs).type(self.text_model.dtype)
                x = x + self.text_model.positional_embedding.type(self.text_model.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND

                # B. Transformer (完整通过所有层)
                x = self.text_model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                
                # C. LayerNorm & Take EOS feature
                x = self.text_model.ln_final(x).type(self.text_model.dtype)
                
                # 取出 [EOS] token 对应的特征
                sent_emb = x[torch.arange(x.shape[0]), text_inputs.argmax(dim=-1)]
                
                # D. Projection (映射到与图像对齐的空间)
                if self.text_model.text_projection is not None:
                    sent_emb = sent_emb @ self.text_model.text_projection

                # 6. 归一化 (L2 Normalize)
                # 这一步至关重要，CLIP 是基于 Cosine Similarity 的
                sent_emb = sent_emb / sent_emb.norm(dim=-1, keepdim=True)
                
                # 7. 加入列表 (转为 float32 方便后续计算)
                anchors_list.append(sent_emb.float())
                
                # print(f"  - Stage {i+1} anchors generated. Template: '{template.format('CLASS')}'")

        # 返回包含 4 个 Tensor 的列表，每个 Tensor 形状 [Num_Classes, Dim]
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
    def extra_init(self):
        pass

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

        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values())) # 将参数打包为一维参数，方便传输

        self.extra_init()

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

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature=None, target=None):
        # 1. 基础损失
        total_loss = loss + self.conf.meta_L2 * self.model.L2_decay()

        # 2. 计算层级锚点损失
        # 检查是否开启对齐 (可以通过 output 里的 stage_features 是否为空，或者 check use_align)
        # 注意：这里需要确保你已经正确解包了 output，例如 x, logits = output
        # 或者你现在的 output 就是 logits，而 stage_features 存在 self.model.stage_features 里
        
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
            anchor_loss_scalar = torch.sum(loss_vec) 
            
            total_loss += self.conf.anchor_loss * anchor_loss_scalar

        return total_loss

    def add_grad(self):
        pass

    def _train(self):
        self.prepare_train() # 初始化指标追踪器，优化器等
        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:

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
                    loss = self.local_training_with_extra_calculate(loss, output, data_batch, feature=feature,target = _target)  # 计算L2损失
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