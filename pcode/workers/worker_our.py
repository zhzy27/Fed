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

        self.text_model, _, self.output_dim = self.load_clip_text_model()
        conf.output_dim = self.output_dim 
        self.conf = conf

        self.anchor = self.generate_text_anchors()
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
        output_dim = model.text_projection.shape[1]
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

        return model, device, output_dim
    

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
        
        print(f"Generating {len(prompts)} text anchors for dataset: {dataset_name}...")

        # 4. Tokenize
        # clip.tokenize 会自动截断或填充到 77 token 长度
        text_inputs = clip.tokenize(prompts).to(device)
        
        # 5. 提取特征 (Inference)
        # 确保不计算梯度，确保模型处于 eval 模式
        self.text_model.eval()
        with torch.no_grad():
            # encode_text 是 CLIP 模型提取文本特征的标准方法
            text_features = self.text_model.encode_text(text_inputs)
            
            # [重要] 归一化
            # CLIP 的特征必须做 L2 归一化，因为它是基于余弦相似度训练的
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        # text_features shape: [num_classes, feature_dim] (e.g., [10, 512] for cifar10)
        return text_features

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
        """
        计算额外的损失项。
        包含严格的调试检查：如果缺少 feature, anchor 或 target，直接报错。
        """
        
        # 1. 计算原有的 L2 正则损失
        # 这一步通常不会出错，所以先算出来
        l2_loss = self.conf.meta_L2 * self.model.L2_decay()
        total_loss = loss + l2_loss

        # ==========================================================
        # 严格调试模式：检查所有必要条件
        # ==========================================================

        # [检查 1] 检查 feature 是否传入
        if feature is None:
            raise ValueError(
                "[Debug Error] 'feature' is None! \n"
                "请检查: \n"
                "1. self._inference() 是否正确返回了 feature？\n"
                "2. self._train() 调用此函数时是否传入了 feature=feature？"
            )

        # [检查 2] 检查 anchor 是否存在
        if not hasattr(self, 'anchor') or self.anchor is None:
            raise RuntimeError(
                "[Debug Error] 'self.anchor' not found or is None! \n"
                "请检查: \n"
                "1. 是否调用了 self.generate_text_anchors()？\n"
                "2. 是否成功执行了 self.register_buffer('anchor', ...)? \n"
                "3. 确保 dataset name 正确且在生成列表里。"
            )

        # 对feature进行归一化
        feature_norm = F.normalize(feature, p=2, dim=1)
        
        # [检查 3] 检查 Target (标签) 是否存在
        # 我们需要标签来从 10 个锚点里挑出正确的那 1 个
        current_target = None
        if "target" in data_batch:
            current_target = data_batch["target"]
        elif target is not None:
            current_target = target.to(feature_norm.device)
        
        if current_target is None:
            raise ValueError(
                "[Debug Error] No target label found! \n"
                "无法找到类别标签，不知道该匹配哪个锚点。\n"
                "请检查 data_batch['target'] 是否存在，或是否显式传入了 target 参数。"
            )

        # [检查 4] 检查设备一致性 (可选，但推荐)
        if feature_norm.device != self.anchor.device:
            # 尝试自动修复，或者报错
            # 这里为了调试，如果不在一个设备上可能导致严重性能问题，这里选择自动迁移但打印警告，或者直接迁移
            self.anchor = self.anchor.to(feature_norm.device)

        # ==========================================================
        # 计算 Anchor Loss
        # ==========================================================
        
        # 1. 选取对应锚点
        # 如果 label 越界 (例如 cifar10 出现了 label 10)，这里会报 PyTorch IndexError
        try:
            target_anchors = self.anchor[current_target]
        except IndexError as e:
            raise IndexError(
                f"[Debug Error] Label index out of range! \n"
                f"self.anchor shape: {self.anchor.shape}, "
                f"Max label in batch: {current_target.max()}. \n"
                f"Original Error: {e}"
            )
        target_anchors = target_anchors.to(feature.dtype)
        # 2. 计算 MSE
        # feature: [B, Dim], target_anchors: [B, Dim]
        anchor_loss_val = F.mse_loss(feature_norm, target_anchors.detach())
        
        # 3. 加权
        total_loss += self.conf.anchor_loss * anchor_loss_val

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