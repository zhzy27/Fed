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


class WorkerFedHM(object):
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

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature = None):
        return loss + 0.0001*self.model.L2_decay()

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
                    loss, performance, output = self._inference(data_batch)    # 前向传播

                with self.timer("extra_forward_pass", epoch=self.scheduler.epoch_):
                    loss = self.local_training_with_extra_calculate(loss, output, data_batch)  # 计算L2损失
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
        _,output = self.model(data_batch["input"]) 

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
        return loss, performance, output

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


