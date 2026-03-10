# -*- coding: utf-8 -*-
import os
import copy

import numpy as np
import torch
import torch.distributed as dist
from pcode.utils.module_state import ModuleState
import pcode.master_utils as master_utils
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.cross_entropy as cross_entropy
from pcode.utils.early_stopping import EarlyStoppingTracker
import torch.nn.utils as torch_utils
from pcode.models.resnet import MetaBasicBlock
from torch import nn
import clip

class MasterFedOur(object):
    def __init__(self, conf):
        self.conf = conf
        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))

        assert conf.meta == True

        # create model as well as their corresponding state_dicts.
        # 获取未分解的模型
        # _, self.master_model = create_model.define_model( # 获取全局模型
        #     conf, to_consistent_model=False
        # )
        # 恢复全局模型

        self.text_model, _ = self.load_clip_text_model()
        self.anchor = self.generate_text_anchors()
        self.output_dim = self.anchor[0].shape[-1] 
        conf.output_dim = self.output_dim
        self.conf = conf

        self.used_client_archs =[create_model.determine_arch(conf, client_id, use_complex_arch=True) for client_id in range(1, 1 + conf.n_clients)]  # 所有客户端模型架构名称
        
        
        self.conf.used_client_archs = self.used_client_archs

        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")

        self.client_models = [ # 获取客户端模型
            create_model.define_model(conf, to_consistent_model=False, client_id=i, arch=arch)
            for i,arch in enumerate(self.used_client_archs)
        ]

        # 拷贝构建全局模型

        self.master_model = copy.deepcopy(self.client_models[0][1])
        if "resnet" in self.conf.arch:
            for m in self.master_model.modules():
                    if isinstance(m, MetaBasicBlock):
                        m.recover()
        elif "cnn" in self.conf.arch:
            self.master_model.recover_model()

        # self.decom_recover_loss()
        self.clientid2arch = list( # 获取所有客户端的模型名称
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(0, conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch

        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
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
        ) # 虽然划分了数据但是全丢了，只保留数据划分器



        conf.logger.log(f"Master initialized the local training data with workers.")

        # create val loader.
        # right now we just ignore the case of partitioned_by_user.
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None

        # create test loaders.
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        if conf.partitioned_by_user:
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id - 1,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            self.test_loaders = [test_loader] # 这里的test_loader和上面的val_loader都是总的，也就是说未平分给客户端的

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")   # 模型的损失函数，这里传入mean表示计算的损失会求平均
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")  # 模型的评估，计算准确率之类
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)   # 记录最佳性能
        if not self.conf.train_fast:
            self.local_coordinator = [create_coordinator.Coordinator(conf, self.metrics) for _ in
                                      range(self.conf.n_clients)]

        conf.logger.log(f"Master initialized the coordinator.\n")

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )   # 早停装置，为0时关掉早停功能

        # save arguments to disk.
        conf.is_finished = False    # 用于控制全局训练是否已经结束
        checkpoint.save_arguments(conf) # 将超参数保存在json文件中



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

        return model, device

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

### 只有在resnet下才能用这个函数

    # def decom_recover_loss(self):
    #     self.master_model.eval()
    #     self.master_model.to('cuda')
    #     old_state_dict = copy.deepcopy(self.master_model.state_dict())


    #     for m in self.master_model.modules():
    #             if isinstance(m, MetaBasicBlock):
    #                 m.decom(100000000)


    #     self.master_model.to('cuda')
    #     for m in self.master_model.modules():
    #             if isinstance(m, MetaBasicBlock):
    #                 m.recover()
        
    #     new_state_dict = copy.deepcopy(self.master_model.state_dict())
        
    #     total_diff = 0.0
    #     max_diff = 0.0

    #     for key in old_state_dict:
    #         # 只比较卷积层权重，跳过 num_batches_tracked 等统计量
    #         if 'weight' in key or 'bias' in key:
    #             w_old = old_state_dict[key].float()
    #             # 确保 new_state_dict 里有这个 key (因为分解层结构变了又变回来，key 应该一致)
    #             if key in new_state_dict:
    #                 w_new = new_state_dict[key].float()
    #                 # 计算两个张量的差异 (L1 距离)
    #                 diff = (w_old - w_new).abs().sum().item()
    #                 # 记录最大元素级误差
    #                 current_max = (w_old - w_new).abs().max().item()
                    
    #                 total_diff += diff
    #                 max_diff = max(max_diff, current_max)
    #             else:
    #                 print(f"⚠️ 警告: Key {key} 在还原后的模型中丢失！")

    #     self.master_model.to('cpu')
        
    #     print(f"📊 检查结果:")
    #     print(f"   累积总误差 (Sum Abs Diff): {total_diff:.4f}")
    #     print(f"   最大单点误差 (Max Abs Diff): {max_diff:.6f}")
    #     # 6. 自动判断
    #     if max_diff < 1e-3:
    #         print("✅ 成功: 还原误差极小。SVD 维度变换逻辑正确！")
    #     else:
    #         print("❌ 失败: 还原误差过大！")
    #         print("   原因可能是：")
    #         print("   1. decom/recover 里的 permute/view 维度搞反了 (最可能)。")
    #         print("   2. decom 时传入的 Rank 太小，导致信息被大量截断。")
    #     print("="*40 + "\n")

    #     self.master_model.to('cpu')

    def run(self):

        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )   # 生成参与数长的列表，每个值为每个客户端的训练轮数
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.
            selected_client_ids = self._random_select_clients() # 随机选择客户端

            # detect early stopping.
            self._check_early_stopping() # 检查是否满足早停，可能未开启早停

            # init the activation tensor and broadcast to all clients (either start or stop).
            self.activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
            )

            # will decide to send the model or stop the training.
            if not self.conf.is_finished:
                self.send_extra_info_to_selected_clients(selected_client_ids)

                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids)

            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return

            self.receive_extra_info_from_selected_clients(selected_client_ids)

            # wait to receive the local models.
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            ) # 接受模型参数

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model_and_evaluate(flatten_local_models, selected_client_ids)   # 聚合模型，评估和存档

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()
        self._finishing()

    def receive_extra_info_from_selected_clients(self, selected_client_ids):
        pass


    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()  # 随机选择客户端
        selected_client_ids.sort()
        ids = [selected_client_id-1 for selected_client_id in selected_client_ids]
        selected_client_ids = ids
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs, to_send_history=False
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        msg_len = 3 # 消息行数

        activation_msg = torch.zeros((msg_len, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids) # 第一行装客户端id
        activation_msg[1, :] = comm_round                       # 第二行装联邦轮次
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)     # 第三行装每个客户端训练轮数

        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")

        
        self.updata_selected_clients_models(selected_client_ids)

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id] # 获取当前循环模型名称
            client_model_state_dict = self.client_models[selected_client_id][1].state_dict()

            flatten_model = TensorBuffer(list(client_model_state_dict.values())) # 打包模型参数
            dist.send(tensor=flatten_model.buffer, dst=worker_rank) # 发送参数到对应模型
            self.conf.logger.log(
                f"\tMaster send the current model={arch} to process_id={worker_rank}."
            )

        dist.barrier()

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        pass

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        for selected_client_id in selected_client_ids: # 为接受数据准备容器，这也可以解释为什么要在服务器创建模型
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[selected_client_id][1].state_dict().values())
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )   # 这里并不会阻塞，相当于告诉客户端数据放在哪里
            reqs.append(req)

        for req in reqs:
            req.wait()  # 这里会阻塞等待数据

        dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models

    def _fedavg(self, flatten_local_models, weights=None):
        n_selected_clients = len(flatten_local_models)

        if weights == None:
            weights = [
                torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
            ]

        # NOTE: the arch for different local models needs to be the same as the master model.
        # retrieve the local models.
        local_models = {}
        for client_idx, flatten_local_model in flatten_local_models.items():
            _arch = self.clientid2arch[client_idx]
            _model = copy.deepcopy(self.client_models[_arch])
            _model_state_dict = self.client_models[_arch].state_dict()
            flatten_local_model.unpack(_model_state_dict.values())
            _model.load_state_dict(_model_state_dict)
            local_models[client_idx] = _model

        # uniformly average the local models.
        # assume we use the runtime stat from the last model.
        _model = copy.deepcopy(_model)
        local_states = [
            ModuleState(copy.deepcopy(local_model.state_dict()))
            for _, local_model in local_models.items()
        ]
        model_state = local_states[0] * weights[0]
        for idx in range(1, len(local_states)):
            model_state += local_states[idx] * weights[idx]
        model_state.copy_to_module(_model)
        return _model

    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )

            fedavg_model = self._fedavg(flatten_local_models) # 平均算法
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models


    def aggregate(self, flatten_local_models, selected_client_ids):
        # uniformly average local models with the same architecture.
        self.load_para2selectedmodels(flatten_local_models, selected_client_ids)    # 把收到参数放入self.clientmodels模型中
        
        fedavg_models = self.recover_aggrevate(selected_client_ids)
        return fedavg_models

    def updata_selected_clients_models(self, selected_client_ids):
        # 注意检查变量名：之前你用的是 conf.list_rank 还是 conf.rank_list？
        # 假设之前保存的是 conf.list_rank
        rank_list = self.conf.rank_list 
        
        # 获取 Master 参数（放在 CPU 上以节省显存，如果都在 GPU 则不需要 .cpu()）
        master_model_state_dict = self.master_model.state_dict()

        for selected_client_id in selected_client_ids:
            model = self.client_models[selected_client_id][1] # 取出模型对象
            model.to('cuda')
            # -----------------------------------------------------------
            # 【步骤 1：结构对齐】
            # 在加载参数前，必须先检查模型是否处于分解状态。
            # 如果是分解状态（FactorizedConv），必须先 recover 回 Conv2d，
            # 否则 load_state_dict 会因为 Key 不匹配或形状不匹配而报错。
            # -----------------------------------------------------------
            if "resnet" in self.conf.arch:
                for m in model.modules():
                    if isinstance(m, MetaBasicBlock):
                        # 判断依据：如果 conv1 不是标准的 Conv2d，说明它是分解过的 FactorizedConv
                        if not isinstance(m.conv1, nn.Conv2d):
                            m.recover()
            elif "cnn" in self.conf.arch:
                if model.meta:
                    model.recover_model()
            model.to('cuda')
            # -----------------------------------------------------------
            # 【步骤 2：加载全量参数】
            # 现在 model 结构和 master 一模一样了，可以安全加载
            # -----------------------------------------------------------
            model.load_state_dict(master_model_state_dict)

            # -----------------------------------------------------------
            # 【步骤 3：按需分解】
            # 根据该客户端特定的 rank 进行 SVD 分解
            # -----------------------------------------------------------
            current_rank = rank_list[selected_client_id] # 获取该客户端对应的 rank (或 ratio)
            if "resnet" in self.conf.arch:
                for m in model.modules():
                    if isinstance(m, MetaBasicBlock):
                        # 这里的 decom 内部会执行 SVD 并把 Conv2d 替换为 FactorizedConv
                        m.decom(current_rank)
            elif "cnn" in self.conf.arch:
                model.decom_model(current_rank)
            model.to('cpu')


    # def recover_aggrevate(self, selected_client_ids):
    #     """
    #     1. 遍历 selected_client_ids 中的每个模型。
    #     2. 找到模型中所有的 MetaBasicBlock，调用其 recover() 方法将其恢复为标准卷积。
    #     3. 对恢复后的模型进行参数平均聚合。
    #     4. 返回聚合后的新模型。
    #     """
        
    #     # --- 第一步：恢复所有选中的模型 ---
    #     # 注意：这里会直接修改 self.client_models 中存储的模型对象结构
    #     for client_id in selected_client_ids:
    #         # 获取模型对象 (记得取元组的第2个元素)
    #         model = self.client_models[client_id][1]
            
    #         # 使用 modules() 递归遍历所有子模块，确保涵盖 body 和 personalized 中的所有块
    #         # 这里的 MetaBasicBlock 需要确保你的代码环境中能访问到该类定义
    #         for m in model.modules():
    #             if isinstance(m, MetaBasicBlock):
    #                 # 调用你提供的 recover 方法，它会将 FactorizedConv 替换回 Conv2d
    #                 m.recover()

    #     if not self.conf.train_fast:  # test all the selected_clients
    #         for client_idx in selected_client_ids:
    #             # _arch = self.clientid2arch[client_idx]
    #             # _model_state_dict = copy.deepcopy(self.client_models[client_idx][1].state_dict())
    #             # flatten_local_model.unpack(_model_state_dict.values())
    #             # real_arch = _arch[1] if isinstance(_arch, tuple) else _arch
    #             # _, test_model = create_model.define_model(self.conf, to_consistent_model=False, client_id=client_idx , arch=real_arch)
    #             # test_model.load_state_dict(_model_state_dict)
    #             test_model = copy.deepcopy(self.client_models[client_idx][1])
    #             master_utils.do_validation(
    #                 conf=self.conf,
    #                 coordinator=self.local_coordinator[client_idx],
    #                 model=test_model,
    #                 criterion=self.criterion,
    #                 metrics=self.metrics,
    #                 data_loaders=self.test_loaders,
    #                 label=f"aggregated_test_loader_{client_idx}",
    #             )
    #         self.additional_validation()
    #     # --- 第二步：初始化聚合容器 ---
    #     # 选取第一个模型作为聚合的“底板”
    #     base_client_id = selected_client_ids[0]
    #     base_model = self.client_models[base_client_id][1]
        
    #     # 深拷贝一份 state_dict 用于累加，避免修改原模型数据
    #     global_params = copy.deepcopy(base_model.state_dict())
        
    #     # 将容器清零，准备累加
    #     for key in global_params:
    #         global_params[key].zero_()

    #     # --- 第三步：累加所有模型的参数 ---
    #     for client_id in selected_client_ids:
    #         model = self.client_models[client_id][1]
    #         local_params = model.state_dict()
            
    #         for key in global_params:
    #             # 累加参数
    #             # 注意：需保证所有模型在同一设备上 (CPU/GPU)
    #             global_params[key] += local_params[key]

    #     # --- 第四步：取平均 ---
    #     num_models = len(selected_client_ids)
    #     for key in global_params:
    #         # 区分浮点数参数和整数参数 (如 BatchNorm 的 num_batches_tracked)
    #         if global_params[key].is_floating_point():
    #             global_params[key] /= num_models
    #         else:
    #             val_float = global_params[key].float() / num_models
    #             # 2. 根据该参数原本的类型，决定如何赋值回去
    #             if global_params[key].is_floating_point():
    #                 # 如果原本就是浮点数 (如 weight, bias)，直接赋值
    #                 global_params[key].copy_(val_float)
    #             else:
    #                 # 如果原本是整数 (如 num_batches_tracked)，需要四舍五入后转回整数
    #                 # 使用 .round() 避免地板除的偏差，然后转为 .long()
    #                 global_params[key].copy_(torch.round(val_float).long())

    #     # --- 第五步：构建返回的模型对象 ---
    #     # 我们深拷贝一个已经恢复结构的模型作为载体
    #     aggregated_model = copy.deepcopy(base_model)
    #     # 加载计算好的平均参数
    #     aggregated_model.load_state_dict(global_params)

    #     return aggregated_model

    def recover_aggrevate(self, selected_client_ids):
        """
        1. 遍历 selected_client_ids 中的每个模型。
        2. 找到模型中所有的 MetaBasicBlock，调用其 recover() 方法将其恢复为标准卷积。
        3. 对恢复后的模型进行参数平均聚合。
        4. 返回聚合后的新模型。
        """
        
        # --- 第一步：恢复所有选中的模型 ---
        # 注意：这里会直接修改 self.client_models 中存储的模型对象结构
        # 这一步必须保留，确保聚合的是完整卷积核 W，而不是分解因子 U/V
        for client_id in selected_client_ids:
            # 获取模型对象 (记得取元组的第2个元素)
            model = self.client_models[client_id][1]
            
            # 使用 modules() 递归遍历所有子模块，确保涵盖 body 和 personalized 中的所有块
            # 这里的 MetaBasicBlock 需要确保你的代码环境中能访问到该类定义
            if "resnet" in self.conf.arch:
                for m in model.modules():
                    if isinstance(m, MetaBasicBlock):
                        # 调用你提供的 recover 方法，它会将 FactorizedConv 替换回 Conv2d
                        m.recover()
            elif "cnn" in self.conf.arch:
                model.recover_model()

        # --- 第二步：聚合逻辑 (替换为 ModuleState 方式) ---
        
        # 1. 定义权重 (Origin逻辑: 默认为均匀平均)
        n_selected_clients = len(selected_client_ids)
        weights = [1.0 / n_selected_clients for _ in range(n_selected_clients)]

        # 2. 将所有选中的客户端模型状态封装为 ModuleState 对象
        # 这完全对应 Origin 代码中的 local_states = [ModuleState(...) for ...]
        local_states = []
        for client_id in selected_client_ids:
            # Modified 版本中模型存储在 list 的元组里，取 [1]
            local_model = self.client_models[client_id][1]
            # 关键：使用 deepcopy 确保数据独立，防止引用修改
            # 注意：此时模型已经是 recover() 过的状态，所以提取的是完整的卷积参数
            local_states.append(ModuleState(copy.deepcopy(local_model.state_dict())))

        # 3. 执行加权聚合 (完全照搬 Origin 的数学逻辑)
        # model_state = state[0] * w[0] + state[1] * w[1] + ...
        # ModuleState 内部全精度浮点运算，避免了整数地板除的问题
        model_state = local_states[0] * weights[0]
        
        for idx in range(1, len(local_states)):
            model_state += local_states[idx] * weights[idx]

        # 4. 将聚合后的状态复制回模型对象
        # 取第一个客户端的模型作为结构底板 (template)
        base_client_id = selected_client_ids[0]
        # 注意：这里深拷贝的 base_model 已经是 recover 过的结构（标准卷积）
        aggregated_model = copy.deepcopy(self.client_models[base_client_id][1])
        
        # 使用 ModuleState 自带的 copy_to_module 方法
        # 它会自动处理类型转换 (float -> long) 和设备放置
        model_state.copy_to_module(aggregated_model)

        return aggregated_model

    def load_para2selectedmodels(self, flatten_local_models, selected_client_ids):
        for client_id in selected_client_ids:
            # 获取对应的模型对象 (注意：需确认 self.client_models 的索引方式是否正确)
            # 如果 self.client_models 是列表且长度不够，这里可能会报错，请确保初始化时为每个 client_id 都预留了位置
            if isinstance(self.client_models, list):
                # 假设 client_id 是 1-based，列表是 0-based
                target_model = self.client_models[client_id ][1] 
            elif isinstance(self.client_models, dict):
                target_model = self.client_models[client_id][1]
            else:
                # 根据您的实际结构调整
                target_model = self.client_models[client_id][1]

            # 【修复重点】使用 state_dict().values() 接收所有参数（含 BN 统计量）
            # 获取 buffer 对象
            client_buffer = flatten_local_models[client_id]
            
            # 必须先获取 state_dict 的引用
            target_state_dict = target_model.state_dict()
            
            # 使用 TensorBuffer 自带的 unpack 方法填充 state_dict 的 values
            client_buffer.unpack(target_state_dict.values())
            
            # 将更新后的 state_dict 重新加载回模型（确保数据生效）
            target_model.load_state_dict(target_state_dict)



    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        # aggregate the local models.
        self.selected_client_ids = selected_client_ids
        aggregated_model = self.aggregate(
            flatten_local_models,
            selected_client_ids
        ) # 计算平均后的模型

        client_models = {0: aggregated_model}

        self.master_model.load_state_dict(
            list(client_models.values())[0].state_dict()


        
        )
                
         # 更新全局模型
        # for arch, _client_model in client_models.items():
        #     self.client_models[arch].load_state_dict(_client_model.state_dict())

        # for arch, _client_model in client_models.items():
        #     # arch 现在是 0
        #     if arch in self.client_models:
        #         target = self.client_models[arch]
                
        #         # 【修复重点】判断是否为元组，如果是，取第2个元素
        #         if isinstance(target, tuple):
        #             target[1].load_state_dict(_client_model.state_dict())
        #         else:
        #             target.load_state_dict(_client_model.state_dict())

        # evaluate the aggregated model on the test data.
        master_utils.do_validation( # 最终评估
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label=f"aggregated_test_loader_0",
        )

        # self.conf.train_fast=False
        
        if not self.conf.train_fast:  # test all the selected_clients
            for client_idx in selected_client_ids:
                # _arch = self.clientid2arch[client_idx]
                # _model_state_dict = copy.deepcopy(self.client_models[client_idx][1].state_dict())
                # flatten_local_model.unpack(_model_state_dict.values())
                # real_arch = _arch[1] if isinstance(_arch, tuple) else _arch
                # _, test_model = create_model.define_model(self.conf, to_consistent_model=False, client_id=client_idx , arch=real_arch)
                # test_model.load_state_dict(_model_state_dict)
                test_model = copy.deepcopy(self.client_models[client_idx][1])
                master_utils.do_validation(
                    conf=self.conf,
                    coordinator=self.local_coordinator[client_idx],
                    model=test_model,
                    criterion=self.criterion,
                    metrics=self.metrics,
                    data_loaders=self.test_loaders,
                    label=f"aggregated_test_loader_{client_idx}",
                )
            self.additional_validation()


        torch.cuda.empty_cache()
    def additional_validation(self):
        pass

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:   # 是否设置目标性能
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
    
class DummyLayer:
    def __init__(self, dtype):
        self.weight = type('DummyTensor', (), {'dtype': dtype})()

class DummyVisual:
    def __init__(self, dtype):
        self.conv1 = DummyLayer(dtype)