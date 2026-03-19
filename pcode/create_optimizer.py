# -*- coding: utf-8 -*-
import torch


def define_optimizer(conf, model, optimizer_name, lr=None):
    # define the param to optimize.
    weight_decay = conf.weight_decay
    base_lr = conf.lr if lr is None else lr
    params = [  # BN层特殊处理，BN的仿射变换不需要L2惩罚
        {
            "params": [value],
            "name": key,
            "weight_decay": weight_decay if "bn" not in key else 0.0,
            "lr": base_lr * 10.0 if any(k in key for k in ["prompt_ctx", "clip_adapter", "logit_scale"]) else base_lr,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=conf.momentum_factor,
            nesterov=conf.use_nesterov, # 一种改进动量，不一定使用
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(conf.adam_beta_1, conf.adam_beta_2),
            eps=conf.adam_eps,
        )
    else:
        raise NotImplementedError
    return optimizer
