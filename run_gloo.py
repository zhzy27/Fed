import torch.multiprocessing as mp
# -*- coding: utf-8 -*-
from parameters import get_args
from pcode.masters import *
from pcode.workers import *
import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser
import random

MethodTable = {
    "fedavg": [Master, Worker],
    "fedgen": [MasterFedgen, WorkerFedGen],
    "feddistill": [MasterFedDistill, WorkerFedDistill],
    "moon": [MasterMoon, WorkerMoon],
    "fedgkd": [MasterFedGKD, WorkerFedGKD],
    "fedprox": [Master, WorkerFedProx],
    "feddyn":[MasterFedDyn, WorkerFedDyn],
    "fedadam":[MasterFedAdam, Worker],
    "fedadam_gkd":[MasterFedAdam, WorkerFedGKD],
    "fedensemble":[MasterFedEnsemble, Worker],
    "fedhm":[MasterFedHM, WorkerFedHM],
}

def random_rank_creator(conf):
    # 定义候选的 rank 值
    candidates = [0.8]
    
    # 从候选列表中随机采样，size 指定生成数量
    # replace=True (默认) 表示允许重复选择同一个值
    return np.random.choice(candidates, size=conf.n_clients)
    # return np.random.uniform(low=0.1, high=1.0, size=conf.n_clients)

def main(rank, size, conf, port): # rank 为当前进程的序号
    # init the distributed world.
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group("gloo", rank=rank, world_size=size) # 这是一个同步阻塞函数，会等待其他子进程，并且建立通信
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    init_config(conf)
    


    assert MethodTable[conf.method] is not None
    master, worker = MethodTable[conf.method] # 取别名Master, Worker

    # start federated learning.
    process = master(conf) if conf.graph.rank == 0 else worker(conf) # 0号为服务器，其他的为client
    process.run()


def init_config(conf):
    # define the graph for the computation.
    conf.graph = topology.define_graph_topology( # 设置GPU序列，即哪个进程在哪个GPU上跑
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank() # 获取进程序号

    # init related to randomness on cpu.
    if not conf.same_seed_process:  # 判断是否需要设置随机种子
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank # 这里设置不同随机种子

    os.environ["TOKENIZERS_PARALLELISM"] = "true"   # 屏蔽一些警告
    #os.environ['PYTHONHASHSEED'] = str(conf.manual_seed)
    #random.seed(conf.manual_seed)
    #np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed) # 根据随机种子生成随机序列
    torch.manual_seed(conf.manual_seed) 
    init_cuda(conf)

    # init the model arch info.
    conf.arch_info = ( # 参数解析，{'master': 'resnet8', 'worker': 'resnet8'}
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":") # 切割成列表

    # define checkpoint for logging (for federated learning server).
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank)) # 初始化检查点路径

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    dist.barrier() # 同步进程


def init_cuda(conf):
    torch.cuda.set_device(torch.device("cuda:" + str(conf.graph.rank % torch.cuda.device_count())))
    torch.cuda.manual_seed(conf.manual_seed)
    #torch.cuda.manual_seed_all(conf.manual_seed)
    torch.backends.cudnn.enabled = True # 加速
    torch.backends.cudnn.benchmark = True   # 加速
    torch.backends.cudnn.deterministic = True # 使用确定性算法，确保复现


import time

if __name__ == "__main__":
    conf = get_args()
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5) # 向上取整计算参与数量
    conf.timestamp = str(int(time.time()))
    size = conf.n_participated + 1 # 这里是启动进程数量，客户端 + 一个服务器
    processes = []
    if conf.is_random_rank:
        conf.rank_list = random_rank_creator(conf)

    mp.set_start_method("spawn") # 设置子进程启动方式，子进程只会继承运行所需的资源（run 参数），它不会像 fork 那样直接复制父进程的所有内存空间。因此，子进程启动时会重新导入并执行代码中的模块。
    for rank in range(size):
        p = mp.Process(target=main, args=(rank, size, conf, conf.port)) # 创建子进程，指定启动函数为上方的main函数，并传递参数
        p.start()
        processes.append(p)

    for p in processes: # 主进程等待子进程
        p.join()
