import torch
import random
import numpy as np
import os

def set_seed(seed, rank, world_size):
    rng = random.Random(seed)
    seed_per_rank = [rng.randint(0, 2**32-1) for _ in range(world_size)]
    cur_seed = seed_per_rank[rank]

    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed(cur_seed)
    torch.cuda.manual_seed_all(cur_seed)
    np.random.seed(cur_seed)
    random.seed(cur_seed)
    #torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(cur_seed)
