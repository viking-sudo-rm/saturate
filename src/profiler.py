from typing import List
import torch
import gc
from logging import Logger

def profile_cuda_tensors(log: Logger) -> None:
    """Print all GPU tensors to log."""
    n_total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if getattr(obj, "is_cuda", None):
                    log.info(type(obj), obj.size())
                    n_total += 1
        except:
            pass
    log.info(f"{n_total} total tensors found on GPU.")

def profile_memory(log: Logger, devices: List[int] = [0]) -> None:
    for dev in devices:
        mem = torch.cuda.memory_allocated(dev)
        log.info(f"{dev} mem: {mem // 1024**3}G")
