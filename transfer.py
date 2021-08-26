import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def transfer_loader(model, weights_dir, zero_head=True):
    weights = torch.load(weights_dir)
    with torch.no_grad():
        if zero_head: # initialize head to zero
            nn.init.zeros_(weights["head.weight"])
            nn.init.zeros_(weights["head.bias"])

    model.load_state_dict(weights)

    return model

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))



