import torch
from torch.optim.optimizer import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, params, lr=0.1):
        super(GradientDescent, self).__init__(params, {'lr': lr})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-group['lr'])
