import torch
from torch.optim.optimizer import Optimizer


class GradientDescentWithMomentum(Optimizer):
    def __init__(self, params, lr=0.1, decay=0.9):
        super(GradientDescentWithMomentum, self).__init__(
            params, {'lr': lr, 'decay': decay})
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['v'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    state['v'].mul_(group['decay']).add_(
                        p.grad, alpha=-group['lr'])
                    p.add_(state['v'])
