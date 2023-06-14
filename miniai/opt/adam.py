import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, decays=(0.9, 0.999), delta=1e-8):
        super(Adam, self).__init__(
            params, {'lr': lr, 'decays': decays, 'delta': delta})
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['t'] = 0
                state['v'] = torch.zeros_like(p)
                state['r'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    state['v'].mul_(group['decays'][0]).add_(
                        g, alpha=1-group['decays'][0])
                    state['r'].mul_(group['decays'][1]).addcmul_(
                        g, g, value=1 - group['decays'][1])
                    state['t'] += 1
                    bias_corrected_v = state['v'].div(
                        1 - group['decays'][0] ** state['t'])
                    bias_corrected_r = state['r'].div(
                        1 - group['decays'][1] ** state['t'])
                    divisor = bias_corrected_r.sqrt().add_(group['delta'])
                    p.addcdiv_(bias_corrected_v, divisor, value=-group['lr'])
