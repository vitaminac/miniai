import torch
from torch.optim.optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.001, decay=0.9, delta=1e-6):
        super(RMSProp, self).__init__(
            params, {'lr': lr, 'decay': decay, 'delta': delta})
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['r'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    # r = decay * r + (1 - decay) * g (element-wise-mul) g
                    state['r'].mul_(group['decay']).addcmul_(
                        g, g, value=1 - group['decay'])
                    # divisor = sqrt(r + delta)
                    divisor = state['r'].add(group['delta']).sqrt_()
                    # p = p - lr * g (element-wise-div) divisor
                    p.addcdiv_(g, divisor, value=-group['lr'])
