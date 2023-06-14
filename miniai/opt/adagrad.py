import torch
from torch.optim.optimizer import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.1, delta=1e-7):
        super(AdaGrad, self).__init__(params, {'lr': lr, 'delta': delta})
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['r'] = torch.zeros_like(p)

    # https://github.com/pytorch/pytorch/pull/17679#issuecomment-487234302
    def share_memory():
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['r'].share_memory_()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    # r = r + g (element-wise-mul) g
                    state['r'].addcmul_(g, g, value=1.0)
                    # divisor = sqrt(r) + delta
                    divisor = state['r'].sqrt().add_(group['delta'])
                    # p = p - lr * g (element-wise-div) divisor
                    p.addcdiv_(g, divisor, value=-group['lr'])
