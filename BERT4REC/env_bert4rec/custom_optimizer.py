import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class AdamW(torch.optim.Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # if grad.is_sparse:
                #     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # next_m = beta1 * m + (1.0 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha = 1.0 - beta1)
                # next_v = beta2 * v + (1.0 - beta2) * grad * grad
                v.mul_(beta2).addcmul_(grad, grad, value = 1.0 - beta2)
                update = m / (v.sqrt().add_(group["eps"]))

                # Add weight decay
                if group["weight_decay"] > 0.0:
                    old_p = p.data.clone()
                    p.data.mul_(group['weight_decay']).add_(update).mul_(-group['lr']).add_(old_p)
                else:
                    p.data.add_(update, alpha = -group['lr'])

        return loss
