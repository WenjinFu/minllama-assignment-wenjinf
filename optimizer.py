



from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                
                #todo
                
 

                # State should be stored in this dictionary
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                # Decay the first and second moment running average coefficient
                state['step'] += 1
                # Update exponential moving averages
                state['exp_avg'] = state['exp_avg'] * beta1 + (1 - beta1) * grad 
                state['exp_avg_sq'] = state['exp_avg_sq'] * beta2 + (1 - beta2) * grad * grad

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']
                # corrected_exp_avg = state['exp_avg']  / bias_correction1
                # print("corrected_exp_avg", corrected_exp_avg)

                # corrected_exp_avg_sq = state['exp_avg_sq']  / bias_correction2
                # print("corrected_exp_avg_sq", corrected_exp_avg_sq)

                # Update parameterss

                # Compute denominator
                alpha = group['lr'] * math.sqrt((bias_correction2)) / (bias_correction1)
                denom = (state['exp_avg_sq'] .sqrt() + group['eps'])


                
                p.data = p.data - alpha * ( state['exp_avg'] / denom)

                # Add weight decay after the main gradient-based updates.
                if group['weight_decay'] > 0:
                    p.data = p.data + (-group['lr']) * p.data * weight_decay
                # Please note that the learning rate should be incorporated into this update.


        return loss