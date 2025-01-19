import math
from typing import List

import torch
from torch import Tensor
from torch.optim import Optimizer


class NagHess(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        betas=(0.9, 0.92, 0.99),
        weight_decay=0,
        eps=1e-8,
        foreach: bool = True,
        **kwargs
    ):
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, foreach=foreach)

        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']
            bias_correction3 = 1.0 - beta3**group['step']

            vector = []
            grads = []
            params = []
            exp_avgs = []
            exp_avg_sqs = []
            neg_pre_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['neg_pre_grad'] = p.grad.detach().clone().mul_(-1.0)
                    state['exp_avg'] = p.grad.detach().clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)

                vector.append(self.state[p]['exp_avg'])
                grads.append(p.grad)
                params.append(p)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                neg_pre_grads.append(state['neg_pre_grad'])

            hvp = torch.autograd.grad(outputs=grads, inputs=params, grad_outputs=vector)

            kwargs = dict(
                params=params,
                grads=grads,
                hvp=hvp,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                neg_pre_grads=neg_pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps']
            )

            if group['foreach']:
                _multi_tensor_naghess(**kwargs)
            else:
                _single_tensor_naghess(**kwargs)
        return loss


@torch.no_grad()
def _single_tensor_naghess(
    params: List[Tensor],
    grads: List[Tensor],
    hvp: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    neg_pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    for i, p in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        neg_pre_grad = neg_pre_grads[i]

        neg_pre_grad.add_(grad).mul_(beta2).add_(grad)

        exp_avg_sq.mul_(beta3).addcmul_(neg_pre_grad, neg_pre_grad, value=1 - beta3)
        denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
        exp_avg.add_(hvp[i], alpha=-lr).mul_(beta1).addcdiv_(neg_pre_grad, denom, value=1-beta1)

        d_p = exp_avg / bias_correction1

        if weight_decay != 0:
            d_p = d_p.add(p, alpha=weight_decay)

        p.add_(d_p, alpha=-lr)
        neg_pre_grad.copy_(grad).mul_(-1.0)


@torch.no_grad()
def _multi_tensor_naghess(
    params: List[Tensor],
    grads: List[Tensor],
    hvp: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    neg_pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs,
                            neg_pre_grads,
                            neg_pre_grads,
                            value=1 - beta3)
    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    torch._foreach_add_(exp_avgs, hvp, alpha=-lr)
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_addcdiv_(exp_avgs, neg_pre_grads, denom, value=1 - beta1)

    d_p = torch._foreach_div(exp_avgs, bias_correction1)
    if weight_decay != 0:
        torch._foreach_add_(d_p, params, alpha=weight_decay)

    torch._foreach_add_(params, d_p, alpha=-lr)
    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)
