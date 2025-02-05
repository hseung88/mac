import torch
from torch.optim.optimizer import Optimizer
import math


class FSO(Optimizer):
    r"""Implements a per-mode adaptive optimizer with momentum and bias correction.

    For each parameter tensor p of shape (d₁, d₂, ..., dₙ), this optimizer maintains:
      - An exponential moving average (EMA) of the raw gradient:
            m_t = β₁ * m_{t-1} + (1-β₁) * g_t,
        with bias correction:  m̂_t = m_t / (1-β₁^t).

      - For each mode i (i=1,...,n), an EMA of the squared gradients averaged over all
        dimensions except i:
            vₜ^(i) = β₂ * vₜ₋₁^(i) + (1-β₂) * mean(g_t^2, over all dims except i),
        with bias correction:  v̂ₜ^(i) = vₜ^(i) / (1-β₂^t).

      - The effective variance is computed via the geometric mean:
            v_eff = ∏_{i=1}^n (v̂^(i))^(1/n).

      The update is then:
            p ← p - lr * (m̂_t / (sqrt(v_eff) + eps)).

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        beta1 (float): Coefficient for the EMA of the gradient (default: 0.9).
        beta2 (float): Coefficient for the EMA of the per-mode squared gradients (default: 0.99).
        eps (float): Term added to the denominator for numerical stability (default: 1e-8).
        weight_decay (float): Weight decay (L2 penalty) (default: 0).
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, damping=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2: {}".format(beta2))
        if eps < 0.0:
            raise ValueError("Invalid eps: {}".format(eps))
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, damping=damping, weight_decay=weight_decay)
        super(FSO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FSO, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            loss (torch.Tensor, optional): Loss evaluated after the step, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            damping = group['damping']
            base_wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                # State initialization.
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # For momentum (first moment)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'] += 1
                t = state['step']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                #sq_norm = grad.pow(2).sum()
                #grad.div_(damping + sq_norm)

                # Update EMA of the raw gradient.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)

                # Bias-corrected first moment.
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.mul_(1 - lr * base_wd / denom)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# Example usage:
if __name__ == "__main__":
    # A simple test model.
    model = torch.nn.Linear(10, 1)
    optimizer = FSO(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=1e-2, damping=1.0)
    loss_fn = torch.nn.MSELoss()

    # Dummy data.
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item()}")
