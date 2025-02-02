import math
import torch
from torch.optim.optimizer import Optimizer


class AdaTensor(Optimizer):
    r"""Implements the AdaCurv optimizer.

    AdaCurv is a novel elementwise optimizer that incorporates an estimate of
    local curvature by tracking the change in gradients over time.

    For each parameter, it maintains:
      - m_t: Exponential moving average (EMA) of the gradient.
      - c_t: EMA of the absolute difference between consecutive gradients,
             used as a proxy for curvature.
      - prev_grad: The gradient from the previous step.

    The update rule is:
        m_t = β₁ * m₍ₜ₋₁₎ + (1-β₁) * g_t,
        δ_t = g_t - g₍ₜ₋₁₎   (if available),
        c_t = β₂ * c₍ₜ₋₁₎ + (1-β₂) * |δ_t|,
        with bias corrections:
            m̂_t = m_t / (1 - β₁^t),
            ĉ_t = c_t / (1 - β₂^t),
        and then the parameter update is:
            x₍ₜ₊₁₎ = xₜ - lr * (m̂_t / (ĉ_t + ε)).

    For the very first step (when no previous gradient is available), c_t is
    initialized to a small value (or to ε) so that the update falls back to something
    similar to AdamW.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        beta1 (float): coefficient for the EMA of the gradient.
        beta2 (float): coefficient for the EMA of the gradient differences.
        eps (float): term added to the denominator for numerical stability.
        weight_decay (float): weight decay (L2 penalty).
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if g.is_sparse:
                    raise RuntimeError('AdaCurv does not support sparse gradients')

                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['c'] = torch.zeros_like(p.data)
                    state['prev_grad'] = None  # No previous gradient for the first step.

                state['step'] += 1
                t = state['step']

                # Decoupled weight decay (like in AdamW)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                m = state['m']
                # Update first moment: EMA of gradient.
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                m_hat = m / (1 - beta1 ** t)

                # Update curvature estimate.
                if state['prev_grad'] is None:
                    # On the first step, there is no previous gradient.
                    state['prev_grad'] = g.clone()
                    # Initialize curvature to a small value.
                    c = state['c']
                    c.fill_(eps)
                else:
                    delta = g - state['prev_grad']
                    c = state['c']
                    c.mul_(beta2).add_(delta.abs(), alpha=1 - beta2)
                    state['prev_grad'] = g.clone()
                c_hat = c / (1 - beta2 ** t)

                # Compute the update.
                update = m_hat / (c_hat + eps)

                # Update the parameter.
                p.data.add_(update, alpha=-lr)

        return loss


# Example usage:
if __name__ == "__main__":
    import torch.nn as nn

    # A simple model for demonstration.
    model = nn.Linear(10, 1)
    optimizer = AdaTensor(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=1e-4)

    for i in range(100):
        optimizer.zero_grad()
        x = torch.randn(16, 10)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        optimizer.step()
        print(f"Step {i}, Loss: {loss.item()}")
