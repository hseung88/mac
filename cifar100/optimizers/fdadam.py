import torch
from torch.optim.optimizer import Optimizer
import math


class PerModeAdam(Optimizer):
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

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2: {}".format(beta2))
        if eps < 0.0:
            raise ValueError("Invalid eps: {}".format(eps))
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(PerModeAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PerModeAdam, self).__setstate__(state)

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
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("PerModeAdam does not support sparse gradients")

                # Optionally apply decoupled weight decay.
                if weight_decay != 0:
                    p.data.mul_(1-step_size*weight_decay)

                state = self.state[p]
                # State initialization.
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # For momentum (first moment)
                    # For each mode, maintain an EMA of squared gradients.
                    state['v'] = []
                    for dim, size in enumerate(p.data.shape):
                        state['v'].append(torch.zeros(size, device=p.device, dtype=p.dtype))
                state['step'] += 1
                t = state['step']

                exp_avg = state['exp_avg']
                v_list = state['v']

                # Update EMA of the raw gradient.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Bias-corrected first moment.
                bias_correction1 = 1 - beta1 ** t
                exp_avg_corr = exp_avg / bias_correction1

                # Compute elementwise squared gradient.
                grad_sq = grad.pow(2)
                dims = grad.dim()
                # Update per-mode EMA for squared gradients.
                for i in range(dims):
                    reduce_dims = tuple(j for j in range(dims) if j != i)
                    # When grad is 1D, reduce_dims might be empty; in that case, use grad_sq directly.
                    if len(reduce_dims) > 0:
                        mode_mean = grad_sq.mean(dim=reduce_dims)
                    else:
                        mode_mean = grad_sq
                    v_list[i].mul_(beta2).add_(mode_mean, alpha=1 - beta2)

                # Compute bias correction for per-mode EMA; here we use the same t for all modes.
                bias_correction2 = 1 - beta2 ** t

                # Combine the corrected per-mode EMAs into an effective variance.
                effective_var = None
                for i in range(dims):
                    # Corrected per-mode variance.
                    v_i_corr = v_list[i] / bias_correction2
                    # Reshape to broadcast: shape = [1, ..., d_i, ..., 1]
                    shape = [1] * dims
                    shape[i] = v_i_corr.shape[0]
                    v_i_expanded = v_i_corr.view(*shape)
                    # Geometric mean: multiply the per-mode factors raised to (1/dims).
                    if effective_var is None:
                        effective_var = v_i_expanded.pow(1.0 / dims)
                    else:
                        effective_var = effective_var * v_i_expanded.pow(1.0 / dims)

                # Compute the denominator.
                denom = effective_var.sqrt().add_(eps)

                # Update parameters.
                p.data.addcdiv_(exp_avg_corr, denom, value=-lr)

        return loss


# Example usage:
if __name__ == "__main__":
    # A simple test model.
    model = torch.nn.Linear(10, 1)
    optimizer = PerModeAdam(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=1e-2)
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
