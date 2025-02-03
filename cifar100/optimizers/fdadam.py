import torch
from torch.optim.optimizer import Optimizer
import math


class PerModeAdam(Optimizer):
    r"""Implements a per-mode adaptive optimizer.

    For each parameter tensor \(p\) of shape \((d_1,d_2,\dots,d_N)\), we maintain
    a separate EMA of the squared gradient along each mode:

    For each mode \(i\) (0-indexed),

      v^{(i)}_t = β * v^{(i)}_{t-1} + (1-β) * mean(g², over all dims except i)

    Then, the effective variance for each coordinate is computed as the geometric mean:

      v_eff(i_1, ..., i_N) = ∏_{i=1}^N [v^{(i)}[i_i]]^(1/N)

    The parameter update is then given by:

      p ← p - lr * g / (sqrt(v_eff) + eps)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        beta (float): coefficient for computing running averages (default: 0.99).
        eps (float): term added to the denominator for numerical stability (default: 1e-8).
        weight_decay (float): weight decay (L2 penalty) (default: 0).
    """

    def __init__(self, params, lr=1e-3, beta=0.99, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
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
            loss (torch.Tensor, optional): loss evaluated after the step, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("PerModeAdam does not support sparse gradients")

                # Optionally apply decoupled weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                # Initialize state for this parameter.
                if not state:
                    state['step'] = 0
                    # For each mode, store a 1D tensor of EMA of squared gradients.
                    # For a tensor of shape (d1, d2, ..., dN), we have N EMA vectors.
                    state['v'] = []
                    for dim, size in enumerate(p.data.shape):
                        # Initialize with zeros for each mode.
                        state['v'].append(torch.zeros(size, device=p.device, dtype=p.dtype))
                state['step'] += 1

                # Compute elementwise squared gradient.
                grad_sq = grad.pow(2)

                # Update per-mode accumulators.
                # For each mode i, compute the mean over all dimensions except i.
                v_list = state['v']
                dims = grad.dim()
                for i in range(dims):
                    # Reduce over all dimensions except i.
                    reduce_dims = tuple(j for j in range(dims) if j != i)
                    # If reduce_dims is empty (e.g. a 1D tensor), then grad_sq.mean() returns a scalar.
                    # In that case, we want to use the gradient squared itself.
                    if len(reduce_dims) > 0:
                        mode_mean = grad_sq.mean(dim=reduce_dims)
                    else:
                        mode_mean = grad_sq
                    # Update the EMA: v = β * v + (1-β) * mode_mean
                    v_list[i].mul_(beta).add_(mode_mean, alpha=1 - beta)

                # Now, combine the per-mode accumulators to form an effective variance.
                # We use the geometric mean: effective_var = ∏_{i=1}^N (v_i)^(1/N)
                effective_var = None
                for i in range(dims):
                    # Reshape v_list[i] to be broadcastable to p.data.
                    shape = [1] * dims
                    shape[i] = v_list[i].shape[0]
                    v_i_expanded = v_list[i].view(*shape)
                    if effective_var is None:
                        effective_var = v_i_expanded.pow(1.0 / dims)
                    else:
                        effective_var = effective_var * v_i_expanded.pow(1.0 / dims)

                # Compute the denominator.
                denom = effective_var.sqrt().add_(eps)

                # Update the parameter.
                p.data.addcdiv_(grad, denom, value=-lr)

        return loss


# Example usage:
if __name__ == "__main__":
    # A simple model for testing.
    model = torch.nn.Linear(10, 1)
    optimizer = PerModeAdam(model.parameters(), lr=1e-3, beta=0.99, eps=1e-8, weight_decay=1e-2)
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
