import torch
from torch.optim.optimizer import Optimizer
import math


class ROTA(Optimizer):
    r"""Implements an optimizer that uses EMA on both the gradient and a rank-1
    curvature approximation computed from mode-wise means of the gradient.

    For each parameter tensor T of shape (d₁, d₂, ..., dₙ):
      1. Compute the per-mode means:
             μ^(i) = mean(grad, over all dims except i)
         (if there is no dimension to average over, use the tensor directly)
      2. Form a rank-1 tensor:
             R = μ^(1) ⊗ μ^(2) ⊗ ... ⊗ μ^(n)
      3. Maintain an EMA of the gradient (exp_avg) and an EMA of R (exp_avg_rank1)
      4. Update using:
             p ← p - lr * (exp_avg / (sqrt(exp_avg_rank1) + eps))

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): learning rate.
        beta1 (float): coefficient used for computing running average of gradients (default: 0.9)
        beta2 (float): coefficient used for computing running average of rank-1 tensor (default: 0.99)
        eps (float): term added to the denominator for numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 value: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 value: {}".format(beta2))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(ROTA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ROTA, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

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
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Rank1EMAOptimizer does not support sparse gradients")

                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state['step'] = 0
                    # EMA of gradient.
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # EMA of rank-1 curvature tensor.
                    state['exp_avg_rank1'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                exp_avg_rank1 = state['exp_avg_rank1']

                state['step'] += 1

                # Optionally apply decoupled weight decay.
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update the EMA of the gradient.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the current rank-1 tensor approximation from mode-wise means.
                # For a tensor grad of shape (d1, d2, ..., dN):
                #   For each mode i, compute the mean over all dimensions except i.
                if grad.dim() == 0:
                    cur_rank1 = grad.abs()
                else:
                    dims = grad.dim()
                    cur_rank1 = None
                    for dim in range(dims):
                        # Compute mean over all dims except the current one.
                        dims_to_average = tuple(i for i in range(dims) if i != dim)
                        # If dims_to_average is empty (e.g. for a 1D tensor), use grad directly.
                        if len(dims_to_average) == 0:
                            mode_mean = grad
                        else:
                            mode_mean = grad.mean(dim=dims_to_average, keepdim=False)
                        # Reshape for broadcasting: shape becomes [1,...,d, ...,1]
                        shape_expanded = [1] * dims
                        # Here, mode_mean should be 1D. For safety, ensure it has a dimension.
                        if mode_mean.dim() == 0:
                            mode_mean = mode_mean.unsqueeze(0)
                        else:
                            mode_mean = mode_mean.view(-1)
                        shape_expanded[dim] = mode_mean.shape[0]
                        mode_mean = mode_mean.view(*shape_expanded)
                        if cur_rank1 is None:
                            cur_rank1 = mode_mean
                        else:
                            cur_rank1 = cur_rank1 * mode_mean  # elementwise product via broadcasting.
                    cur_rank1 = cur_rank1.abs()

                # Update the EMA of the rank-1 tensor.
                exp_avg_rank1.mul_(beta2).add_(cur_rank1, alpha=1 - beta2)

                # Compute the effective scaling denominator.
                denom = exp_avg_rank1.sqrt().add_(eps)

                # Parameter update: use the EMA of the gradient.
                p.data.addcdiv_(exp_avg, denom, value=-lr)

        return loss


# Example usage:
if __name__ == "__main__":
    # Simple test model.
    model = torch.nn.Linear(10, 1)
    optimizer = ROTA(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.99, weight_decay=1e-2)
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
