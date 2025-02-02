import torch
from torch.optim.optimizer import Optimizer


class AdaTensor(Optimizer):
    r"""
    Implements an adaptive tensor-factorized preconditioning optimizer with bias correction,
    iterative bias-correction updates, and precomputation of reduction dimensions.

    For any parameter tensor of shape (d1, d2, ..., dk) with k >= 2, this optimizer maintains:
      - A first moment estimate (EMA) of the gradient.
      - One accumulator per mode for the squared gradient, where for mode i the accumulator
        is an EMA of the squared gradients averaged over all dimensions except the i-th.

    The per-element preconditioning factor is computed as:

         P = (v^(1) * v^(2) * ... * v^(k))^(1/(2*k))

    and the update is given by:

         update = m_hat / (P + eps)

    where m_hat is the bias-corrected first moment.

    For parameters with fewer than 2 dimensions (0D or 1D), a standard AdamW–like update is used.

    Hyperparameters:
      - lr: learning rate.
      - beta1: decay rate for the EMA on the gradient (first moment).
      - beta2: decay rate for the EMA on the squared gradient (second moment).
      - eps: term added to the denominator for numerical stability.
      - weight_decay: L2 penalty.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.9, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(AdaTensor, self).__init__(params, defaults)

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
                grad = p.grad.data

                # Apply decoupled weight decay (in-place update).
                p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]

                # Initialize step and bias correction terms if needed.
                if 'step' not in state:
                    state['step'] = 0
                    state['bias_correction1'] = 1.0
                    state['bias_correction2'] = 1.0
                state['step'] += 1
                # Update the bias correction factors iteratively.
                state['bias_correction1'] *= beta1
                state['bias_correction2'] *= beta2

                # Get current bias corrections.
                bc1 = state['bias_correction1']
                bc2 = state['bias_correction2']

                dim = grad.dim()
                if dim >= 2:
                    # ---------- First Moment: EMA on the Gradient ----------
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad)
                    m = state['exp_avg']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    m_hat = m / (1 - bc1)

                    # ---------- Second Moment: Factorized EMA on Squared Gradients ----------
                    if 'accumulators' not in state:
                        # Create one accumulator per mode.
                        state['accumulators'] = [
                            torch.zeros(p.shape[i], device=p.device, dtype=p.dtype)
                            for i in range(dim)
                        ]
                    accs = state['accumulators']

                    # Precompute reduction dimensions if not already done.
                    if 'dims_to_reduce' not in state:
                        state['dims_to_reduce'] = [
                            tuple(j for j in range(dim) if j != i) for i in range(dim)
                        ]
                    dims_to_reduce = state['dims_to_reduce']

                    grad_sq = grad.pow(2)
                    # Loop over each mode (dimension).
                    for i in range(dim):
                        agg = grad_sq.mean(dim=dims_to_reduce[i])
                        #agg = grad.mean(dim=dims_to_reduce[i])
                        accs[i].mul_(beta2).add_(agg, alpha=1 - beta2)

                    # Bias-correct the accumulators.
                    corrected_accs = [accs[i] / (1 - bc2) for i in range(dim)]

                    # ---------- Combine Accumulators into a Per-element Preconditioning Factor ----------
                    precond = None
                    for i in range(dim):
                        # Reshape so that each accumulator broadcasts to the parameter shape.
                        shape = [1] * dim
                        shape[i] = corrected_accs[i].shape[0]
                        acc_i = corrected_accs[i].view(*shape)
                        precond = acc_i if precond is None else precond * acc_i
                    precond = precond.pow(1.0 / (2 * dim))

                    # ---------- Compute the Update ----------
                    update = m_hat / (precond + eps)
                else:
                    # For 0D or 1D parameters, use standard AdamW-like updates.
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad)
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    m = state['exp_avg']
                    v = state['exp_avg_sq']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                    m_hat = m / (1 - bc1)
                    v_hat = v / (1 - bc2)
                    update = m_hat / (v_hat.sqrt() + eps)

                p.data.add_(update, alpha=-lr)

        return loss


# Example usage:
if __name__ == "__main__":
    import torch.nn as nn

    # Create a dummy parameter with an arbitrary shape (e.g., a 4D tensor).
    param = nn.Parameter(torch.randn(2, 3, 4, 5, requires_grad=True))
    optimizer = AdaTensor([param], lr=1e-3, beta1=0.9, beta2=0.9, eps=1e-8, weight_decay=0)

    for step in range(10):
        optimizer.zero_grad()
        # Example loss: sum of squares of the parameter.
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item()}")
