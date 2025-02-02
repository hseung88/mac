import torch
from torch.optim.optimizer import Optimizer


class AdaTensor(Optimizer):
    r"""
    Implements an adaptive tensor-factorized preconditioning optimizer.

    For any parameter tensor of shape (d1, d2, ..., dk) with k >= 1, this
    optimizer maintains:
      - A first moment estimate (EMA) of the gradient.
      - One accumulator per mode for the squared gradient. For mode i, the
        accumulator is updated as an EMA of the squared gradients aggregated over
        all dimensions except the i-th.

    The per-element preconditioning factor is computed as:

         P = (v^(1) * v^(2) * ... * v^(k))^(1/(2*k))

    and the update is given by:

         update = m / (P + eps)

    For 0-dimensional (scalar) parameters, a standard AdamW–like update is used.

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

                # Apply weight decay if specified.
                p.data.mul_(1-lr*weight_decay)

                state = self.state[p]

                # Update step count.
                if 'step' not in state:
                    state['step'] = 0
                state['step'] += 1
                t = state['step']

                dim = grad.dim()

                if dim >= 2:
                    # ---------- First Moment: EMA on the Gradient ----------
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad)
                    m = state['exp_avg']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Bias correction for first moment.
                    m_hat = m / (1 - beta1 ** t)

                    # ---------- Second Moment: Factorized EMA on Squared Gradients ----------
                    if 'accumulators' not in state:
                        # Create one accumulator per mode (dimension).
                        state['accumulators'] = [torch.zeros(p.shape[i], device=p.device, dtype=p.dtype)
                                                 for i in range(dim)]
                    accs = state['accumulators']
                    grad_sq = grad.pow(2)
                    for i in range(dim):
                        # Aggregate over all dimensions except the i-th.
                        dims_to_reduce = tuple(j for j in range(dim) if j != i)
                        agg = grad_sq.mean(dim=dims_to_reduce)
                        accs[i].mul_(beta2).add_(agg, alpha=1 - beta2)

                    # ---------- Bias Correction for Second Moment ----------
                    corrected_accs = []
                    for i in range(dim):
                        corrected_accs.append(accs[i] / (1 - beta2 ** t))

                    # ---------- Combine Accumulators ----------
                    # We want: P = (acc_corrected_0 * acc_corrected_1 * ... * acc_corrected_{dim-1})^(1/(2*dim))
                    precond = None
                    for i in range(dim):
                        # Reshape each accumulator to broadcast to grad's shape.
                        shape = [1] * dim
                        shape[i] = corrected_accs[i].shape[0]
                        acc_i = corrected_accs[i].view(*shape)
                        precond = acc_i if precond is None else precond * acc_i
                    precond = precond.pow(1.0 / (2 * dim))

                    # ---------- Compute the Update ----------
                    update = m_hat / (precond + eps)

                else:
                    # For scalar (0D) parameters, use a standard AdamW–like update.
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad)
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    m = state['exp_avg']
                    v = state['exp_avg_sq']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    update = m_hat / (v_hat.sqrt() + eps)

                    # ---------- Update Parameter ----------
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