import torch
from torch.optim.optimizer import Optimizer


def mode_product(tensor, matrix, mode):
    """
    Multiply a tensor by a matrix along the specified mode.
    This function permutes the tensor so that the given mode is first,
    reshapes it to a 2D matrix, multiplies, and then reshapes back.
    """
    shape = list(tensor.shape)
    # Permute so that the mode of interest is first.
    perm = [mode] + [i for i in range(len(shape)) if i != mode]
    tensor_perm = tensor.permute(perm)
    d = shape[mode]
    tensor_mat = tensor_perm.reshape(d, -1)
    # Multiply.
    product = matrix @ tensor_mat
    # Reshape back.
    new_shape = [matrix.shape[0]] + shape[:mode] + shape[mode + 1:]
    result = product.reshape(new_shape)
    # Inverse permutation to return to original order.
    inv_perm = [perm.index(i) for i in range(len(shape))]
    result = result.permute(inv_perm)
    return result


class AdaTensor(Optimizer):
    r"""Tensor-level preconditioning optimizer with modewise rank-1 inverse preconditioner,
    EMA on the preconditioners, and heavyball momentum.

    For each parameter tensor \(p \in \mathbb{R}^{d_1\times d_2 \times \cdots \times d_k}\) with \(k\ge2\),
    the optimizer does the following for each mode \(i\):
      - Computes the mean gradient over all dimensions except mode \(i\):
            \(v^{(i)} = \text{mean}(G, \text{over all modes except } i)\).
      - Forms a rank-1 approximation of the second moment:
            \(v^{(i)}(v^{(i)})^T\).
      - With damping, defines
            \(P^{(i)} = v^{(i)}(v^{(i)})^T + \varepsilon I\).
      - Computes the inverse preconditioner along mode \(i\) as
            \(M^{(i)} = \bigl(P^{(i)}\bigr)^{-1/k}\),
        i.e. scaling by \(1/(\|v^{(i)}\|^2+\varepsilon)^{1/k}\) along \(v^{(i)}\)
        and \(1/\varepsilon^{1/k}\) in the orthogonal subspace.
      - Updates an EMA of \(M^{(i)}\) using a coefficient `precond_beta`.
      - Applies the resulting preconditioner along mode \(i\) to the gradient.
    For parameters with fewer than 2 dimensions, it falls back to a diagonal update (AdamW–like).

    Finally, heavyball momentum is applied:
      \[
      v \leftarrow \mu\, v + (\text{preconditioned gradient}),
      \]
      \[
      p \leftarrow p - \eta\, v.
      \]

    Hyperparameters:
      - lr: learning rate.
      - eps: damping constant.
      - weight_decay: decoupled weight decay.
      - momentum: heavyball momentum coefficient.
      - precond_beta: EMA coefficient for preconditioners.
    """

    def __init__(self, params, lr=1e-3, eps=1e-8, weight_decay=0, beta1=0.9, beta2=0.999):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(AdaTensor, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']

            for p in group['params']:
                if p.grad is None:
                    continue
                G = p.grad.data
                state = self.state[p]

                # Apply decoupled weight decay.
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Initialize heavyball momentum buffer.
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                k = G.dim()  # number of modes (works even for 1D, where k == 1)
                # Initialize EMA of preconditioners for each mode if not present.
                if 'EMA_M' not in state:
                    state['EMA_M'] = [None] * k  # One for each mode.

                # For each mode, compute the rank-1 inverse preconditioner and update its EMA.
                for i in range(k):
                    # For a 1D tensor, dims will be empty; handle that separately.
                    dims = [j for j in range(k) if j != i]
                    if dims:
                        v = G.mean(dim=dims)
                    else:
                        v = G.clone()  # For 1D, use the entire vector.

                    norm_v2 = v.pow(2).sum()
                    # Scaling in the orthogonal subspace.
                    inv_scale_orth = 1.0
                    # Scaling in the direction of v.
                    inv_scale_v = (norm_v2 + eps) ** (-1.0 / k)
                    diff_scale = inv_scale_orth - inv_scale_v
                    norm_v2_safe = norm_v2 if norm_v2 > 1e-4 else 1e-4
                    d = p.shape[i]
                    I = torch.eye(d, device=p.device, dtype=p.dtype)
                    outer = (v.unsqueeze(1) @ v.unsqueeze(0)) / norm_v2_safe
                    M_i = inv_scale_orth * I - diff_scale * outer

                    # Update EMA of preconditioner for mode i.
                    if state['EMA_M'][i] is None:
                        state['EMA_M'][i] = M_i.clone()
                    else:
                        state['EMA_M'][i].mul_(beta2).add_(M_i, alpha=1 - beta2)
                    EMA_M_i = state['EMA_M'][i]
                    # Precondition gradient along mode i.
                    G = mode_product(G, EMA_M_i, i)

                # Now G is preconditioned along all modes.
                state['momentum_buffer'].mul_(beta1).add_(G)
                p.data.add_(state['momentum_buffer'], alpha=-lr)

            return loss

# Example usage:
if __name__ == "__main__":
    import torch.nn as nn
    # Define a parameter tensor; this example uses a 4D tensor.
    param = nn.Parameter(torch.randn(4, 3, 3, 3, requires_grad=True))
    # Also try a 1D parameter.
    param1d = nn.Parameter(torch.randn(10, requires_grad=True))

    optimizer = AdaTensor([param, param1d],
                                                lr=0.1, eps=1e-8,
                                                weight_decay=0.0001,
                                                beta1=0.9,
                                                beta2=0.95)

    for step in range(10):
        optimizer.zero_grad()
        # Dummy loss: sum of squares of both parameters.
        loss = (param ** 2).sum() + (param1d ** 2).sum()
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item()}")