import torch
from torch.optim.optimizer import Optimizer


class AdaTensor(Optimizer):
    r"""Implements the NAGCurv optimizer.

    NAGCurv uses a Nesterov look‐ahead step to obtain a finite-difference estimate
    of the local (elementwise) curvature, which is then used to scale the update.

    For each parameter \(p\), let:
      - \(g_t\) be the gradient at the current parameters.
      - \(v\) be the momentum buffer.
      - We form a lookahead parameter:
            \(p^{\text{look}} = p + \mu\,v\),
        and compute its gradient \(g_t^{\mathrm{NAG}}\) via a closure.
      - The curvature is estimated as:
            \(c_t = |\,g_t^{\mathrm{NAG}} - g_t|\).
      - Then, we update the momentum buffer:
            \(v \leftarrow \mu\, v + \eta\, \frac{g_t}{c_t + \epsilon}\),
        and update the parameter via a Nesterov step:
            \(p \leftarrow p - \Bigl(\mu\, v + \eta\, \frac{g_t}{c_t+\epsilon}\Bigr)\).

    Note:
      This optimizer requires a closure that recomputes the loss and its gradients.
      Because it performs two forward/backward passes (one at the current parameters and
      one at the lookahead parameters), its per-step cost is roughly 2× that of AdamW.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        momentum (float): momentum coefficient (\(\mu\)).
        eps (float): small term added to denominator for numerical stability.
        weight_decay (float): decoupled weight decay coefficient.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super(AdaTensor, self).__init__(params, defaults)

    def step(self, closure):
        """Performs a single optimization step.

        The closure should clear gradients, compute loss, and perform backpropagation.
        It will be called twice: first to compute the current gradients, and then to
        compute the lookahead gradients.
        """
        if closure is None:
            raise RuntimeError("NAGCurv requires a closure that reevaluates the model and returns the loss")

        # ----- 1. Compute current gradients at the current parameters -----
        loss = closure()

        # Save current gradients and a copy of current parameters.
        current_grads = {}
        original_params = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                current_grads[p] = p.grad.data.clone()
                original_params[p] = p.data.clone()

        # ----- 2. Apply decoupled weight decay to current parameters -----
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            if weight_decay != 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data.mul_(1 - lr * weight_decay)

        # ----- 3. Compute lookahead parameters: p_look = p + momentum * v -----
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'v' not in state:
                    state['v'] = torch.zeros_like(p.data)
                # p.data = original + momentum * v
                p.data.add_(state['v'], alpha=momentum)

        # ----- 4. Compute gradients at lookahead parameters -----
        loss_look = closure()
        lookahead_grads = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                lookahead_grads[p] = p.grad.data.clone()

        # ----- 5. Revert parameters to original values -----
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.copy_(original_params[p])

        # ----- 6. For each parameter, compute curvature and update momentum and parameters -----
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = current_grads[p]  # current gradient
                g_look = lookahead_grads[p]  # lookahead gradient
                # Elementwise curvature estimate: finite difference between lookahead and current gradients.
                c = (g_look - g).abs()

                state = self.state[p]
                v = state['v']
                # Update momentum buffer:
                #   v <- momentum * v + lr * (g / (c + eps))
                v.mul_(momentum).add_(g / (c + eps), alpha=lr)
                # Nesterov update: use the lookahead idea
                update = momentum * v + lr * (g / (c + eps))
                p.data.add_(-update)

        return loss


# Example usage:
if __name__ == "__main__":
    import torch.nn as nn

    # A simple model for demonstration.
    model = nn.Linear(10, 1)
    optimizer = NAGCurv(model.parameters(), lr=1e-3, momentum=0.9, eps=1e-8, weight_decay=1e-4)


    # For NAGCurv, the optimizer.step() requires a closure.
    def closure():
        optimizer.zero_grad()
        # A dummy forward pass.
        x = torch.randn(16, 10)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        return loss


    for i in range(50):
        loss = optimizer.step(closure)
        print(f"Step {i}, Loss: {loss.item()}")
