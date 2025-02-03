import torch
from torch.optim.optimizer import Optimizer, required
import math


class FDAdam(Optimizer):
    r"""Implements Finite-Difference Curvature-Enhanced Adam (FD-Adam).

    This optimizer is a variant of AdamW that enriches the elementwise second-moment
    estimate with curvature information approximated via finite differences between
    successive gradients and parameter updates.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        curvature_beta (float, optional): coefficient for exponential moving average
            of the finite-difference curvature (default: 0.9)
        alpha (float, optional): scaling factor for the curvature term (default: 1e-1)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        eps_fd (float, optional): small term to prevent division by zero in finite-difference curvature (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 curvature_beta=0.9, alpha=1e-1, eps=1e-8, eps_fd=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= curvature_beta < 1.0:
            raise ValueError("Invalid curvature_beta value: {}".format(curvature_beta))

        defaults = dict(lr=lr, betas=betas, curvature_beta=curvature_beta,
                        alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(FDAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FDAdam, self).__setstate__(state)

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
            beta1, beta2 = group['betas']
            curvature_beta = group['curvature_beta']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                # Get current gradient
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FDAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Step count
                    state['step'] = 0
                    # Exponential moving average of gradient (for momentum, if desired)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient (AdamW-style second moment)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of the finite-difference curvature estimate
                    state['exp_avg_curv'] = torch.zeros_like(p.data)
                    # Store previous gradient for finite-difference curvature computation
                    state['prev_grad'] = torch.zeros_like(p.data)
                    # Store previous parameter value
                    state['prev_param'] = p.data.clone()

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_curv = state['exp_avg_curv']
                prev_grad = state['prev_grad']
                prev_param = state['prev_param']

                state['step'] += 1

                # Optionally apply weight decay (AdamW-style decoupled weight decay)
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update exponential moving averages for grad and squared grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute finite-difference curvature estimate elementwise.
                # We compute: curvature_est = |grad - prev_grad| / (|p - prev_param| + eps_fd)
                diff_grad = (grad - prev_grad).abs_()
                diff_param = (p.data - prev_param).abs_()
                curvature_est = diff_grad / (diff_param + eps)

                # Update exponential moving average of curvature
                exp_avg_curv.mul_(curvature_beta).add_(curvature_est, alpha=1 - curvature_beta)

                # Combine squared gradient and curvature estimate into effective second moment
                #effective_second_moment = exp_avg_sq + alpha * exp_avg_curv
                effective_second_moment = exp_avg_curv

                # Compute bias corrections (optional, similar to Adam)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute the adaptive step. Here we use the standard AdamW update rule,
                # replacing exp_avg_sq with effective_second_moment.
                #denom = (effective_second_moment.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                denom = (effective_second_moment / bias_correction2).add_(eps)
                step_size = lr / bias_correction1

                # Parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Save current grad and param for next step's finite difference
                state['prev_grad'].copy_(grad)
                state['prev_param'].copy_(p.data)

        return loss


# Example usage:
if __name__ == "__main__":
    # Simple test model
    model = torch.nn.Linear(10, 1)
    optimizer = FDAdam(model.parameters(), lr=1e-3, alpha=1e-1, weight_decay=1e-2)
    loss_fn = torch.nn.MSELoss()

    # Dummy data
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
