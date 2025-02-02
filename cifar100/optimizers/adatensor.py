import torch
from torch.optim.optimizer import Optimizer

class AdaRobustCurv(Optimizer):
    r"""Implements the AdaRobustCurv optimizer.

    This optimizer is a novel elementwise method that uses a robust curvature
    estimate to scale the effective learning rate.

    For each parameter tensor \(p\), we maintain:
      - exp_avg: the EMA of the gradient (as in AdamW).
      - prev_grad: the previous gradient (to compute differences).
      - exp_avg_abs_diff: EMA of the absolute difference |g_t - g_{t-1}|.
      - exp_avg_sq_diff: EMA of the squared difference (g_t - g_{t-1})².

    At each step, if a previous gradient is available, we define the per-element
    curvature estimate as

         c_t = (|d_t| + α * sqrt(exp_avg_sq_diff_corrected)) / (1 + α),

    where d_t = g_t - prev_grad and the squared difference accumulator is bias–corrected.
    (You could also bias–correct the absolute accumulator; here we assume the sqrt of
    the squared EMA is our robust RMS measure.) Then the parameter update is

         p ← p − lr * ( m̂ / (c_t + eps) ),

    where m̂ is the bias–corrected first moment (EMA of the gradient). Decoupled weight
    decay is applied as in AdamW.

    Hyperparameters:
      - lr: learning rate.
      - beta1: decay rate for the first moment (gradient EMA).
      - beta2: decay rate for the difference accumulators.
      - alpha: blending parameter for curvature estimation (nonnegative).
      - eps: term added to the denominator for numerical stability.
      - weight_decay: decoupled weight decay.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, alpha=0.5, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(AdaRobustCurv, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr     = group['lr']
            beta1  = group['beta1']
            beta2  = group['beta2']
            alpha  = group['alpha']
            eps    = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Apply decoupled weight decay (like AdamW)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # EMA of gradient
                    state['exp_avg_abs_diff'] = torch.zeros_like(p.data)  # EMA of |g_t - g_{t-1}|
                    state['exp_avg_sq_diff'] = torch.zeros_like(p.data)   # EMA of (g_t - g_{t-1})^2
                    state['prev_grad'] = None

                state['step'] += 1
                t = state['step']

                # Update first moment (as in AdamW)
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Bias-corrected first moment
                bias_correction1 = 1 - beta1 ** t
                m_hat = exp_avg / bias_correction1

                # If previous gradient is available, compute differences; otherwise, initialize.
                if state['prev_grad'] is None:
                    state['prev_grad'] = grad.clone()
                    # For the first step, set the differences to a small default value.
                    diff = torch.zeros_like(grad)
                else:
                    diff = grad - state['prev_grad']
                    state['prev_grad'] = grad.clone()

                # Update EMA of absolute difference and squared difference.
                exp_avg_abs_diff = state['exp_avg_abs_diff']
                exp_avg_sq_diff  = state['exp_avg_sq_diff']
                exp_avg_abs_diff.mul_(beta2).add_(diff.abs(), alpha=1 - beta2)
                exp_avg_sq_diff.mul_(beta2).addcmul_(diff, diff, value=1 - beta2)

                # Bias correction for the squared difference accumulator.
                bias_correction2 = 1 - beta2 ** t
                corrected_sq = exp_avg_sq_diff / bias_correction2
                # Robust RMS measure of differences.
                rms_diff = corrected_sq.sqrt()

                # Form the curvature estimate by blending the absolute difference and its RMS.
                # (One could also bias-correct the absolute accumulator, but here we keep it simple.)
                curvature = (exp_avg_abs_diff + alpha * rms_diff) / (1 + alpha)

                # Compute the update.
                update = m_hat / (curvature + eps)

                # Update parameter.
                p.data.add_(update, alpha=-lr)

        return loss

# Example usage:
if __name__ == "__main__":
    import torch.nn as nn
    # Define a simple model.
    model = nn.Linear(10, 1)
    optimizer = AdaRobustCurv(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999, alpha=0.5, eps=1e-8, weight_decay=1e-4)

    for step in range(100):
        optimizer.zero_grad()
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item()}")
