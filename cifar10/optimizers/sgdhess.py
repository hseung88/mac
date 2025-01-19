import torch
from torch.optim import Optimizer


class SGDHess(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        clip=False,
        **kwargs
    ):
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.clip = clip
        self.iteration = -1

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGDHess, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('nesterov', False)
            for p in group['params']:
                state = self.state[p]
                state['displacement'] = torch.zeros_like(p)
                state['max_grad'] = torch.zeros(1, device=p.device)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.iteration += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            vector = []
            grads = []
            param = []

            for p in group['params']:
                if p.grad is None:
                    continue
                vector.append(self.state[p]['displacement'])
                grads.append(p.grad)
                param.append(p)

            hvp = torch.autograd.grad(outputs=grads, inputs=param, grad_outputs=vector)
            with torch.no_grad():
                i = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    displacement, max_grad = state['displacement'], state['max_grad']

                    with torch.no_grad():
                        d_p = p.grad
                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)
                        if momentum != 0:
                            if 'momentum_buffer' not in state:
                                buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = state['momentum_buffer']
                                buf.add_(hvp[i]).add_(
                                    displacement,
                                    alpha=weight_decay
                                ).mul_(momentum).add_(d_p, alpha=1 - dampening)

                                if self.clip:
                                    torch.nn.utils.clip_grad_norm_(buf, max_grad)
                                    max_grad.copy_(torch.maximum((1-dampening)/(1-momentum)*torch.norm(d_p),
                                                                 max_grad))
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf
                        displacement.copy_(d_p).mul_(-group['lr'])
                        p.add_(displacement)
                    i += 1

        return loss
