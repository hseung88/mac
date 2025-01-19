import math
import torch
import torch.optim as optim
from .utils.tensor_utils import reshape_grad
from .utils.kfac_utils import ComputeCovA, ComputeCovG, update_running_stat

class KFAC(optim.Optimizer):
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.03,
                 kl_clip=0.001,
                 weight_decay=1e-5,
                 Tcov=5,
                 Tinv=50,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        super(KFAC, self).__init__(params, defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self._model = None
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.grad_outputs = {}
        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.Tcov = Tcov
        self.Tinv = Tinv

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self._prepare_model()

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.Tcov == 0:
            aa = self.CovAHandler(input[0].detach(), module)
            if self.steps == 0:
                self.m_aa[module] = torch.eye(aa.size(0), device=aa.device)
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.Tcov == 0:
            gg = self.CovGHandler(grad_output[0].detach(), module, self.batch_averaged)
            if self.steps == 0:
                self.m_gg[module] = torch.eye(gg.size(0), device=gg.device)
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _update_inv(self, m):
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.linalg.eigh(self.m_aa[m])
        self.d_g[m], self.Q_g[m] = torch.linalg.eigh(self.m_gg[m])

        self.d_a[m].clamp_min_(eps)
        self.d_g[m].clamp_min_(eps)

    def _get_natural_grad(self, m, p_grad_mat, damping):
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view_as(m.weight.grad)
            v[1] = v[1].view_as(m.bias.grad)
        else:
            v = [v.view_as(m.weight.grad)]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        vg_sum = sum(
            (v[0] * m.weight.grad * lr ** 2).sum().item() +
            (v[1] * m.bias.grad * lr ** 2).sum().item()
            if m.bias is not None else (v[0] * m.weight.grad * lr ** 2).sum().item()
            for m, v in updates.items()
        )
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m, v in updates.items():
            m.weight.grad.copy_(v[0]).mul_(nu)
            if m.bias is not None:
                m.bias.grad.copy_(v[1]).mul_(nu)

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.Tcov:
                    d_p.add_(p.data, alpha=weight_decay)

                param_state = self.state[p]
                buf = param_state.get('momentum_buffer', torch.zeros_like(p.data))

                buf.mul_(momentum).add_(d_p)
                d_p = buf
                param_state['momentum_buffer'] = buf

                p.data.add_(d_p, alpha=-group['lr'])

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}

        for m in self.modules:
            if self.steps % self.Tinv == 0:
                self._update_inv(m)

            p_grad_mat = reshape_grad(m)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v

        self._kl_clip_and_update_grad(updates, lr)
        self._step(closure)
        self.steps += 1
