from scipy.linalg import eigh
import torch
import torch.nn as nn
from common.path import path_join, file_name
from utils.torch_utils import trainable_modules
from utils.utils import _batch_to_device, save_numpy, load_numpy


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


class EigenAnalyzer:
    """
    Perofrms the eigenvalue decomposition of activation covariance matrix
    To execute, run the following command:
    ```
    python main.py network=lenet5 trainer=act_eigendecomp save=True \
    trainer.pretrained_model_file='checkpoint_E25S4875.pt
    ```
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # path to .pt file
        self.checkpoint_dir = path_join(config.trainer.pretrained_model_dir, 'checkpoints')
        self.pretrained_model_path = path_join(self.checkpoint_dir,
                                               config.trainer.pretrained_model_file)
        self.checkpoint_name, _ = file_name(self.pretrained_model_path)
        self.num_eigen_pairs = config.trainer.num_eigen_pairs
        self.target_layers = config.trainer.target_layers
        self.working_dir = config.app_root
        self.preactv_cov = {}
        self.avg_outg = {}  # E[preactivation gradient], i.e., dL/dz

        # defined for compliance with other classes but not used
        self._epoch = 0
        self._steps = 0
        self._iters = 0

    def _capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
    ):
        # backprops = forward_output[0].detach().clone()
        # batch_size = backprops.size(0)
        # backprops = backprops.view(batch_size, -1)
        # avg_dldz = backprops.mean(0)
        # Z = backprops.t() @ (backprops / batch_size)

        g = forward_output[0].detach().clone()
        batch_size = g.size(0)
        is_conv = isinstance(module, nn.Conv2d)
        if is_conv:
            spatial_size = g.size(2) * g.size(3)
            g = g.transpose(1, 2).transpose(2, 3)

        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))
        if is_conv:
            g = g * spatial_size

        Z = g.t() @ (g / batch_size)
        avg_dldz = g.mean(0)

        if module in self.preactv_cov:
            self.preactv_cov[module] += Z
            self.avg_outg[module] += avg_dldz
        else:
            self.preactv_cov[module] = Z
            self.avg_outg[module] = avg_dldz

    def fit(self, model, data_module):
        train_loader = data_module.train_loader()

        model = model.to(self.device)

        state = torch.load(path_join(self.working_dir, self.pretrained_model_path))
        model.load_state_dict(state['model_state'])
        model.train()

        net = model.net

        # add hooks to capture the activations
        layer_map = {}
        idx = 0
        for _module_name, module in trainable_modules(net):
            # skip the normalization layer
            if isinstance(module,
                          (nn.BatchNorm2d, nn.GroupNorm)):
                continue

            check = 'O'
            if self.target_layers is not None:
                if idx not in self.target_layers:
                    check = 'X'

            idx_str = "[{0}]".format(idx)
            print(f"{idx_str:5s} {_module_name:20s} {check}")
            idx += 1

            if check == 'X':
                continue

            h_bwd_hook = module.register_full_backward_hook(self._capture_backprops_hook)
            layer_map[module] = {'name': _module_name, 'hook': h_bwd_hook}

        for batch_idx, batch in enumerate(train_loader):
            batch = _batch_to_device(batch, self.device)
            model.zero_grad()
            model.training_step(self._epoch, batch_idx, batch, do_step=False)

        # compute Eigenvalues
        save_path = path_join(self.working_dir, self.checkpoint_dir, self.checkpoint_name)

        for module, preactv_cov in self.preactv_cov.items():
            module_name = layer_map[module]['name']
            n = preactv_cov.size(0)
            n_batches = len(train_loader)

            # store preactivation-covariance matrix
            covmat_fname = f'{module_name}_preact.pth'
            cov_mat = preactv_cov.cpu().numpy() / n_batches
            save_numpy(cov_mat, save_path, covmat_fname)

            # skip if the result already exists
            eigval_fname = f'{module_name}_preact_eigvals.pth'
            eigvec_fname = f'{module_name}_preact_eigvecs.pth'
            eigvals = load_numpy(save_path, eigval_fname)
            eigvecs = load_numpy(save_path, eigvec_fname)
            evd_performed = False
            if eigvals is None or eigvecs is None:
                cov_mat = preactv_cov.cpu().numpy() / n_batches
                eigvals, eigvecs = eigh(cov_mat,
                                        subset_by_index=[max(n-self.num_eigen_pairs, 0), n-1])
                evd_performed = True

            print(f"* {module_name} Preactivation Eigenvalues:")
            print(f"{eigvals}\n")

            if self.config.save and evd_performed:
                save_numpy(eigvals, save_path, eigval_fname)
                save_numpy(eigvecs, save_path, eigvec_fname)

            if self.config.save:
                avg_dldg = self.avg_outg[module].cpu().numpy() / n_batches
                save_numpy(avg_dldg, save_path, f'{module_name}_preact_mean.pth')
