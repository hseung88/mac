from typing import List
from scipy.linalg import eigh
import re
import torch
import torch.nn as nn
from common.path import path_join, file_name
from utils.torch_utils import trainable_modules
from utils.utils import save_numpy, load_numpy
from utils.opt_utils import extract_patches
from functools import partial
import math


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
        self.attn = {}
        self.avg_attn = {}  # attn.mean(dim=0)

    def _capture_activation(
        self,
        net,
        layer_map,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        actv = forward_input[0].detach().clone()
        attn_qkv = ('attn.qkv' in layer_map[module]['name'])

        is_conv = isinstance(module, nn.Conv2d)

        if is_conv:
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride,
                                   module.padding, depthwise)
        elif actv.ndim > 2:  # linear layers in transformers
            if attn_qkv:
                B, N, D = actv.shape
            actv = actv.reshape(-1, actv.size(-1))

        if attn_qkv:
            qkv_out = _forward_output.detach().clone()
            # _forward_output is assumed to be [B, N, 3 * dim]
            B, N, three_dim = qkv_out.shape
            if hasattr(net, 'layers'): # for swin-transformer
                layer_name = layer_map[module]['name']
                match = re.search(r'layers\.(\d+)\.blocks\.(\d+)', layer_name)
                stage_idx = int(match.group(1))
                block_idx = int(match.group(2))
                num_heads = net.layers[stage_idx].blocks[block_idx].attn.num_heads
                head_dim = net.layers[stage_idx].blocks[block_idx].dim // num_heads
            elif hasattr(net, 'blocks'): # for deit
                num_heads = net.blocks[0].attn.num_heads
                head_dim = net.embed_dim // num_heads
            # Reshape and permute to get q, k, v separated.
            qkv = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)  # Each is [B, num_heads, N, head_dim]

            scale = 1.0 / math.sqrt(head_dim)
            R = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
            attn = torch.softmax(R, dim=-1)
            avg_attn = attn.mean(dim=(0, 1, 2))  # [N, ]

            if module in self.attn:
                self.attn[module] += attn
                self.avg_attn[module] += avg_attn
            else:
                self.attn[module] = attn
                self.avg_attn[module] = avg_attn

    def fit(self, model, data_module):
        train_loader = data_module.train_loader()

        model = model.to(self.device)

        state = torch.load(path_join(self.working_dir, self.pretrained_model_path))
        model.load_state_dict(state['model_state'])
        model.eval()

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

            h_fwd_hook = module.register_forward_hook(partial(self._capture_activation, net=net, layer_map=layer_map))
            layer_map[module] = {'name': _module_name, 'hook': h_fwd_hook}

        for images, _ in train_loader:
            images = images.to(self.device)
            net.zero_grad()
            net(images)

        # compute Eigenvalues of E[xx^t]
        save_path = path_join(self.working_dir, self.checkpoint_dir, self.checkpoint_name)

        for module, attn in self.attn.items():
            module_name = layer_map[module]['name']
            n = attn.size(0)
            n_batches = len(train_loader)

            # store activation-covariance matrix
            covmat_fname = f'{module_name}attn.pth'
            cov_mat = attn.cpu().numpy() / n_batches
            save_numpy(cov_mat, save_path, covmat_fname)

            # skip if the result already exists
            eigval_fname = f'{module_name}_attn_eigvals.pth'
            eigvec_fname = f'{module_name}_attn_eigvecs.pth'
            eigvals = load_numpy(save_path, eigval_fname)
            eigvecs = load_numpy(save_path, eigvec_fname)
            evd_performed = False
            if eigvals is None or eigvecs is None:
                eigvals, eigvecs = eigh(cov_mat,
                                        subset_by_index=[max(n-self.num_eigen_pairs, 0), n-1])
                evd_performed = True

            print(f"* {module_name} Eigenvalues:")
            print(f"{eigvals}\n")

            if self.config.save and evd_performed:
                save_numpy(eigvals, save_path, eigval_fname)
                save_numpy(eigvecs, save_path, eigvec_fname)

            if self.config.save:
                avg_attn = self.avg_attn[module].cpu().numpy() / n_batches
                save_numpy(avg_attn, save_path, f'{module_name}_attn_mean.pth')
