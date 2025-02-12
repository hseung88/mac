import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial

# Custom DeiT Model for CIFAR-100 (32x32 images)
class DeiT32(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=100, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super(DeiT32, self).__init__()
        self.deit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes
        )
        self.deit.default_cfg = _cfg()
        # Expose the patch embedding layer for potential optimizer grouping
        self.patch_embed = self.deit.patch_embed

    def forward(self, x):
        return self.deit(x)