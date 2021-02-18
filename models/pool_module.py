import torch
import torch.nn.functional as F
import torch.nn as nn


class AvgPoolModule(nn.Module):
    def __init__(self, d_model, pool_conv=False, pool_norm=False):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = pool_conv
        self.norm = pool_norm
        if pool_conv:
            self.pool_conv = nn.Conv2d(d_model, d_model, 1)
            self.pos_conv = nn.Conv2d(d_model, d_model, 1)
        if pool_norm:
            self.pool_norm = nn.LayerNorm([d_model, 1, 1])

    def forward(self, tensors):
        src, mask, pos = tensors
        if self.conv:
            src = self.pool_conv(src)
            pos = self.pos_conv(pos)

        out = src + pos
        out = self.pool(out)

        if self.norm:
            out = self.pool_norm(out)

        bs = out.size(0)
        mask = torch.zeros((bs, 1, 1)).to(torch.bool).to(out.device)

        return (out, mask, None)


def build_pooling_module(args):
    if args.pooling_method in [
        "none",
        "avghack",
        "transformer_pool",
        "encoder_pool",
        "",
    ]:
        return nn.Identity()
    elif args.pooling_method in ["avgpool", "avg"]:
        return AvgPoolModule(args.hidden_dim, args.pool_conv, args.pool_norm)
