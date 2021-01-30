import torch
import torch.nn.functional as F
import torch.nn as nn


class AvgPoolModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, tensors):
        src, mask, pos = tensors
        out = src + pos
        out = self.pool(out)

        bs = out.size(0)
        mask = torch.zeros((bs, 1, 1)).to(torch.bool).to(out.device)

        return (out, mask, None)


def build_pooling_module(args):
    if args.pooling_method in ["none", "avghack", ""]:
        return nn.Identity()
    elif args.pooling_method in ["avgpool", "avg"]:
        return AvgPoolModule()
