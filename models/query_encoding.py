import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DefaultQueryEncoding(nn.Module):
    def __init__(self, num_queries, hidden_dim) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, targets):
        return self.query_embed.weight


class RectangleQueryEncoding(nn.Module):
    def __init__(self, num_queries, hidden_dim, shuffle=False):
        super().__init__()
        self.num_queries = num_queries
        self.rectangle_encoder = MLP(4, hidden_dim, hidden_dim, 1)
        self.shuffle = shuffle

    def forward(self, batch):
        # get max len of  the elements in a batch
        e = "boxes"
        ml = max([len(b[e]) for b in batch])
        if ml == 0:
            ml = 1  # assert ML > 0
        for b in batch:
            feat = torch.cat([b[e], b[e].new_zeros(ml - b[e].size(0), b[e].size(1))], 0)
            # here we actually forward the stuff
            feat = self.rectangle_encoder(feat)
            if self.shuffle:
                idx = torch.randperm(feat.shape[0])
                feat = feat[idx].view(feat.size())
            b["boxfeat"] = feat
        out = torch.stack([b["boxfeat"] for b in batch], 1)

        return out


class CentreQueryEncoding(nn.Module):
    def __init__(self, num_queries, hidden_dim):
        super().__init__()
        self.num_queries = num_queries
        self.centre_encoder = MLP(2, hidden_dim, hidden_dim, 1)

    @staticmethod
    def _get_midpoint(bbox):
        assert bbox.size(1) == 4
        center = torch.stack(
            [(bbox[..., 0] + bbox[..., 2]) / 2, (bbox[..., 1] + bbox[..., 3]) / 2], 0
        ).transpose(1, 0)
        return center

    def forward(self, batch):
        # get max len of  the elements in a batch
        e = "boxes"
        ml = max([len(b[e]) for b in batch])
        if ml == 0:
            ml = 1  # assert ML > 0
        for b in batch:
            b["center"] = CentreQueryEncoding._get_midpoint(b[e])
            feat = torch.cat(
                [
                    b["center"],
                    b["center"].new_zeros(
                        ml - b["center"].size(0), b["center"].size(1)
                    ),
                ],
                0,
            )
            # here we actually forward the stuff
            feat = self.centre_encoder(feat)
            b["center"] = feat
        out = torch.stack([b["center"] for b in batch], 1)
        return out


class CentreAreaQueryEncoding(nn.Module):
    def __init__(self, num_queries, hidden_dim, norm=False):
        super().__init__()
        self.num_queries = num_queries
        self.norm = norm
        self.centre_encoder = MLP(3, hidden_dim, hidden_dim, 1)

    def forward(self, batch):
        # get max len of  the elements in a batch
        ml = max([len(b["boxes"]) for b in batch])
        if ml == 0:
            ml = 1  # assert ML > 0
        for b in batch:
            b["centers"] = CentreQueryEncoding._get_midpoint(b["boxes"])
            b["area_feat"] = b["area"].unsqueeze(1)
            if self.norm:
                b["area_feat"] = b["area_feat"] / torch.prod(b["size"])
            feat_box = torch.cat(
                [
                    b["centers"],
                    b["centers"].new_zeros(
                        ml - b["centers"].size(0), b["centers"].size(1)
                    ),
                ],
                0,
            )
            feat_area = torch.cat(
                [
                    b["area_feat"],
                    b["area_feat"].new_zeros(
                        ml - b["area_feat"].size(0), b["area_feat"].size(1)
                    ),
                ],
                0,
            )
            feat = torch.cat([feat_box, feat_area], 1)
            # here we actually forward the stuff
            feat = self.centre_encoder(feat)
            b["box_area"] = feat
        out = torch.stack([b["box_area"] for b in batch], 1)
        return out


def build_query_encoding(args):
    if args.query_encoding == "default":
        return DefaultQueryEncoding(args.num_queries, args.hidden_dim)
    elif args.query_encoding == "b1":
        return RectangleQueryEncoding(
            args.num_queries, args.hidden_dim, shuffle=args.query_shuffle
        )
    elif args.query_encoding == "b2":
        return CentreQueryEncoding(args.num_queries, args.hidden_dim)
    elif args.query_encoding == "b3":
        return CentreAreaQueryEncoding(args.num_queries, args.hidden_dim)
    elif args.query_encoding == "b3_norm":
        return CentreAreaQueryEncoding(args.num_queries, args.hidden_dim, True)
