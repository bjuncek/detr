# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pool_in_sum=False,
        aggregate_t=False,
        pool_encoded_values=False,
        encoder_pool_sdim=-1,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            pool_in_sum,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        # here we add token parameter to use transformer to aggregate values into
        self.aggregate_t = aggregate_t
        if self.aggregate_t:
            # aggregation embedding is learned
            self.agg_emb = nn.Parameter(torch.rand(1, d_model))
            self.agg_pos = nn.Parameter(torch.rand(1, d_model))

        # this avg pools the values after encoder
        self.pool_encoder = pool_encoded_values

        self.encoder_pool_sdim = encoder_pool_sdim

        if self.encoder_pool_sdim > 0:
            self.encpool = nn.AdaptiveAvgPool2d(
                (self.encoder_pool_sdim, self.encoder_pool_sdim)
            )
            self.pool_conv = nn.Conv2d(d_model, d_model, 1)
            self.pool_pos = nn.Conv2d(d_model, d_model, 1)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.pool = pool_in_sum

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed: Optional[Tensor] = None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        # Bruno's modification: if there is nothing here, position has been added previously
        # to the source and it's highly questionable what the fuck will happen later on (yey)
        if pos_embed is not None:
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # bruno's modification for playing with querry embeddings
        if len(query_embed.size()) == 2:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)

        # if we're aggregating the model in the encoder
        # we need to add an additional parammetes that we're going to aggregate
        # over
        if self.aggregate_t:
            src = torch.cat([self.agg_emb.unsqueeze(0).repeat(1, bs, 1), src], dim=0)
            pos_embed = torch.cat(
                [self.agg_pos.unsqueeze(0).repeat(1, bs, 1), pos_embed], dim=0
            )
            mask_append = torch.zeros((bs, 1)).to(torch.bool).to(mask.device)
            mask = torch.cat([mask_append, mask], dim=1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # if we are aggregating the model in the encoder we need to select
        # just the outputs corresponding to the first element (self.agg_emb)
        if self.aggregate_t:
            memory = memory[0, ...].unsqueeze(0)
            pos_embed = pos_embed[0, ...].unsqueeze(0)
            mask = mask[..., 0].unsqueeze(1)

        # if we are simply pooling the model here
        if self.pool_encoder:
            memory = torch.mean(memory, dim=0, keepdim=True)
            # this is a hack, how to deal with this better
            pos_embed = torch.mean(memory, dim=0, keepdim=True)
            # and we're in always non-padded mode
            mask = torch.zeros((bs, 1)).to(torch.bool).to(mask.device)

        if self.encoder_pool_sdim > 0:
            pdim = self.encoder_pool_sdim * self.encoder_pool_sdim
            memory = memory.permute(1, 2, 0).view(bs, c, h, w)
            memory = self.encpool(self.pool_conv(memory))
            memory = memory.view(bs, c, pdim).permute(2, 0, 1)
            pos_embed = pos_embed.permute(1, 2, 0).view(bs, c, h, w)
            pos_embed = self.encpool(self.pool_pos(pos_embed))
            pos_embed = pos_embed.view(bs, c, pdim).permute(2, 0, 1)

            # and we're in always non-padded mode
            mask = torch.zeros((bs, pdim)).to(torch.bool).to(mask.device)
            h = w = self.encoder_pool_sdim

        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )

        if self.pool or self.aggregate_t or self.pool_encoder:
            h = w = 1

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pool_in_sum=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.pool = pool_in_sum

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        ret = tensor if pos is None else tensor + pos
        if self.pool:
            return torch.mean(ret, dim=0, keepdim=True)
        return ret

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        if self.pool:
            src = torch.mean(src, dim=0, keepdim=True)
            src_key_padding_mask = (
                torch.zeros((src.size(1), 1)).to(torch.bool).to(src.device)
            )
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.pool:
            src = torch.mean(src, dim=0, keepdim=True)
            src2 = torch.mean(src2, dim=0, keepdim=True)
            src_key_padding_mask = (
                torch.zeros((src.size(1), 1)).to(torch.bool).to(src.device)
            )
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        ret = tensor if pos is None else tensor + pos
        return ret

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # this is a dumb hack, but fuck it
        if (pos is not None) and (memory.size() != pos.size()):
            pos = torch.mean(pos, dim=0, keepdim=True)
            memory_key_padding_mask = (
                torch.zeros((memory.size(1), 1)).to(torch.bool).to(memory.device)
            )
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    pool = True if args.pooling_method in ["avghack"] else False
    aggregate = True if args.pooling_method in ["transformer_pool"] else False
    encoder_pool = True if args.pooling_method in ["encoder_pool"] else False
    encoder_dim = args.pooling_dim > 1
    assert sum([pool, aggregate, encoder_pool, encoder_dim]) <= 1

    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pool_in_sum=pool,
        aggregate_t=aggregate,
        pool_encoded_values=encoder_pool,
        encoder_pool_sdim=args.pooling_dim,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

