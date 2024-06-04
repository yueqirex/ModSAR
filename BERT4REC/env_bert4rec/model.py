import os
import math
import numpy as np

import torch
import pdb
from torch import nn as nn
import torch.nn.functional as F
from modules import BERTEmbedding, TransformerBlock, OutputLayer


class BERTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pad_token = 0

        vocab_size = args.num_items + 4
        n_layers = args.num_hidden_layers
        heads = args.num_attention_heads

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=args.hidden_size, max_len=args.max_position, dropout=args.dropout_prob)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args.hidden_size, heads, args.hidden_size * 4, args.dropout_prob, args.attention_dropout_prob) for _ in range(n_layers)])

        # output layer
        self.output = OutputLayer(args.hidden_size)

        # weights initialization
        self.init_weights()

        # bias for similarity calculation
        self.bias = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=1) # (num_items+2, 1)
        self.bias.weight.data.fill_(0)

    def forward(self, x):
        # embedding and mask creation
        tl=x.shape[1]
        mask = (x != self.pad_token).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        if self.args.causal_mask:
            look_ahead_mask = torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=x.device)).unsqueeze(0)
            mask = torch.logical_and(look_ahead_mask, mask)
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask) # change to total_mask

        return self.output(x)

    def init_weights(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            # compute bounds with CDF
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if ('norm' not in n) and ('bias' not in n):
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def similarity_score(self, x, candidates=None):
        if candidates is None: # training
            w = self.embedding.token.weight.transpose(1,0)
            bias = self.bias.weight.transpose(1,0)
            return torch.matmul(x, w) + bias
        if candidates is not None: # evaluation
            index = candidates
            w = self.embedding.token(index).transpose(2,1)
            bias = self.bias(index).transpose(2,1)
            return (torch.bmm(x, w) + bias).squeeze(1)