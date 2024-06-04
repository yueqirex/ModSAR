import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn import LayerNorm

import math

MAX_VAL = 1e12
MIN_VAL = 1e-12

# ------------------------
# Loss
# ------------------------

class Loss(torch.nn.Module):
    def __init__(self, loss_type):
        """ Initilize loss according to the given type from {bpr, ce, bce}.
        """
        super().__init__()

        if loss_type == 'bpr':
            self.loss = torch.nn.Softplus()
        elif loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, *x):
        """ Reuse `self.loss` to get the mean of loss value. 

            If loss_type == `bpr`, check `Softplus <https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html>_` forward function.
            If loss_type == `ce`, check `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>_` forward function.
            If loss_type == `bce`, check `BCEWithLogitsLoss <https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss>_` forward function.
        """

        return self.loss(*x).mean()


# ----------------
# Model
# ----------------

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # args
        self.args = args

        # embeddings module to obtain item, (user and position) embeddings
        self.embedding = Embedding(
            vocab_size=args.num_items + 2, 
            user_size=args.num_users,
            embed_size=args.hidden_size, 
            max_len=args.max_position, 
            pad_token=args.pad_token,
            dropout=args.dropout_prob,
            is_pos=(args.locker_type not in ['initial', 'adapt']),
            is_user=args.concat_user,
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(
                args.hidden_size, 
                args.num_attention_heads, 
                args.hidden_size * 4,
                args.dropout_prob, 
                args.dropout_prob, 
                max_len=args.max_position, 
                is_casual=(args.task_type == 'ar'),
                args=args
            ) for _ in range(args.num_hidden_layers)]
        )

        self.last_norm = LayerNorm(args.hidden_size)

        self.aggregator = Aggregator(
            args.hidden_size,
            args.aggregation_type,
            args.num_output_layers,
        )

        # initialize
        self._init_weights()

        # item bias
        self.bias = torch.nn.Embedding(num_embeddings=args.num_items + 2, embedding_dim=1) # (num_items+2, 1)
        self.bias.weight.data.fill_(0)

    def _init_weights(self):
        """ Initialize the model weights
        """
        for param in self.parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
                # torch.nn.init.constant_(param.data, 0.1) # DONE: turn off
            except:
                pass

    def forward(self, u, x):
        """ Forward function to obtain contextual token embeddings.

            Args: 
                u (torch.LongTensor): user ids with size of (batch_size, )
                x (torch.LongTensor): user sequence ids with size of (batch_size, max_position, )

            Returns:
                x (torch.Tensor): sequence of contexutal token embeddings after transformer blocks, with size of (batch_size, max_position, hidden_size) 
        """

        # mask for padding tokens
        mask = (x != self.args.pad_token)

        if self.args.locker_type == 'adapt':
            # mask for FISM (potentially used in locker-adapt)
            mask_fism = ((x != self.args.pad_token) & (x != self.args.mask_token)).unsqueeze(-1)

        # obtain item, (user and position) embeddings
        x = self.embedding(x, user=u)

        if self.args.locker_type == 'adapt':
            # obtain user feature (potentially used in locker-adapt)
            users = x.masked_fill(mask_fism==False, 0).sum(dim=-2, keepdim=True) / (mask_fism.sum(dim=-2, keepdim=True))
            u = users.repeat(1, x.size(1), 1)

        # running over multiple transformer blocks
        xs = []
        for transformer in self.transformer_blocks:
            x = transformer(x, mask, None, self.args, users=u)
            xs.append(x)
        
        # add layernorm to the output embeddings from the last layer
        xs[-1] = self.last_norm(xs[-1]) 

        # aggregate x from different layers
        return self.aggregator(xs)

    def predict(self, u, x, candidates=None, use_bias=False):
        """ Predict function to obtain logits for item candidates (or all items if candidates=None).

            Args:
                u (torch.LongTensor): user ids with size of (batch_size, )
                x (torch.Tensor): sequence of contexutal token embeddings from `forward`, with size of (\*, hidden_size), \* can be batch_size or others.
                candidates (torch.LongTensor, optional): candidate item ids, default=None. The sizes can be (\*, ) or (\*, num_candidates),  \* can be batch_size or others.

            Returns:
                logits (torch.Tensor): 
                    - if candidates is None: logits with the size of (\*, num_items+2), which is working for all-item ranking.
                    - if candidates size is (\*, ): it means we input single candidate per prediction, so obtain logits with the size of (\*, ).
                    - if candidates size is (\*, num_candidates): it means we input multiple candidates per prediction, so obtain logits with the size of (\*, num_candidates).
        """

        # logits from user embedding if concat_user is True
        logits_from_user = 0

        if self.args.concat_user: 
            x_u = x[..., -self.args.hidden_size // 2:] # (*, embed_size // 2)
            x = x[..., :self.args.hidden_size // 2] # (*, embed_size // 2)
            logits_from_user = (x_u * self.embedding.user(u)).sum(dim=-1, keepdim=True) # (*, 1)

        # logits from item embedding and bias

        # if candidates is None, we compute logits for all items
        if candidates is None:
            w = self.embedding.token.weight.transpose(1,0) # (embed_size, num_items+2), or embed_size // 2 if concat_user is True, same below
            bias = self.bias.weight.transpose(1,0) if use_bias else 0 # (1, num_items+2)
            logits = x @ w + bias # (*, num_items+2)
        
        # if candidates is a single item, we compute logits for this item
        elif len(candidates.shape) == 1:
            w = self.embedding.token(candidates) # (*, embed_size)
            bias = self.bias(candidates) if use_bias else 0 # (*, 1)
            logits = (x * w).sum(dim=-1, keepdim=True) + bias # (*, 1)
        
        # if candidates is a list of items, we compute logits for these items
        elif len(candidates.shape) == 2:
            x = x.unsqueeze(dim=1) # (*, 1, embed_size)
            w = self.embedding.token(candidates).transpose(2,1) # (*, embed_size, candidates)
            bias = self.bias(candidates).transpose(2,1) if use_bias else 0 # (*, candiadates)
            logits = (torch.bmm(x, w) + bias).squeeze(dim=1) # (*, candidates)

        return (logits + logits_from_user).squeeze(dim=-1)
    

# ----------------
# Embedding
# ----------------


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, pad_token=None):
        super().__init__(vocab_size, embed_size, padding_idx=pad_token)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class Embedding(nn.Module):
    """
    Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """
    def __init__(self, vocab_size, user_size, embed_size, max_len, pad_token, dropout=0.1, is_pos=True, is_user=False):
        super().__init__()

        # hyper parameters
        self.is_pos = is_pos
        self.is_user = is_user
        self.scale = embed_size ** 0.5
        
        # item and user embedding
        if is_user:
            assert embed_size % 2 == 0, "embed_size must be even number"
            self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size // 2, pad_token=pad_token)
            self.user = TokenEmbedding(vocab_size=user_size, embed_size=embed_size // 2)
        else:
            self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, pad_token=pad_token)
        
        # positional embedding
        if is_pos:
            self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        
        # dropout and layer normalization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, user=None):
        x = self.token(sequence) * self.scale

        if self.is_user:
            u = self.user(user).unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat([x, u], dim=-1)
        if self.is_pos:
            x = x + self.position(sequence)
        return self.dropout(x)
    
# -----------------
# Transformer Block
# -----------------


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, att_dropout=0.2, residual=True, activate="relu", max_len=50, is_casual=False, args=None):
        super().__init__()

        self.layernorm_first = args.layernorm_first
        self.is_causal = is_casual

        self.attn = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=att_dropout, max_len=max_len, args=args)
        self.attn_layernorm = LayerNorm(hidden)

        self.fwd = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, activate=activate)
        self.fwd_layernorm = LayerNorm(hidden)

    def forward(self, x, mask, stride=None, args=None, users=None):

        # mask for future tokens (only for autoregressive tasks)
        causal_mask = torch.tril(torch.ones(x.size(1), x.size(1))).bool().to(x.device) if self.is_causal else None

        if self.layernorm_first:
            x_ = self.attn_layernorm(x)
            x = x_ + self.attn(x_, x_, x_, key_padding_mask=mask, attn_mask=causal_mask, stride=stride, args=args, users=users)
            x_ = self.fwd_layernorm(x)
            x = x_ + self.fwd(x_)

        else:
            x = x + self.attn(x, x, x, key_padding_mask=mask, attn_mask=causal_mask, stride=stride, args=args, users=users)
            x = self.attn_layernorm(x)
            x = x + self.fwd(x)
            x = self.fwd_layernorm(x)
        return x
    

class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, dropout=0.1, max_len=50, args=None):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(n=max_len, d=d_model, d_k=self.d_k, args=args)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, output=True, stride=None, args=None, users=None):

        batch_size, max_len, _ = query.shape

        # 0) Define masking
        mask = key_padding_mask.unsqueeze(1).repeat(1, max_len, 1).unsqueeze(1) # (batch_size, 1, max_position, max_position)
        if attn_mask is not None:
            mask = mask & attn_mask.unsqueeze(0).unsqueeze(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout, args=args, users=users)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    def __init__(self, n, d, d_k, args):
        super().__init__()

        if args.locker_type == "conv":
            self.attn = ConvAttention(n, d, d_k, args=args)

        elif args.locker_type == "rnn":
            self.attn = RNNAttention(n, d, d_k, args=args)

        elif args.locker_type == "win":
            self.attn = WindowAttention(n, d, d_k, args=args)

        elif args.locker_type == "initial":
            self.attn = InitialAttention(n, d, d_k, args=args)

        elif args.locker_type == "adapt":
            self.attn = AdaptAttention(n, d, d_k, args=args)

        else:
            print("Attention type not found, using vanilla attention")
            self.attn = VanillaAttention()
        
    "Compute 'Scaled Dot Product Attention"
    def forward(self, query, key, value, mask=None, dropout=None, args=None, users=None):
        return self.attn(query=query, key=key, value=value, mask=mask, dropout=dropout, args=args, users=users)


# ----------------
# Utils
# ----------------

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, activate="gelu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = GELU() if activate=="gelu" else nn.ReLU()

    def forward(self, x):
        return self.dropout2(self.w_2(self.activation(self.dropout1(self.w_1(x)))))

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# ----------------
# Aggregator
# ----------------

class Aggregator(nn.Module):
    def __init__(self, hidden_size, aggregation_type, num_output_layers):
        super().__init__()
        self.aggregation_type = aggregation_type
        self.output_layers = self._mlp([hidden_size for _ in  range(num_output_layers)])

    def _mlp(self, layers, last_activate=True):
        # build `nn.Sequential` mlp from layer list
        mlp = []
        for i in range(len(layers) - 1):
            m = nn.Linear(layers[i], layers[i+1])
            mlp.append(m)
            mlp.append(nn.LeakyReLU())
        if last_activate:
            return nn.Sequential(*mlp)
        else:
            return nn.Sequential(*mlp[:-1])

    def forward(self, xs):
        # aggregate the output from x layers
        if self.aggregation_type == 'last':
            x = xs[-1]
        elif self.aggregation_type == 'mean':
            x = torch.cat([x.unsqueeze(dim=1) for x in xs], dim=1).mean(dim=1)
        elif self.aggregation_type == 'max':
            x = torch.cat([x.unsqueeze(dim=1) for x in xs], dim=1).max(dim=1)[0]
        # num_output_layer
        return self.output_layers(x)


# ------------------
# Attention Variants
# ------------------

class VanillaAttention(nn.Module):
    "Compute 'Scaled Dot Product Attention"

    def forward(self, query, key, value, mask=None, dropout=None, **kwargs):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -MAX_VAL)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn 

class ConvAttention(nn.Module):
    def __init__(self, n, d, d_k, args):
        super().__init__()
        self.n = n
        self.h = d // d_k
        self.args = args

        self.global_num = args.num_attention_heads - self.args.locker_config['num_attention_heads']
        self.local_num = self.args.locker_config['num_attention_heads']
        assert self.global_num >= 0 & self.local_num >=0, "0 <= Locker attention heads <= Global num_attention_heads"

        self.local_attns = nn.ModuleList([self.init_conv(d_k, 2*self.args.locker_config['init_val']+1) for _ in range(self.local_num)])

    def init_conv(self, channels, kernel_size=3):
        assert (kernel_size-1) % 2 == 0
        kernel_size = int(kernel_size)
        return nn.Sequential(
            torch.nn.Conv1d(
                in_channels = channels,
                out_channels = channels,
                kernel_size = kernel_size,
                padding = (kernel_size-1) // 2
            ),
            torch.nn.ReLU())

    def forward(self, query, key, value, mask=None, dropout=None, **kwargs):
        query_g, key_g, value_g, value_l = query[:, :self.global_num, ...], key[:, :self.global_num, ...], value[:, :self.global_num, ...], value[:, self.global_num:, ...]

        if self.global_num > 0:
            scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) / math.sqrt(query_g.size(-1))
            scores_g = scores_g.masked_fill(mask == 0, -MAX_VAL)
            p_attn_g = dropout(F.softmax(scores_g, dim=-1))
            value_g = torch.matmul(p_attn_g, value_g)

        p_attn_l = None
        value_l = torch.cat([self.local_attns[i](value_l[:, i, ...].squeeze().permute(0, 2, 1)).unsqueeze(1).permute(0,1,3,2) for i in range(self.local_num)], dim=1)

        if self.global_num > 0:
            return torch.cat([value_g, value_l], dim=1), (p_attn_g, p_attn_l)
        else:
            return value_l, p_attn_l
        

class RNNAttention(nn.Module):
    def __init__(self, n, d, d_k, args):
        super().__init__()
        self.n = n
        self.h = d // d_k
        self.args = args

        self.global_num = args.num_attention_heads - self.args.locker_config['num_attention_heads']
        self.local_num = self.args.locker_config['num_attention_heads']
        assert self.global_num >= 0 & self.local_num >=0, "0 <= Locker attention heads <= Global num_attention_heads"

        self.args.init_val = args.locker_config['init_val']
        position_ids_l = torch.arange(self.args.max_position, dtype=torch.long).view(-1, 1)
        position_ids_r = torch.arange(self.args.init_val+1, dtype=torch.long).view(1, -1)
        self.distance = position_ids_l + position_ids_r
        self.local_attns = nn.ModuleList([nn.GRU(input_size=d_k, hidden_size=d_k, num_layers=1, batch_first=True) for _ in range(self.local_num)])


    def forward(self, query, key, value, mask=None, dropout=None, **kwargs):
        b, _, l, d_k = query.size()
        query_g, key_g, value_g, value_l = query[:, :self.global_num, ...], key[:, :self.global_num, ...], value[:, :self.global_num, ...], value[:, self.global_num:, ...]

        if self.global_num > 0:
            scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) / math.sqrt(query_g.size(-1))
            scores_g = scores_g.masked_fill(mask == 0, -MAX_VAL)
            p_attn_g = dropout(F.softmax(scores_g, dim=-1))
            value_g = torch.matmul(p_attn_g, value_g)

        value_l = torch.cat([value_l, torch.zeros(size=(b, self.local_num, self.args.init_val, d_k)).to(value_l.device)], dim=-2)
        value_aug = value_l[:, :, self.distance.to(value_l.device), :]
        h_0 = torch.zeros(1, b * l * self.local_num, d_k).to(value_l.device)
        p_attn_l = None
        value_l = torch.cat([self.local_attns[i](value_aug.view(-1, self.args.init_val+1, d_k), h_0)[-1].view(b, self.local_num, l, d_k) for i in range(self.local_num)], dim=1)

        if self.global_num > 0:
            return torch.cat([value_g, value_l], dim=1), (p_attn_g, p_attn_l)
        else:
            return value_l, p_attn_l
        
class WindowAttention(nn.Module):
    def __init__(self, n, d, d_k, args):
        super().__init__()
        self.n = n
        self.h = d // d_k
        self.args = args

        self.global_num = args.num_attention_heads - self.args.locker_config['num_attention_heads']
        self.local_num = self.args.locker_config['num_attention_heads']
        assert self.global_num >= 0 & self.local_num >=0, "0 <= Locker attention heads <= Global num_attention_heads"

        position_ids_l = torch.arange(n, dtype=torch.long).view(-1, 1)
        position_ids_r = torch.arange(n, dtype=torch.long).view(1, -1)
        self.distance = (position_ids_r - position_ids_l).abs()
        self.window_size = args.locker_config['init_val']

    def forward(self, query, key, value, mask=None, dropout=None, **kwargs):
        query_g, key_g, value_g = query[:, :self.global_num, ...], key[:, :self.global_num, ...], value[:, :self.global_num, ...]
        query_l, key_l, value_l = query[:, self.global_num:, ...], key[:, self.global_num:, ...], value[:, self.global_num:, ...]

        if self.global_num > 0:
            scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) / math.sqrt(query_g.size(-1))
            scores_g = scores_g.masked_fill(mask == 0, -MAX_VAL)
            p_attn_g = dropout(F.softmax(scores_g, dim=-1))
            value_g = torch.matmul(p_attn_g, value_g)

        scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))

        mask = mask & (self.distance.to(scores_l.device) <= self.window_size)

        scores_l = scores_l.masked_fill(mask == 0, -MAX_VAL)
        p_attn_l = dropout(F.softmax(scores_l, dim=-1))
        value_l = torch.matmul(p_attn_l, value_l)

        if self.global_num > 0:
            return torch.cat([value_g, value_l], dim=1), (p_attn_g, p_attn_l)
        else:
            return value_l, p_attn_l
        

class InitialAttention(nn.Module):
    def __init__(self, n, d, d_k, args):
        super().__init__()
        self.n = n
        self.h = d // d_k
        self.args = args

        self.global_num = args.num_attention_heads - self.args.locker_config['num_attention_heads']
        self.local_num = self.args.locker_config['num_attention_heads']
        assert self.global_num >= 0 & self.local_num >=0, "0 <= Locker attention heads <= Global num_attention_heads"

        self.abs_pos_emb_key = nn.Embedding(n, d_k * self.local_num) 
        self.abs_pos_emb_query = nn.Embedding(n, d_k * self.local_num) 
        self.rel_pos_score = nn.Embedding(2 * n - 1, self.local_num) 
        sigma, alpha = self.args.locker_config['sigma'], self.args.locker_config['alpha']
        x = torch.arange(2 * n - 1) - n
        init_val = (alpha * (torch.exp(-((x/sigma)**2) / 2))).unsqueeze(-1).repeat(1, self.local_num)
        self.rel_pos_score.weight.data = init_val
        position_ids_l = torch.arange(n, dtype=torch.long).view(-1, 1)
        position_ids_r = torch.arange(n, dtype=torch.long).view(1, -1)
        self.distance = position_ids_r - position_ids_l + n - 1

    def forward(self, query, key, value, mask=None, dropout=None, **kwargs):
        query_g, key_g, value_g = query[:, :self.global_num, ...], key[:, :self.global_num, ...], value[:, :self.global_num, ...]
        query_l, key_l, value_l = query[:, self.global_num:, ...], key[:, self.global_num:, ...], value[:, self.global_num:, ...]

        if self.global_num > 0:
            scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) / math.sqrt(query_g.size(-1))
            scores_g = scores_g.masked_fill(mask == 0, -MAX_VAL)
            p_attn_g = dropout(F.softmax(scores_g, dim=-1))
            value_g = torch.matmul(p_attn_g, value_g)

        scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))

        reweight = self.rel_pos_score(self.distance.to(scores_l.device)).unsqueeze(0).permute(0,3,1,2)
        scores_l = scores_l * (reweight / 0.1).sigmoid()
        scores_l = scores_l.masked_fill(mask == 0, -MAX_VAL)
        p_attn_l = dropout(F.softmax(scores_l, dim=-1))
        value_l = torch.matmul(p_attn_l, value_l)

        if self.global_num > 0:
            return torch.cat([value_g, value_l], dim=1), (p_attn_g, p_attn_l)
        else:
            return value_l, p_attn_l


class AdaptAttention(nn.Module):
    def __init__(self, n, d, d_k, args):
        super().__init__()
        self.n = n
        self.h = d // d_k
        self.args = args

        self.global_num = args.num_attention_heads - self.args.locker_config['num_attention_heads']
        self.local_num = self.args.locker_config['num_attention_heads']
        assert self.global_num >= 0 & self.local_num >=0, "0 <= Locker attention heads <= Global num_attention_heads"

        self.abs_pos_emb_key = nn.Embedding(n, d_k * self.local_num) 
        self.abs_pos_emb_query = nn.Embedding(n, d_k * self.local_num) 
        self.rel_pos_emb = nn.Embedding(2 * n - 1, d_k * self.local_num)
        self.user_proj = nn.Linear(d, d_k * self.local_num)
        position_ids_l = torch.arange(n, dtype=torch.long).view(-1, 1)
        position_ids_r = torch.arange(n, dtype=torch.long).view(1, -1)
        self.distance = position_ids_r - position_ids_l + n - 1
        self.mlps = nn.ModuleList([nn.Linear(d_k, 1) for _ in range(self.local_num)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, key, value, mask=None, dropout=None, users=None, **kwargs):
        b, _, l, d_k = query.size()

        query_g, key_g, value_g = query[:, :self.global_num, ...], key[:, :self.global_num, ...], value[:, :self.global_num, ...]
        query_l, key_l, value_l = query[:, self.global_num:, ...], key[:, self.global_num:, ...], value[:, self.global_num:, ...]

        if self.global_num > 0:
            scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) / math.sqrt(query_g.size(-1))
            scores_g = scores_g.masked_fill(mask == 0, -MAX_VAL)
            p_attn_g = dropout(F.softmax(scores_g, dim=-1))
            value_g = torch.matmul(p_attn_g, value_g)

        scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))

        rel_pos_embedding = self.rel_pos_emb(self.distance.to(scores_l.device)).view(l, -1, self.local_num, d_k).permute(2,0,1,3).unsqueeze(0)
        inputs = rel_pos_embedding.repeat(b,1,1,1,1) + value_l.unsqueeze(dim=-2) + value_l.unsqueeze(dim=-3) + self.user_proj(users).view(b, l, -1, d_k).permute(0,2,1,3).unsqueeze(-2)

        reweight = torch.cat([self.mlps[i](inputs[:, i, ...]).squeeze(-1).unsqueeze(1) for i in range(self.local_num)], dim=1)
        scores_l = scores_l + reweight

        scores_l = scores_l.masked_fill(mask == 0, -MAX_VAL)
        p_attn_l = dropout(F.softmax(scores_l, dim=-1))
        value_l = torch.matmul(p_attn_l, value_l)

        if self.global_num > 0:
            return torch.cat([value_g, value_l], dim=1), (p_attn_g, p_attn_l)
        else:
            return value_l, p_attn_l