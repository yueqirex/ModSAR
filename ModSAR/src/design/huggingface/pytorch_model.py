import os 
DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(DIR)

import torch
import torch.nn as nn


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

        # transformer model 
        self.transformer = self.get_transformer(args)


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

        # print(self.args)
        # format input into huggingface style
        inputs = {'input_ids': x, 'attention_mask': mask}
        if self.args.model_type=='transfo_xl':
            inputs = {'input_ids': x}

        # get contextual token embeddings
        x = self.transformer(**inputs)[0] # (batch_size, max_position, hidden_size)
        return x

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

        # logits from item embedding and bias

        # if candidates is None, we compute logits for all items
        if candidates is None:
            w = self.transformer.item_embed.weight.transpose(1,0) # (embed_size, num_items+2), or embed_size // 2 if concat_user is True, same below
            logits = x @ w # (*, num_items+2)
        
        # if candidates is a single item, we compute logits for this item
        elif len(candidates.shape) == 1:
            w = self.transformer.item_embed(candidates) # (*, embed_size)
            logits = (x * w).sum(dim=-1, keepdim=True) # (*, 1)
        
        # if candidates is a list of items, we compute logits for these items
        elif len(candidates.shape) == 2:
            x = x.unsqueeze(dim=1) # (*, 1, embed_size)
            w = self.transformer.item_embed(candidates).transpose(2,1) # (*, embed_size, candidates)
            logits = (torch.bmm(x, w)).squeeze(dim=1) # (*, candidates)

        return logits.squeeze(dim=-1)
    

    def get_transformer(self, args):
        """ Get transformer model from huggingface.
        Args:
            args (argparse.Namespace): arguments
        Returns:
            transformer (huggingface model): transformer model
        """

        if args.model_type.lower() == 'bert':
            if args.task_type == 'ae':
                from .ori_transformers import BertConfig, BertModel
                print('Using huggingface BERT model for AE task.')
                
            elif args.task_type == 'ar':
                from .our_transformers import BertConfig, BertModel
                print('Using our modified huggingface BERT model for AR task.')

            config = BertConfig(
                vocab_size=args.num_items+2,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=args.hidden_size*4,
                max_position_embeddings=args.max_position,
                hidden_act='gelu',
                hidden_dropout_prob=args.dropout_prob,
                attention_probs_dropout_prob=args.dropout_prob,
                type_vocab_size=1,
            )
            model = BertModel(config)
            model.item_embed = model.embeddings.word_embeddings
            return model

        if args.model_type.lower() == 'albert':
            if args.task_type == 'ae':
                from .ori_transformers import AlbertConfig, AlbertModel
                print('Using huggingface BERT model for AE task.')
                
            elif args.task_type == 'ar':
                from .our_transformers import AlbertConfig, AlbertModel
                print('Using our modified huggingface BERT model for AR task.')

            config = AlbertConfig(
                vocab_size=args.num_items+2,
                embedding_size = args.hidden_size,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=args.hidden_size*4,
                hidden_act='gelu',
                hidden_dropout_prob=args.dropout_prob,
                attention_probs_dropout_prob=args.dropout_prob,
                max_position_embeddings=args.max_position,
                type_vocab_size=1,
            )
            model = AlbertModel(config)
            model.item_embed = model.embeddings.word_embeddings
            return model
        
        elif args.model_type.lower() == 'gpt2':
            if args.task_type == 'ae':
                from .our_transformers import GPT2Config, GPT2Model
                print('Using our modified huggingface GPT2 model for AE task.')

            elif args.task_type == 'ar':
                from .ori_transformers import GPT2Config, GPT2Model
                print('Using huggingface GPT2 model for AR task.')

            config = GPT2Config(
                vocab_size=args.num_items+2,
                n_positions=args.max_position,
                n_ctx=args.max_position,
                n_embd=args.hidden_size,
                n_layer=args.num_hidden_layers,
                n_head=args.num_attention_heads,
                activation_function='gelu',
                resid_pdrop=args.dropout_prob,
                embd_pdrop=args.dropout_prob,
                attn_pdrop=args.dropout_prob,
            )
            model = GPT2Model(config)
            model.item_embed = model.wte
            return model
        
        elif args.model_type.lower() == 'llama':
            if args.task_type == 'ae':
                from .our_transformers import LlamaConfig, LlamaModel
                print('Using our modified huggingface Llama model for AE task.')

            elif args.task_type == 'ar':
                from .ori_transformers import LlamaConfig, LlamaModel
                print('Using huggingface Llama model for AR task.')

            config = LlamaConfig(
                vocab_size=args.num_items+2,
                hidden_size=args.hidden_size,
                intermediate_size=4*args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                hidden_act='gelu',
                max_position_embeddings=args.max_position,
            )
            model = LlamaModel(config)
            model.item_embed = model.embed_tokens
            return model
        
        elif args.model_type.lower() == 'transfo_xl':
            if args.task_type == 'ae':
                from .our_transformers import TransfoXLConfig, TransfoXLModel
                print('Using our modified huggingface TransfoXL model for AE task.')

            elif args.task_type == 'ar':
                from .ori_transformers import TransfoXLConfig, TransfoXLModel
                print('Using huggingface TransfoXL model for AR task.')

            config = TransfoXLConfig(
                vocab_size=args.num_items+2,
                div_val=1,
                # cutoffs=[args.max_freq // 4 * i for i in range(1, 4)],
                adaptive=False,
                d_model=args.hidden_size,
                d_embed=args.hidden_size,
                n_head=args.num_attention_heads,
                d_head=int(args.hidden_size/args.num_attention_heads),
                d_inner=4*args.hidden_size,
                per_norm=True,
                n_layer=args.num_hidden_layers,
                mem_len=200,
                clamp_len=200,
                dropout=args.dropout_prob,
                dropatt=args.dropout_prob,
            )
            model = TransfoXLModel(config)
            model.item_embed = model.word_emb.emb_layers[0]
            return model
        
        else:
            raise Exception('model type specified incorrectly')