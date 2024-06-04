from .pytorch_model import Model, Loss
from .pytorch_dataset import Data, Dataset

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torch.nn.functional as F

import multiprocessing
from argparse import Namespace

# ------------------------
# Data Lightning Wrapper
# ------------------------

class LitData(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, task_type, 
                 eval_neg, max_position, mask_prob, item_sse_prob, user_sse_prob,
                 num_users, num_items, pad_token, mask_token):
        super().__init__()

        # args
        self.args = locals()
        self.args = Namespace(**{k: self.args[k] for k in self.args.keys() if k != 'self'})
        self.save_hyperparameters()

    def setup(self, stage):
        # load data
        self.data = Data(self.args)
        # split dataset
        self.train = Dataset(self.data, 'train', self.args)
        self.valid = Dataset(self.data, 'valid', self.args)
        self.test = Dataset(self.data, 'test', self.args)

    def train_dataloader(self):
        # num_workers = int(multiprocessing.cpu_count()/2)
        return DataLoader(self.train, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        # num_workers = int(multiprocessing.cpu_count()/2)
        return DataLoader(self.valid, batch_size=self.args.batch_size, shuffle=False)

    def test_dataloader(self):
        # num_workers = int(multiprocessing.cpu_count()/2)
        return DataLoader(self.test, batch_size=self.args.batch_size, shuffle=False)


# ------------------------
# Model Lightning Wrapper
# ------------------------


class LitModel(pl.LightningModule):
    """ PyTorch-Lightning wrapper of modularized model, which can be used to hide 
        the useless details of model training, checkpointing, logging, multi-device etc.
    """

    def __init__(self, 
                 num_items, num_users, max_freq, pad_token, mask_token, max_position, 
                 model_type, hidden_size, num_attention_heads, num_hidden_layers,
                 loss_type, dropout_prob, task_type, 
                 optimizer_type, lr, weight_decay, eval_neg
        ):
        super().__init__()

        # args
        # self.args = locals()
        # self.args = Namespace(**{k: self.args[k] for k in self.args.keys() if k != 'self'})
        self.args = Namespace(num_items=num_items, num_users=num_users, max_freq=max_freq, pad_token=pad_token, 
            mask_token=mask_token, max_position=max_position, model_type=model_type, hidden_size=hidden_size, 
            num_attention_heads=num_attention_heads, loss_type=loss_type, dropout_prob=dropout_prob, 
            task_type=task_type, num_hidden_layers=num_hidden_layers, optimizer_type=optimizer_type, 
            lr=lr, weight_decay=weight_decay, eval_neg=eval_neg)
        self.save_hyperparameters()

        # model
        self.model = Model(self.args)

        # criterion
        self.criterion = Loss(loss_type)

    # -------------------
    # Forward & Predict
    # -------------------

    def forward(self, u, x):
        """ Forward function to obtain contextual token embeddings.

            Args: 
                u (torch.LongTensor): user ids with size of (batch_size, )
                x (torch.LongTensor): user sequence ids with size of (batch_size, max_position, )

            Returns:
                x (torch.Tensor): sequence of contexutal token embeddings after transformer blocks, with size of (batch_size, max_position, hidden_size) 
        """
        return self.model(u, x)

    def predict(self, u, x, candidates=None):
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

        return self.model.predict(u, x, candidates=candidates)

    # ---------------------
    # Training & Evaluation
    # ---------------------

    def configure_optimizers(self):
        """ Optimizer Configuration using ``args.optimizer_type``.

            Returns:
                optimizer (torch.optim.*): Optimizer class from pytorch with specified learning rate (args.lr).
        """
        if self.args.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(
                0.9, 0.98), weight_decay=self.args.weight_decay)
        elif self.args.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, betas=(
                0.9, 0.98), weight_decay=self.args.weight_decay)
        elif self.args.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, betas=(
                0.9, 0.98), weight_decay=self.args.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        """ Training step given the input batch, which comes from PyTorch Lightning ``test_step`` function.
            During training, ``self.log`` function is used to record important metrics (e.g., training loss)

            Returns:
                loss (torch.Tensor): returned loss with the shape of (1,)
        """

        if self.args.loss_type == 'bpr':
            loss = self._training_step_bpr(batch)
        elif self.args.loss_type == 'bce':
            loss = self._training_step_bce(batch)
        elif self.args.loss_type == 'ce':
            loss = self._training_step_ce(batch)
        self.log('train/loss', loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Validation step per batch, which comes from PyTorch Lightning ``validation_step`` function.
            During validation, ``self.log`` function is used to record important metrics (e.g., validation loss, NDCG)

            Args:
                batch (dict): dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:

                    - ``user (torch.LongTensor)``: user ids, the size is (batch_size, ),
                    - ``input_ids (torch.LongTensor)``: user sequence, the size is (batch_size, max_position),
                    - ``output_ids (torch.LongTensor)``: positive items for prediction, the size is (batch_size, max_position),
                    - ``neg_ids (torch.LongTensor)``: negative items for prediction, the size is (batch_size, max_position).

                batch_idx (int, optional): the index of input training batch

            Returns:
                metrics (dict): Dictionary of metrics of interest.
        """

        return self._evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        """ Testing step per batch, which comes from PyTorch Lightning ``test_step`` function.
            During testing, ``self.log`` function is used to record important metrics (e.g., testing loss, NDCG)

            Args:
                batch (dict): dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:

                    - ``user (torch.LongTensor)``: user ids, the size is (batch_size, ),
                    - ``input_ids (torch.LongTensor)``: user sequence, the size is (batch_size, max_position),
                    - ``output_ids (torch.LongTensor)``: positive items for prediction, the size is (batch_size, max_position),
                    - ``neg_ids (torch.LongTensor)``: negative items for prediction, the size is (batch_size, max_position).

                batch_idx (int, optional): the index of input training batch

            Returns:
                metrics (dict): Dictionary of metrics of interest.
        """

        return self._evaluation_step(batch)
    
    def validation_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, 'test')

    # internal util functions

    def _evaluation_step(self, batch):
        """ Internal evaluation step used by `validation_step` and `test_step`.
            Args:
                batch (dict): dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:

                    - ``user (torch.LongTensor)``: user ids, the size is (batch_size, ),
                    - ``input_ids (torch.LongTensor)``: user sequence, the size is (batch_size, max_position),
                    - ``output_ids (torch.LongTensor)``: positive items for prediction, the size is (batch_size, max_position),
                    - ``neg_ids (torch.LongTensor)``: negative items for prediction, the size is (batch_size, max_position).

                batch_idx (int, optional): the index of input training batch

            Returns:
                metrics (dict): Dictionary of metrics of interest.
        """
        
        # model forward
        u, seq, pos = batch['user'], batch['input_ids'], batch['output_ids']
        vec = self.forward(u, seq)  # (batch_size, max_position, embed_size)

        # select the positions of interest
        inds = self._get_eval_inds(seq)
        vec, pos = vec[inds], pos[inds]

        # select the users of interest (useful if concat_user is True) 
        u = u.unsqueeze(1).repeat(1, seq.size(1))[inds]

        # get candidate logits
        if self.args.eval_neg > 0:
            logits = self.predict(u, vec, batch['neg_ids'])
        else:
            logits = self.predict(u, vec)
            # exclude rated item logits
            # logits[torch.arange(0, u.size(0)).unsqueeze(1).to(u.device), seq] = float("-inf")
            logits = logits.masked_fill(F.one_hot(seq, logits.shape[-1]).sum(1).bool(), float(-2**20))

        # get pos logits
        pos_logits = self.predict(u, vec, pos).unsqueeze(-1)

        # metrics
        rank = (pos_logits < logits).sum(dim=-1).flatten()
        recall_1 = (rank < 1).float()
        recall_5 = (rank < 5).float()
        recall_10 =  (rank < 10).float() #(bs,)
        recall_20 =  (rank < 20).float() #(bs,)
        recall_50 =  (rank < 50).float() #(bs,)
        recall_100 =  (rank < 100).float() #(bs,)
        ndcg_5 = (1 / torch.log2(rank + 2)) * (rank < 5).float() # (bs,)
        ndcg_10 = (1 / torch.log2(rank + 2)) * (rank < 10).float() # (bs,)
        ndcg_20 = (1 / torch.log2(rank + 2)) * (rank < 20).float() # (bs,)
        ndcg_50 = (1 / torch.log2(rank + 2)) * (rank < 50).float() # (bs,)
        ndcg_100 = (1 / torch.log2(rank + 2)) * (rank < 100).float() # (bs,)
        mrr_1 = (1 / (rank + 1)) * (rank < 1).float()
        mrr_5 = (1 / (rank + 1)) * (rank < 5).float()
        mrr_10 = (1 / (rank + 1)) * (rank < 10).float()
        mrr_20 = (1 / (rank + 1)) * (rank < 20).float()
        mrr_50 = (1 / (rank + 1)) * (rank < 50).float()
        mrr_100 = (1 / (rank + 1)) * (rank < 100).float()
        metrics = torch.concat((recall_1.unsqueeze(0), recall_5.unsqueeze(0), recall_10.unsqueeze(0), \
                                recall_20.unsqueeze(0), recall_50.unsqueeze(0), recall_100.unsqueeze(0), \
                                ndcg_5.unsqueeze(0), ndcg_10.unsqueeze(0), ndcg_20.unsqueeze(0), \
                                ndcg_50.unsqueeze(0), ndcg_100.unsqueeze(0), \
                                mrr_1.unsqueeze(0), mrr_5.unsqueeze(0), mrr_10.unsqueeze(0), \
                                mrr_20.unsqueeze(0), mrr_50.unsqueeze(0), mrr_100.unsqueeze(0)),
                                dim=0) #(17,bs)
        metrics = metrics.transpose(0,1) # (bs,17)
        return metrics

    def _evaluation_epoch_end(self, outputs, mode='valid'):
        assert len(outputs) > 0
        outs_all = torch.cat(outputs, dim=0) #(bs*n_bs, 17)
        recall_1 = outs_all[:,0].mean()
        recall_5 = outs_all[:,1].mean()
        recall_10 = outs_all[:,2].mean() #(bs*n_bs,)
        recall_20 = outs_all[:,3].mean() #(bs*n_bs,)
        recall_50 = outs_all[:,4].mean() #(bs*n_bs,)
        recall_100 = outs_all[:,5].mean() #(bs*n_bs,)

        ndcg_5 = outs_all[:,6].mean()
        ndcg_10 = outs_all[:,7].mean() #(bs*n_bs,)
        ndcg_20 = outs_all[:,8].mean() #(bs*n_bs,)
        ndcg_50 = outs_all[:,9].mean() #(bs*n_bs,)
        ndcg_100 = outs_all[:,10].mean() #(bs*n_bs,)

        mrr_1 = outs_all[:,11].mean()
        mrr_5 = outs_all[:,12].mean()
        mrr_10 = outs_all[:,13].mean() #(bs*n_bs,)
        mrr_20 = outs_all[:,14].mean() #(bs*n_bs,)
        mrr_50 = outs_all[:,15].mean() #(bs*n_bs,)
        mrr_100 = outs_all[:,16].mean() #(bs*n_bs,)

        self.log(f'{mode}/Recall@1', recall_1, prog_bar=(mode=='valid'))
        self.log(f'{mode}/Recall@5', recall_5, prog_bar=(mode=='valid'))
        self.log(f'{mode}/Recall@10', recall_10, prog_bar=(mode=='valid'))
        self.log(f'{mode}/Recall@20', recall_20, prog_bar=(mode=='valid'))
        self.log(f'{mode}/Recall@50', recall_50, prog_bar=(mode=='valid'))
        self.log(f'{mode}/Recall@100', recall_100, prog_bar=(mode=='valid'))

        self.log(f'{mode}/NDCG@5', ndcg_5, prog_bar=(mode=='valid'))
        self.log(f'{mode}/NDCG@10', ndcg_10, prog_bar=(mode=='valid'))
        self.log(f'{mode}/NDCG@20', ndcg_20, prog_bar=(mode=='valid'))
        self.log(f'{mode}/NDCG@50', ndcg_50, prog_bar=(mode=='valid'))
        self.log(f'{mode}/NDCG@100', ndcg_100, prog_bar=(mode=='valid'))

        self.log(f'{mode}/MRR@1', mrr_1, prog_bar=(mode=='valid'))
        self.log(f'{mode}/MRR@5', mrr_5, prog_bar=(mode=='valid'))
        self.log(f'{mode}/MRR@10', mrr_10, prog_bar=(mode=='valid'))
        self.log(f'{mode}/MRR@20', mrr_20, prog_bar=(mode=='valid'))
        self.log(f'{mode}/MRR@50', mrr_50, prog_bar=(mode=='valid'))
        self.log(f'{mode}/MRR@100', mrr_100, prog_bar=(mode=='valid'))


    def _training_step_bpr(self, batch):
        """ Training step per batch with bpr loss, which comes from PyTorch Lightning ``training_step`` function.

            Args:
                batch (dict): dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:

                    - ``user (torch.LongTensor)``: user ids, the size is (batch_size, ),
                    - ``input_ids (torch.LongTensor)``: user sequence, the size is (batch_size, max_position),
                    - ``output_ids (torch.LongTensor)``: positive items for prediction, the size is (batch_size, max_position),
                    - ``neg_ids (torch.LongTensor)``: negative items for prediction, the size is (batch_size, max_position).

                batch_idx (int, optional): the index of input training batch

            Returns:
                loss (torch.Scalar): BPR loss for positive items and negative items both.
        """

        # model forward
        u, seq, pos, neg = batch['user'], batch['input_ids'], batch['output_ids'], batch['neg_ids']
        vec = self.forward(u, seq)  # (batch_size, max_position, embed_size)

        # select the positions of interest
        inds = self._get_training_inds(seq, pos)
        vec, pos, neg = vec[inds], pos[inds], neg[inds]

        # select the users of interest (useful if concat_user is True) 
        u = u.unsqueeze(1).repeat(1, seq.size(1))[inds]

        # get logits
        pos_logits = self.predict(u, vec, pos)
        neg_logits = self.predict(u, vec, neg)

        # calculate loss
        loss = self.criterion(neg_logits - pos_logits)

        # return
        return loss

    def _training_step_ce(self, batch):
        """ Training step per batch with ce loss.

            Args:
                batch (dict): dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:

                    - ``user (torch.LongTensor)``: user ids, the size is (batch_size, ),
                    - ``input_ids (torch.LongTensor)``: user sequence, the size is (batch_size, max_position),
                    - ``output_ids (torch.LongTensor)``: positive items for prediction, the size is (batch_size, max_position),
                    - ``neg_ids (torch.LongTensor)``: negative items for prediction, the size is (batch_size, max_position).

                batch_idx (int, optional): the index of input training batch

            Returns:
                loss (torch.Scalar): CE loss for positive items and negative items both.
        """

        # model forward
        u, seq, pos = batch['user'], batch['input_ids'], batch['output_ids']
        vec = self.forward(u, seq)  # (batch_size, max_position, embed_size)

        # select the positions of interest
        inds = self._get_training_inds(seq, pos)
        vec, pos = vec[inds], pos[inds]

        # get logits
        logits = self.predict(u, vec)

        # calculate loss
        loss = self.criterion(logits, pos)

        # return
        return loss

    def _training_step_bce(self, batch):
        """ Training step per batch with bce loss, which comes from PyTorch Lightning ``training_step`` function.

            Args:
                batch (dict): dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:

                    - ``user (torch.LongTensor)``: user ids, the size is (batch_size, ),
                    - ``input_ids (torch.LongTensor)``: user sequence, the size is (batch_size, max_position),
                    - ``output_ids (torch.LongTensor)``: positive items for prediction, the size is (batch_size, max_position),
                    - ``neg_ids (torch.LongTensor)``: negative items for prediction, the size is (batch_size, max_position).

                batch_idx (int, optional): the index of input training batch

            Returns:
                loss (torch.Scalar): BCE loss for positive items and negative items both.
        """

        # model forward
        u, seq, pos, neg = batch['user'], batch['input_ids'], batch['output_ids'], batch['neg_ids']
        vec = self.forward(u, seq)  # (batch_size, max_position, embed_size)

        # select the positions of interest
        inds = self._get_training_inds(seq, pos)
        vec, pos, neg = vec[inds], pos[inds], neg[inds]
        
        # select the users of interest (useful if concat_user is True) 
        u = u.unsqueeze(1).repeat(1, seq.size(1))[inds]

        # get logits
        pos_logits = self.predict(u, vec, pos)
        neg_logits = self.predict(u, vec, neg)

        # calculate loss
        pos_labels, neg_labels = torch.ones_like(pos_logits, device=u.device), torch.zeros_like(neg_logits, device=u.device)
        loss = self.criterion(pos_logits, pos_labels) + self.criterion(neg_logits, neg_labels)

        # return
        return loss

    def _get_training_inds(self, input_ids, output_ids):
        """ Select the positions in which the losses are calcualated. 
            For AE (auto-encoding) tasks, it is useful since we need to identify the masked tokens;
            For AR (auto-regressive) tasks, we select all input_ids positions which is not padding tokens.

            Args:
                input_ids (torch.LongTensor): input user sequence
                output_ids (torch.LongTensor): output user sequence

            Returns:
                Ellipsis or torch.LongTensor: indicies used in `_training_step_*` to select related positions of vectors, ids.
        """
        if self.args.task_type == 'ae':
            return input_ids != output_ids
        if self.args.task_type == 'ar':
            return input_ids != self.args.pad_token

    def _get_eval_inds(self, input_ids):
        """ Select last valid token positions in which we return the ranking item id list by detecting the args.pad_token

            Args:
                input_ids (torch.LongTensor): input user sequence        

            Returns:
                A tuple of two lists: the first list is the batch-wise indices, the second list is the user sequence indices

        """
        return [range(input_ids.size(0)), (input_ids != self.args.pad_token).sum(dim=-1) - 1]
