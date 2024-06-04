import os 
import json
import argparse
import random
import multiprocessing
from datetime import datetime as dt

import torch 
from torch import nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import pickle

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from copy import copy

from utils import last_commit_msg, save_dependencies, str2bool, json_load, custom_ce_loss, load_tf_ckpt, build_vocab, generate_negs, load_pickle

from model import BERTModel
from custom_optimizer import AdamW
import time
import pdb

from dataloader_bert_ae_ar import DataBERT4REC, DatasetBERT4REC
from dataloader_sas_ae_ar import DataSASRec, DatasetSASRec

MIN_VAL = 1e-6

# ------------------------
# Model 
# ------------------------

class Model(pl.LightningModule):

    # --------------------
    # Model Definition
    # --------------------

    def __init__(self, args):
        super().__init__()
        # args
        self.save_hyperparameters()
        self.args = args
        self.model = BERTModel(args)
        self.criterion = custom_ce_loss if args.use_custom_ce else torch.nn.CrossEntropyLoss()
        # init
        if self.args.weight_init == "xavier":
            self._init_weights()
        elif self.args.weight_init == "truncated_normal":
            self._init_weights_truncated_norm(stddev=0.02, threshold=2)
        elif self.args.weight_init == "truncated_normal_resample":
            self._init_weights_truncated_norm_with_resample(stddev=0.02, threshold=2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = parent_parser.add_argument_group("Model")
        p.add_argument('--hidden_size', type=int, default=50, help='embed size for items')
        p.add_argument('--num_hidden_layers', type=int, default=2, help='number of trm blocks')
        p.add_argument('--num_attention_heads', type=int, default=1, help='number of attention heads')
        p.add_argument('--dropout_prob', type=float, default=0.5, help='model dropout ratio')
        p.add_argument('--attention_dropout_prob', type=float, default=0.5, help='model dropout ratio')
        return parent_parser

    def _init_weights_truncated_norm(self, stddev, threshold):
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.normal_(param.data, mean=0.0, std=stddev)
                # Clip values outside of 2 standard deviations from the mean
                param.data.clamp_(-threshold*stddev, threshold*stddev)
                # torch.nn.init.constant_(param.data, 0.2)
            except:
                pass
        print("===NOTE: successfully initialized with truncated normal distribution===")


    def _init_weights_truncated_norm_with_resample(self, stddev, threshold):
        def _truncated_normal_init(shape, mean=0.0, stddev=0.02, threshold=2):
            # Create a tensor with random values drawn from a normal distribution
            tensor = torch.randn(shape) * stddev + mean
            # Create a mask to identify values outside of the truncated range
            mask = torch.logical_or(tensor < mean - threshold * stddev, tensor > mean + threshold * stddev)
            # Re-sample values outside of the truncated range until they are within the range
            while torch.sum(mask) > 0:
                tensor[mask] = torch.randn(torch.sum(mask)) * stddev + mean
                mask = torch.logical_or(tensor < mean - threshold * stddev, tensor > mean + threshold * stddev)
            return tensor

        for name, param in self.model.named_parameters():
            try:
                if not (any(n in name for n in ['norm', 'bias'])):
                    param.data = _truncated_normal_init(param.shape, mean=0, stddev=stddev, threshold=threshold)
                if 'bias' in name:
                    torch.nn.init.zeros_(param.data)
            except:
                pass
        print("===NOTE: successfully initialized with truncated normal distribution (with resample)===")


    def _init_weights(self):
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
                # torch.nn.init.constant_(param.data, 0.2)
            except:
                pass
        print("===NOTE: successfully initialized with xavier distribution===")

    # -------------------
    # Training Definition
    # -------------------
    def configure_optimizers(self):
        # lr linear warm-up and linear decay
        def lr_lambda(current_step):
            # Implements linear decay of the learning rate.
            # warm up
            if current_step < self.args.num_warmup_steps:
                # make sure warm up steps < total steps
                return float(current_step) / float(self.args.num_warmup_steps)
            # decay
            else:
                return float(self.args.max_steps - current_step) / float(self.args.max_steps)

        # filter out non-decay layers
        self.exclude_from_weight_decay = ["bias", "norm"]
        no_decay = []
        decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self.exclude_from_weight_decay and any(exclude_name in name for exclude_name in self.exclude_from_weight_decay):
                no_decay.append(param)
            else:
                decay.append(param)

        # Define optimizer with weight decay
        optimizer_grouped_parameters = [
            {'params': decay, 'weight_decay': self.args.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

        # optimizer
        if self.args.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer_type == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)
            print('===NOTE: Using custom AdamW optimizer===')
        if self.args.scheduler == 'default':
            return optimizer
        elif self.args.scheduler == 'linear_decay':
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }

    def training_step(self, batch, batch_idx):
        if self.args.task_type == 'ae':
            # feats
            input_ids, output_ids = batch['input_ids'], batch['output_ids']
            user_emb = self.model(input_ids)
            # pick masked positions
            inds = (input_ids != output_ids)  # different ids indicate prediction positions
            user_emb = user_emb[inds]  # pick up the user embs of interest
            logits = self.model.similarity_score(user_emb.unsqueeze(dim=1)).squeeze(dim=1)
            # calculate loss
            loss = self.criterion(logits, output_ids[inds])
            # log training loss
            self.log('train/loss', loss.item(), on_epoch=True, on_step=True)

        elif self.args.task_type == 'ar':
            seq, pos = batch['input_ids'], batch['output_ids']
            user_emb = self.model(seq) #(bs, sl, d_)
            user_emb = user_emb[seq != self.args.pad_token]
            labels = pos[seq != self.args.pad_token]
            logits = self.model.similarity_score(user_emb.unsqueeze(1)).squeeze(1)
            loss = self.criterion(logits, labels)
            # log training loss
            self.log('train/loss', loss.item(), on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch, 'test')

    def validation_epoch_end(self, outs):
        outs_all = torch.cat(outs, dim=0)
        recall_1 = outs_all[:,0].mean()
        recall_5 = outs_all[:,1].mean()
        recall_10 = outs_all[:,2].mean()
        recall_20 = outs_all[:,3].mean()
        recall_50 = outs_all[:,4].mean()
        recall_100 = outs_all[:,5].mean()

        ndcg_5 = outs_all[:,6].mean()
        ndcg_10 = outs_all[:,7].mean()
        ndcg_20 = outs_all[:,8].mean()
        ndcg_50 = outs_all[:,9].mean()
        ndcg_100 = outs_all[:,10].mean()

        self.log('valid/R@1', recall_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/R@5', recall_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/R@10', recall_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/R@20', recall_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/R@50', recall_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/R@100', recall_100, on_step=False, on_epoch=True, prog_bar=True)

        self.log('valid/N@5', ndcg_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/N@10', ndcg_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/N@20', ndcg_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/N@50', ndcg_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid/N@100', ndcg_100, on_step=False, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outs):
        outs_all = torch.cat(outs, dim=0)
        recall_1 = outs_all[:,0].mean()
        recall_5 = outs_all[:,1].mean()
        recall_10 = outs_all[:,2].mean()
        recall_20 = outs_all[:,3].mean()
        recall_50 = outs_all[:,4].mean()
        recall_100 = outs_all[:,5].mean()

        ndcg_5 = outs_all[:,6].mean()
        ndcg_10 = outs_all[:,7].mean()
        ndcg_20 = outs_all[:,8].mean()
        ndcg_50 = outs_all[:,9].mean()
        ndcg_100 = outs_all[:,10].mean()

        self.log('test/R@1', recall_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/R@5', recall_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/R@10', recall_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/R@20', recall_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/R@50', recall_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/R@100', recall_100, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/N@5', ndcg_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/N@10', ndcg_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/N@20', ndcg_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/N@50', ndcg_50, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/N@100', ndcg_100, on_step=False, on_epoch=True, prog_bar=True)
    
    # internal util functions
    def _evaluation_step(self, batch, mode='valid', task_type = None):
        if self.args.task_type == 'ae':
            # feats
            input_ids, output_ids, negs = batch['input_ids'], batch['output_ids'], batch['negs']
            user_emb = self.model(input_ids)
            # pick masked positions
            inds = (input_ids != output_ids)  # different ids indicate prediction positions
            if args.data_version == 'bert4rec':
                if (args.load_negs_tf is not None) and ((not self.args.use_valid) or (mode=='test')):
                    assert (output_ids[inds]==negs[:,0]).sum()/negs[:,0].shape[0] == 1, 'eval positive samples does not match'
                    negs = negs[:, 1:]
            user_emb = user_emb[inds]  # pick up the user embs of interest
            # neg logits
            if args.all_ranking:
                # derive all logits
                neg_logits = self.model.similarity_score(user_emb.unsqueeze(dim=1)).squeeze(1) # (bs, n_item+2)
                # excluding ranked items # also excluding the clonze mask
                neg_logits = neg_logits.masked_fill(F.one_hot(input_ids, neg_logits.shape[1]).sum(1).bool(), float(-2**20))
                # excluding self besides above
                neg_logits = neg_logits.masked_fill(F.one_hot(output_ids, neg_logits.shape[1]).sum(1).bool(), float(-2**20))
                # excluding padding/look_ahead mask 0
                neg_logits[:,0] = float(-2**20)
            else:
                neg_logits = self.model.similarity_score(user_emb.unsqueeze(dim=1), negs)
            # pos logits
            pos_logits = self.model.similarity_score(user_emb.unsqueeze(dim=1), output_ids[inds].unsqueeze(dim=1))
            
        elif self.args.task_type == 'ar':
            seq, pos, negs = batch['input_ids'], batch['output_ids'], batch['negs']
            pos_ = pos.clone()
            if args.data_version == 'bert4rec':
                if (args.load_negs_tf is not None) and ((not self.args.use_valid) or (mode=='test')):
                    assert (pos.squeeze(-1)==negs[:,0]).sum()/negs[:,0].shape[0] == 1, 'eval positive samples does not match'
                    negs = negs[:, 1:]
            if args.padding_mode == 'tail':
                user_emb = self.model(seq)[range(seq.size(0)), ((seq != self.args.pad_token).sum(dim=-1)-1)] # (bs, sl, d_) -> (bs, d_)
                pos = pos[range(seq.size(0)), ((seq != self.args.pad_token).sum(dim=-1)-1)].unsqueeze(-1) #(bs,1)
            else:
                user_emb = self.model(seq)[:, -1, :] # (bs, sl, d_) -> (bs, d_)
                pos = pos[:,-1].unsqueeze(-1) #(bs,1)
            # neg logits
            if args.all_ranking:
                # derive all logits
                neg_logits = self.model.similarity_score(user_emb.unsqueeze(dim=1)).squeeze(1) # (bs, n_item+2)
                # excluding ranked items
                neg_logits = neg_logits.masked_fill(F.one_hot(seq, neg_logits.shape[1]).sum(1).bool(), float(-2**20))
                # excluding self besides seq
                neg_logits = neg_logits.masked_fill(F.one_hot(pos_, neg_logits.shape[1]).sum(1).bool(), float(-2**20))
                # excluding padding/look_ahead mask 0
                neg_logits[:,0] = float(-2**20)
            else:
                neg_logits = self.model.similarity_score(user_emb.unsqueeze(dim=1), negs)
            # pos logits
            pos_logits = self.model.similarity_score(user_emb.unsqueeze(dim=1), pos)

        # metrics
        rank = (pos_logits < neg_logits).sum(dim=-1).flatten()
        recall_1 = (rank < 1).float()
        recall_5 = (rank < 5).float()
        recall_10 =  (rank < 10).float()
        recall_20 =  (rank < 20).float()
        recall_50 =  (rank < 50).float()
        recall_100 =  (rank < 100).float()
        ndcg_5 = (1 / torch.log2(rank + 2)) * (rank < 5).float()
        ndcg_10 = (1 / torch.log2(rank + 2)) * (rank < 10).float()
        ndcg_20 = (1 / torch.log2(rank + 2)) * (rank < 20).float()
        ndcg_50 = (1 / torch.log2(rank + 2)) * (rank < 50).float()
        ndcg_100 = (1 / torch.log2(rank + 2)) * (rank < 100).float()
        metrics = torch.concat((recall_1.unsqueeze(0), recall_5.unsqueeze(0), recall_10.unsqueeze(0), \
                                recall_20.unsqueeze(0), recall_50.unsqueeze(0), recall_100.unsqueeze(0), \
                                ndcg_5.unsqueeze(0), ndcg_10.unsqueeze(0), ndcg_20.unsqueeze(0), \
                                ndcg_50.unsqueeze(0), ndcg_100.unsqueeze(0)), \
                                dim=0)
        metrics = metrics.transpose(0,1)
        return metrics
    


# ------------------------
# Argument
# ------------------------

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    # check python main.py --help for details
    # task control
    p.add_argument('--task_type', type=str, default='ae')
    p.add_argument('--causal_mask', type=str2bool, default=False)
    p.add_argument('--use_valid', type=str2bool, default=True)
    p.add_argument('--all_ranking', type=str2bool, default=False, required=True)
    # overall
    p.add_argument('--seed', type=int, default=42, help='random seed for model training')
    # dataset
    p.add_argument('--data_version', type=str, required=True, help='whether to use sasrec/bert4rec-like raw data')
    p.add_argument('--negs_type', type=str, required=True, help='whether to use pop or uniform random sample negs for eval')
    p.add_argument('--dataset', type=str, required=True, help='dataset name')    
    p.add_argument('--pad_token', type=int, default=0, help='pad token')
    p.add_argument('--padding_mode', type=str, default='tail', help='padding mode')
    p.add_argument('--mask_prob', type=float, default=0.5, required=True)
    p.add_argument('--max_position', type=int, default=200, help='max length of user history')
    # trainer
    p.add_argument('--batch_size', type=int, default=128, help='number of epochs for train')
    p.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    p.add_argument('--optimizer_type', type=str, default='adamw')
    p.add_argument('--scheduler', type=str, default='linear_decay')
    p.add_argument('--weight_decay', type=float, default=0, help='weight decay for model params except layer norms or bias')
    p.add_argument('--weight_init', type=str, default='xavier')
    p.add_argument('--num_warmup_steps', type=int, default=100)
    # test
    p.add_argument('--evaluate_samples', type=str2bool, default=False)
    p.add_argument('--eval_neg', type=int, default=100, help='sample {eval_neg} items for evaluation')
    p.add_argument('--test_ckpt', type=str, default=None, help='model checkpoint for testing only')
    p.add_argument('--use_custom_ce', type=str2bool, default=True)
    # ckpt
    p.add_argument('--load_negs_tf', type=str, default=None)
    p.add_argument('--early_stopping', type=str2bool, default=False)
    p.add_argument('--patience', type=int, default=10)
    # model 
    p = Model.add_model_specific_args(p)
    # trainer 
    p = Trainer.add_argparse_args(p)
    
    return p.parse_args()

# ------------------------
# Main Function
# ------------------------

if __name__ == "__main__":
    # args
    args = parse_arguments()
    seed_everything(args.seed)

    # logging folder
    print(f'=== Using all_ranking: {args.all_ranking} ===')
    cm = 1 if args.causal_mask else 0
    args.default_root_dir = os.path.join(f'./output_datatype_{args.data_version}', f'allrank_{args.all_ranking}', f'{args.task_type}_cm{cm}_negs_{args.negs_type}_loss_ce', \
                                         f'{args.dataset}_seed{args.seed}_head{args.num_attention_heads}_dp{args.dropout_prob}_wd{args.weight_decay}')

    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    # dataset type
    dv = args.data_version.split('_')[0]
    if dv == 'bert4rec':
        Data = DataBERT4REC
        Dataset = DatasetBERT4REC
        print('===NOTE:Using BERT4REC dataloader===')
    else:
        Data = DataSASRec
        Dataset = DatasetSASRec
        print('===NOTE:Using SASRec dataloader===')

    # dataset
    # assert args.use_valid
    data_path = f"../data_{args.data_version}/{args.dataset}"
    assert os.path.isdir(data_path)
    print(f'=== using data_{args.data_version} ===')
    print(f"set ckpt as {args.default_root_dir}, data path as {data_path}")
    
    dataset = Data(data_path, args)
    args.num_users = dataset.num_users
    args.num_items = dataset.num_items
    args.mask_token = args.num_items

    # data loader
    if multiprocessing.cpu_count() > 16:
        num_workers = 8
    else:
        num_workers = 8
    train_loader = DataLoader(Dataset(data=dataset, mode="train", args=args), batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    if args.use_valid:
        val_loader = DataLoader(Dataset(data=dataset, mode="valid", args=args), batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Dataset(data=dataset, mode="test", args=args), batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # model and training
    # adding loggers
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.default_root_dir, name="tb_logs")
    csv_logger = pl.loggers.CSVLogger(save_dir=args.default_root_dir, name="csv_logs")
    callbacks = [
            ModelCheckpoint( 
            monitor = 'valid/R@10', 
            mode="max", 
            save_last=True, 
            save_top_k=1, 
            every_n_epochs=1),
            ]
    if args.early_stopping:
        early_stop_callback = EarlyStopping(monitor="valid/R@10", patience=args.patience, verbose=False, mode="max")
        callbacks.append(early_stop_callback)
    
    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        log_every_n_steps = 100,
        gradient_clip_val=5.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        default_root_dir=args.default_root_dir,
        deterministic = True, 
        logger=[tb_logger, csv_logger],
        callbacks=callbacks
        )
    
    # actual fitting code
    if args.test_ckpt=='None': args.test_ckpt=None
    if args.test_ckpt is None:
        model = Model(args)
        if args.use_valid:
            print('===NOTE: Entering train, valid and test branch===')
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            print('===NOTE: Entering train and test (no valid set) branch===')
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            print("=== NOTE: Entering reproducing curve mode, test set -> validation")
        trainer.test(model=model, dataloaders=test_loader, ckpt_path='best')
    elif args.test_ckpt is not None:
        model = Model(args)
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.test_ckpt)
    else:
        raise Exception('===NOTE: incorrect ckpt configuration===')

    