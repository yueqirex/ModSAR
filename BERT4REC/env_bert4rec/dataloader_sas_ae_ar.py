import random

import os 
import json
import torch 
import numpy as np
import pdb
from utils import json_load
from copy import deepcopy

# ------------------------
# Dataset 
# ------------------------
class DataSASRec(object):
    def __init__(self, path, args):
        """
            Data:
                Dictionary of list
                train:{1:[1,2,3...], 2:[]}
                val:{1:[4],2:[5]}
                test:{1[5],2:[7]}
        """
        # meta
        self.meta = json_load(path, 'meta.json')
        self.num_users = len(self.meta['user2id'])+1
        self.num_items = len(self.meta['item2id'])+1
        # train
        self.train = json_load(path, 'train.json', key_type=int)
        # valid
        self.valid_all = json_load(path, 'valid.json', key_type=int)
        self.test_all = json_load(path, 'test.json', key_type=int)
        # item_freq
        self.item_freq = json_load(path, 'item_freq.json', key_type=int)
        if (args.evaluate_samples) and (len(self.valid_all) >= args.sample_len):
            # this means evaluate 10000 samples
            self.valid = json_load(path, 'valid_samples.json', key_type=int)
            self.test = json_load(path, 'test_samples.json', key_type=int)
            # filter train[user] length <3, prevent data leakage for this type of dataloading
            self.valid_ = {k: (self.train[k] + v if len(self.train[k]+self.valid_all[k]+self.test_all[k])>=3 else []) for k, v in self.valid.items()}
            self.test_ = {k: (self.train[k] + self.valid_all[k] + v if len(self.train[k]+self.valid_all[k]+self.test_all[k])>=3 else []) for k, v in self.test.items()}
            self.valid = self.valid_
            self.test = self.test_
        else:
            self.valid = json_load(path, 'valid.json', key_type=int)
            self.test = json_load(path, 'test.json', key_type=int)
            # filter train[user] length <3, prevent data leakage for this type of dataloading
            self.valid_ = {k: (self.train[k] + v if len(self.train[k]+self.valid[k]+self.test[k])>=3 else []) for k, v in self.valid.items()}
            self.test_ = {k: (self.train[k] + self.valid[k] + v if len(self.train[k]+self.valid[k]+self.test[k])>=3 else []) for k, v in self.test.items()}
            self.valid = self.valid_
            self.test = self.test_
        # remove user that has [] for valid or test from corresponding sets.
        for k, v in deepcopy(self.valid).items():
            if v == []: del self.valid[k]
        for k, v in deepcopy(self.test).items():
            if v == []: del self.test[k]
        # splits
        self.splits = {'train': self.train, 'valid': self.valid, 'test': self.test}



class DatasetSASRec(torch.utils.data.Dataset):
    """ This is `Dataset` Class to read data splits (training / valid / test) and used for `Dataloader`.
    """
    def __init__(self, data, mode, args):
        """ Load data split. 
            Args:
                data (`Data`): Instantized Data class.
                mode (str): Split of data, which can be "train", "valid" or "test".
                args (`argparse Namespace`): args parsed from three level argument groups (check shared trainer `arguments <../usage/main_detail.html>`_).
            
        """
        super().__init__()

        if args.task_type == 'ar':
            self.dataset = ARDatasetSASRec(data, mode, args)
            print(f'=== NOTE: {mode} Using ar dataset===')
        if args.task_type == 'ae':
            self.dataset = AEDatasetSASRec(data, mode, args)
            print(f'=== NOTE: {mode} Using ae dataset===')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """ Call `ARDataset.__getitem__(index)` if args.task_type == 'ar',
            or call `AEDataset.__getitem__(index)` if args.task_type == 'ae'.
        """
        return self.dataset.__getitem__(index)



class ARDatasetSASRec(torch.utils.data.Dataset):
    """ This is `Auto-Regressive Dataset` Class to read data splits (training / valid / test) and used for `Dataloader`.
    """
    def __init__(self, data, mode, args):
        """ Load data split. 
            Args:
                data (`Data`): Instantized Data class.
                mode (str): Split of data, which can be "train", "valid" or "test".
                args (`argparse Namespace`): args parsed from three level argument groups (check shared trainer `arguments <../usage/main_detail.html>`_).
            
        """
        self.args = args
        self.data = data.splits[mode]
        # for popularity-based sample
        self.item_freq = data.item_freq
        self.items = np.array(list(self.item_freq.keys()))
        self.prob = np.array(list(self.item_freq.values()))/sum(self.item_freq.values())

        self.keys = list(self.data.keys())
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """ Return a data sample for Dataloader. 
            Args:
                index (int): index to retrive a data sample from the data split.
            Returns:
                dict: dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:
                    - ``user (int)``: user id,
                    - ``input_ids (numpy.array, torch.LongTensor)``: user sequence, the size is padded to (args.max_position, ),
                    - ``output_ids (numpy.array, torch.LongTensor)``: positive items for prediction, the size is padded to (args.max_position, ),
                    - ``neg_ids (numpy.array, torch.LongTensor)``: negative items for prediction, the size is padded to (args.max_position, ) in `training` mode; the size is padded to (args.max_position, args.eval_neg) in validation or testing mode, where `args.eval_neg` is the number of negative items for evaluation.
        """

        # user id 
        userid = self.keys[index]
        # seq
        seq = self.data[userid]
        # pos seq
        pos = seq[1:]
        seq = seq[:-1]
        pos_set = set(pos)
        # neg seq
        if self.mode == 'train':
            if self.args.negs_type == 'uniform':
                neg = self._padding([self._random_neg_uniform(pos_set) for _ in pos], self.args.padding_mode)
            else:
                neg = self._padding(self._random_neg_pop(pos_set, len(pos)), self.args.padding_mode)
        else:
            if self.args.negs_type == 'uniform':
                neg = np.array([self._random_neg_uniform(pos_set) for _ in range(self.args.eval_neg)])
            else:
                neg = np.array(self._random_neg_pop(pos_set, self.args.eval_neg))
        return {
            'user': userid,
            'input_ids': self._padding(seq, self.args.padding_mode),
            'output_ids': self._padding(pos, self.args.padding_mode),
            'negs': neg
        }

    # internal util functions

    def _random_neg_uniform(self, s):
        t = random.randint(1, self.args.num_items-1) # end inclusive
        while t in s:
            t = random.randint(1, self.args.num_items-1)
        return t 
    

    def _random_neg_pop(self, s, sample_num):
        negs = np.random.choice(self.items, sample_num, p=self.prob)
        for i in range(negs.shape[0]):
            while negs[i] in s:
                negs[i] = np.random.choice(self.items, 1, p=self.prob)[0]
        return negs.tolist()
    
    def _padding(self, s, mode):
        if mode == 'head':
            return np.array([self.args.pad_token] * (self.args.max_position - len(s)) + s[-self.args.max_position:])
        elif mode == 'tail':    
            return np.array(s[-self.args.max_position:] + [self.args.pad_token] * (self.args.max_position - len(s)))
        else:
            raise Exception('illegal padding mode')



class AEDatasetSASRec(torch.utils.data.Dataset):
    """ This is `Auto-Encoding Dataset` Class to read data splits (training / valid / test) and used for `Dataloader`.
    """
    def __init__(self, data, mode, args):
        """ Load data split. 
            Args:
                data (`Data`): Instantized Data class.
                mode (str): Split of data, which can be "train", "valid" or "test".
                args (`argparse Namespace`): args parsed from three level argument groups (check shared trainer `arguments <../usage/main_detail.html>`_).
            
        """
        self.args = args
        self.data = data.splits[mode]
        # for popularity-based sample
        self.item_freq = data.item_freq
        self.items = np.array(list(self.item_freq.keys()))
        self.prob = np.array(list(self.item_freq.values()))/sum(self.item_freq.values())
        self.keys = list(self.data.keys())
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """ Return a data sample for Dataloader. 
            Args:
                index (int): index to retrive a data sample from the data split.
            Returns:
                dict: dictionary of torch tensors for batch-wise inputs, the keys and values in this dict are:
                    - ``user (torch.LongTensor)``: user ids, the size is (batch_size, ),
                    - ``input_ids (torch.LongTensor)``: user sequence, the size is (batch_size, max_position),
                    - ``output_ids (torch.LongTensor)``: positive items for prediction, the size is (batch_size, max_position),
                    - ``neg_ids (torch.LongTensor)``: negative items for prediction, the size is (batch_size, max_position).
        """
        # user id & seq
        userid = self.keys[index]
        seq = self.data[userid]
        if self.mode == 'train':
            return self._train_sample(userid, seq)
        else:
            return self._non_train_sample(userid, seq)

    def _train_sample(self, userid, seq):
        # masking
        tokens = seq[-self.args.max_position:]
        labels = tokens.copy()
        labels_set = set(labels)
        index = np.arange(len(tokens))
        np.random.shuffle(index)

        max_predict = min(int(self.args.max_position * self.args.mask_prob), max(1, int(round(len(index) * self.args.mask_prob))))
        for i in range(max_predict):
            tokens[index[i]] = self.args.mask_token

        # negative sampling
        if self.args.negs_type == 'uniform':
            neg = [self._random_neg_uniform(labels_set) for _ in labels]
        else:
            neg = self._random_neg_pop(labels_set, len(labels))

        # padding
        return {
            'user': userid,
            'input_ids': self._padding(tokens, self.args.padding_mode), 
            'output_ids': self._padding(labels, self.args.padding_mode),
            'negs': self._padding(neg, self.args.padding_mode)
        }

    def _non_train_sample(self, userid, seq):
        # masking
        tokens = seq[-self.args.max_position:]
        labels = tokens.copy()
        pos_set = set(labels)
        tokens[-1] = self.args.mask_token

        # negative sampling
        if self.args.negs_type == 'uniform':
            neg = [self._random_neg_uniform(pos_set) for _ in range(self.args.eval_neg)]
        else:
            neg = self._random_neg_pop(pos_set, self.args.eval_neg)

        # padding
        return {
            'user': userid,
            'input_ids': self._padding(tokens, self.args.padding_mode), 
            'output_ids': self._padding(labels, self.args.padding_mode),
            'negs': np.array(neg)
        }

    # internal util functions

    def _random_neg_uniform(self, s):
        t = random.randint(1, self.args.num_items-1)
        while t in s:
            t = random.randint(1, self.args.num_items-1)
        return np.array(t)
    

    def _random_neg_pop(self, s, sample_num):
        negs = np.random.choice(self.items, sample_num, p=self.prob)
        for i in range(negs.shape[0]):
            while negs[i] in s:
                negs[i] = np.random.choice(self.items, 1, p=self.prob)[0]
        return negs.tolist()
    
    def _padding(self, s, mode):
        if mode == 'head':
            return np.array([self.args.pad_token] * (self.args.max_position - len(s)) + s[-self.args.max_position:])
        elif mode == 'tail':    
            return np.array(s[-self.args.max_position:] + [self.args.pad_token] * (self.args.max_position - len(s)))
        else:
            raise Exception('illegal padding mode')