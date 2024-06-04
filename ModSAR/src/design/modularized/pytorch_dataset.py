import random

import os 
import json
import torch 
import numpy as np
import pdb


# ------------------------
# Dataset 
# ------------------------

class Data(object):
    """ This is `Data` Class to read dataset, and record basic info (e.g., number of items, number of users, data splits).
    """
    def __init__(self, args):
        """ Load dataset. 

            Args:
                path (str): the dictionary of recommendation dataset. We require four ``*.json`` files in this directionary:

                    - `meta.json`: ``user2id`` dictionary and ``item2id`` dictionary in this metadata json file.
                    - `train.json`: dictionary from ``user_id`` to user sequence (i.e., a list of training items).
                    - `valid.json`: dictionary from ``user_id`` to validation items (i.e., a list of validation items, it is usually single one).
                    - `test.json`: dictionary from ``user_id`` to testing items (i.e., a list of testing items, it is usually single one).

                args (argparse.args): all arguments
        """
        # args
        self.args = args
        # path
        path = args.data_dir
        # meta
        self.meta = self._json_load(path, 'meta.json')
        self.num_users = len(self.meta['user2id'])
        self.num_items = len(self.meta['item2id'])
        # train
        self.train = self._json_load(path, 'train.json', key_type=int)
        # valid
        self.valid = self._json_load(path, 'valid.json', key_type=int)
        # self.valid = {k: self.train[k]+v for k, v in self.valid.items()}
        # test
        self.test = self._json_load(path, 'test.json', key_type=int)
        # self.test = {k: self.valid[k]+v for k, v in self.test.items()} 

        from copy import deepcopy
        # filter train[user] length <3, prevent data leakage for this type of dataloading
        self.valid_ = {k: (self.train[k] + v if len(self.train[k]+self.valid[k]+self.test[k])>=3 else []) for k, v in self.valid.items()}
        self.test_ = {k: (self.train[k] + self.valid[k] + v if len(self.train[k]+self.valid[k]+self.test[k])>=3 else []) for k, v in self.test.items()}
        self.valid = self.valid_
        self.test = self.test_
        # remove user that has [] for valid or test from those sets.
        for k, v in deepcopy(self.valid).items():
            if v == []: del self.valid[k]
        for k, v in deepcopy(self.test).items():
            if v == []: del self.test[k]

        # splits
        self.splits = {'train': self.train, 'valid': self.valid, 'test': self.test}
        # update args
        self.check_args(args)

    def _json_load(self, path, x, key_type=None):
        # pdb.set_trace()
        res = json.load(open(os.path.join(path, x), 'r'))
        if key_type is not None:
            res = {key_type(k): v for k, v in res.items()}
        return res
    
    def check_args(self, args):
        """ Update num_users, num_items, pad_token, mask_token to args numspace.

            Args:
                args (argparse.Namespace): argparse namespace to update num_users, num_items, pad_token, mask_token after loading dataset.
        """
        assert args.num_users == self.num_users + 1, f"args.num_users: {args.num_users} != self.num_users: {self.num_users}"
        assert args.num_items == self.num_items, f"args.num_items: {args.num_items} != self.num_items: {self.num_items}"

        print(f"load {self.args.data_dir}!")
        print(f"num_users: {args.num_users}, num_items: {args.num_items}, pad_token: {args.pad_token}, mask_token: {args.mask_token}")
        

class ARDataset(torch.utils.data.Dataset):
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
        self.keys = sorted(list(self.data.keys()))
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
            neg = [self._random_neg(pos_set) for _ in pos]
        else:
            neg = [self._random_neg(pos_set) for _ in range(self.args.eval_neg)]
        return {
            'user': userid,
            'input_ids': np.array(seq),
            'output_ids': np.array(pos),
            'neg_ids': np.array(neg)
        }

    # internal util functions

    def _random_neg(self, s):
        t = random.randint(1, self.args.num_items)
        while t in s:
            t = random.randint(1, self.args.num_items)
        return t 
    

class AEDataset(torch.utils.data.Dataset):
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
        index = np.arange(len(tokens))
        np.random.shuffle(index)

        max_predict = min(int(self.args.max_position * self.args.mask_prob), max(1, int(round(len(index) * self.args.mask_prob))))
        for i in range(max_predict):
            tokens[index[i]] = self.args.mask_token

        # negative sampling
        neg = [self._random_neg(set([i])) for i in labels]

        # padding
        return {
            'user': userid,
            'input_ids': np.array(tokens), 
            'output_ids': np.array(labels),
            'neg_ids': np.array(neg)
        }

    def _non_train_sample(self, userid, seq):
        # masking
        tokens = seq[-self.args.max_position:]
        labels = tokens.copy()
        pos_set = set(labels)
        tokens[-1] = self.args.mask_token

        # negative sampling
        neg = [self._random_neg(pos_set) for _ in range(self.args.eval_neg)]

        # padding
        return {
            'user': userid,
            'input_ids': np.array(tokens), 
            'output_ids': np.array(labels),
            'neg_ids': np.array(neg)
        }

    # internal util functions

    def _random_neg(self, s):
        t = random.randint(1, self.args.num_items)
        while t in s:
            t = random.randint(1, self.args.num_items)
        return np.array(t)
    

class Dataset(torch.utils.data.Dataset):
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

        self.args = args

        if args.task_type == 'ar':
            self.dataset = ARDataset(data, mode, args)

        if args.task_type == 'ae':
            self.dataset = AEDataset(data, mode, args)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """ Call `ARDataset.__getitem__(index)` if args.task_type == 'ar',
            or call `AEDataset.__getitem__(index)` if args.task_type == 'ae'.
        """
        sample = self.dataset.__getitem__(index)

        # item sse 
        if np.random.rand() < self.args.item_sse_prob:
            # generate random sequence indices to apply sse
            sse_prob = np.random.rand(len(sample['input_ids']))
            sse_index = (sse_prob > self.args.item_sse_prob)
            # generate random input or output sequence to replace 
            for_input = np.random.randint(1, self.args.num_items+1, size=sse_index.sum())
            for_output = np.random.randint(1, self.args.num_items+1, size=sse_index.sum())
            # replace
            sample['input_ids'][sse_index] = for_input
            sample['output_ids'][sse_index] = for_output

        # user sse
        if np.random.rand() < self.args.user_sse_prob:
            sample['user'] = np.random.randint(1, self.args.num_users)

        # padding 
        sample['input_ids'] = self._padding(sample['input_ids'])
        sample['output_ids'] = self._padding(sample['output_ids'])
        # in training mode, neg_ids are used in loss function, so we need to pad them
        # in testing mode, neg_ids are used in evaluation, so we don't need to pad them
        if self.dataset.mode == 'train':
            sample['neg_ids'] = self._padding(sample['neg_ids'])

        return sample

    def _padding(self, s):
        s = s.tolist()
        return np.array(s[-self.args.max_position:] + [self.args.pad_token] * (self.args.max_position - len(s)))
