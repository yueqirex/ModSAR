import json
import os
import torch
import pickle
import numpy as np
from copy import copy
from utils import json_load
# ------------------------
# Dataset 
# ------------------------

class DataBERT4REC(object):
    def __init__(self, path, args):
        """
            Data:
                List of dictionary
                train: [{input_ids:..., mask_is:..., output_ids:...}, {inputs_ids:...}]
                    masking a % of input ids
                test: [{input_ids:..., mask_is:..., output_ids:...}, {inputs_ids:...}]
                    masking only the last input id
            Returns:
                A dictionary of train test, and negs data
        """
        # meta
        self.meta = json_load(path, 'meta.json')
        self.num_users = len(self.meta['user2id'])
        self.num_items = len(self.meta['item2id'])
        # train
        with open(os.path.join(path, 'train.json')) as f:
            self.train = json.load(f)
        # valid
        if args.use_valid: 
            with open(os.path.join(path, 'valid.json')) as f:
                self.valid = json.load(f)
            with open(os.path.join(path, f'valid_negs_{args.negs_type}.json')) as f:
                self.val_negs = json.load(f)
            print(f'=== Using valid_negs_{args.negs_type}.json ===')
        with open(os.path.join(path, 'test.json')) as f:
            self.test = json.load(f)

        # test negs
        if args.load_negs_tf is not None:
            # tf converted negs, containing the positive -> 101
            with open(args.load_negs_tf, 'rb') as f:
                self.test_negs = pickle.load(f, encoding='latin1')
            print(f'===NOTE: sucessfully loaded negs from tf converted: {args.load_negs_tf}===')
        else:
            # torch random sampled negs, and does not contain the positive -> 100
            with open(os.path.join(path, f'test_negs_{args.negs_type}.json')) as f:
                self.test_negs = json.load(f)
            print(f'=== Using test_negs_{args.negs_type}.json ===')
            
        # splits
        if args.use_valid:
            self.splits = {'train': self.train, 'valid':self.valid, 'test': self.test}# 'vocab': self.vocab}
        else:
            self.splits = {'train': self.train, 'test': self.test}# 'vocab': self.vocab}

class DatasetBERT4REC(torch.utils.data.Dataset):
    def __init__(self, data, mode, args):
        self.args = args
        # self.vocab = data.splits['vocab']
        self.data = data.splits[mode]
        self.test_negs = data.test_negs
        if self.args.use_valid:
            self.val_negs = data.val_negs
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """ Get input ids, mask id and output ids (ground-truth ids) 

        Args:
            index: index of self.data, which is used to retreive data sample self.data[i], for which the keys are:
                input_ids: int numpy array, token sequence of user behavior
                mask_id: the id of mask token
                output_ids: ground-truth user sequence

        Returns:
            res: results of input ids, mask id and output ids, if there are in test mode, we add negs, the specific keys are:
                input_ids: int numpy array, token sequence of user behavior
                mask_id: the id of mask token
                output_ids: ground-truth user sequence
                negs: optional, 100 negative items
        """
        if self.mode == 'train':
            if self.args.task_type == 'ar':
                res = {}
                output_ids = np.array(copy(self.data[index]['output_ids']))
                res['input_ids'] = self._padding(output_ids[output_ids!=self.args.pad_token][:-1].tolist(), mode=self.args.padding_mode)
                res['output_ids'] = self._padding(output_ids[output_ids!=self.args.pad_token][1:].tolist(), mode=self.args.padding_mode)

            elif self.args.task_type == 'ae':
                res = self.data[index]
        
        elif self.mode == 'valid':
            if self.args.task_type == 'ar':
                res = {}
                output_ids = np.array(copy(self.data[index]['output_ids']))
                res['input_ids'] = self._padding(output_ids[output_ids!=self.args.pad_token][:-1].tolist(), mode=self.args.padding_mode)
                res['output_ids'] = self._padding(output_ids[output_ids!=self.args.pad_token][1:].tolist(), mode=self.args.padding_mode)

            elif self.args.task_type == 'ae':
                res = self.data[index]

            res.update({'negs': self.val_negs[index]})

        elif self.mode == 'test':
            if self.args.task_type == 'ar':
                res = {}
                output_ids = np.array(copy(self.data[index]['output_ids']))
                res['input_ids'] = self._padding(output_ids[output_ids!=self.args.pad_token][:-1].tolist(), mode=self.args.padding_mode)
                res['output_ids'] = self._padding(output_ids[output_ids!=self.args.pad_token][1:].tolist(), mode=self.args.padding_mode)
                
            elif self.args.task_type == 'ae':
                res = self.data[index]

            res.update({'negs': self.test_negs[index]})
        
        return {k: np.array(res[k]) for k in res}
        
    
    def _padding(self, s, mode):
        if mode == 'head':
            return np.array([self.args.pad_token] * (self.args.max_position - len(s)) + s[-self.args.max_position:])
        elif mode == 'tail':    
            return np.array(s[-self.args.max_position:] + [self.args.pad_token] * (self.args.max_position - len(s)))
        else:
            raise Exception('illegal padding mode')