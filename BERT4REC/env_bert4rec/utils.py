import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
import pickle
from tqdm import tqdm

def custom_ce_loss(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    one_hot_labels = F.one_hot(labels, num_classes=logits.shape[1])
    loss = -torch.mean(torch.sum(log_probs * one_hot_labels, dim=-1))
    return loss

def load_tf_ckpt(model, ckpt_path):
    with open(ckpt_path, 'rb') as f:
        state_dict = pickle.load(f, encoding='latin1')
    torch_state_dict = {k:torch.from_numpy(v) for k,v in state_dict.items()}
    model.load_state_dict(torch_state_dict)
    return model

def last_commit_msg():
    try:
        import re
        from subprocess import check_output
        hashed_id = check_output('git log -1 --pretty=format:"%H"'.split()).decode('utf-8').rstrip('\n').replace('\n', '').replace('\"', '')[:8]
        msg_short = '_'.join(re.sub("\s\s+", " ", check_output('git log -1 --oneline --pretty=format:"%s"'.split()).decode('utf-8').strip('\n').replace('\n', '').replace('\"', '')).split(' '))
        current_branch = check_output('git rev-parse --abbrev-ref HEAD'.split()).decode('utf-8').rstrip('\n').replace('\n', '').replace('\"', '')
        return current_branch, f"{msg_short}_{hashed_id}"
    except:
        return "", "no_commit"

def load_pickle(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res

def build_vocab(datasets, tar_dir, args):
    all_items = []
    for dataset in datasets:
        for d in dataset:
            output_ids = np.array(d['output_ids'])
            all_items.extend(output_ids[output_ids!=args.pad_token].tolist())
    vocab = {}
    for d in tqdm(all_items):
        if d in vocab:
            vocab[d] += 1
        else:
            vocab[d] = 0
    with open(tar_dir, 'wb') as f:
        pickle.dump(vocab, f)
    print('=== vocab saved to disk successfully ===')
    return vocab

def generate_negs(vocab, dataset):
    """ Generate popularity-based negs according to https://github.com/FeiSun/BERT4Rec/blob/master/run.py#L198

    Sample 100 negs according to the global popularity of items, interacted items would be rejected in sampling

    Traget negative sample fiel format:
        {tgt_dir}/negs.json: dict of {num_of_test_sample: list of negative popular items}

    Args:
        tgt_dir: target directory to save json files 
        test: list of testing samples generated from `convert_test`
    """

    # calculate popularity

    keys = list(vocab.keys())
    values = list(vocab.values())
    sum_value = np.sum([x for x in values])
    probability = [value / sum_value for value in values]

    # generate negs
    negs = [] # (bs, 100)

    for idx in range(len(dataset)):
        rated = set(dataset[idx]['output_ids']) if 'output_ids' in dataset[idx].keys() else set(np.concatenate((dataset[idx]['seq'],dataset[idx]['pos'])))
        rated.add(0)
        item_idx = []
        while len(item_idx) < 101:
            sampled_ids = np.random.choice(keys, 101, replace=False, p=probability)
            sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        item_idx = item_idx[:100]

        assert len(item_idx) == 100
        negs.append([int(i) for i in item_idx])
    return negs

def save_dependencies(ckpt):
    try:
        import os
        os.system(f"pip freeze > {ckpt}/requirements.txt")
    except:
        pass


def str2bool(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def json_load(path, x, key_type=None):
    res = json.load(open(os.path.join(path, x), 'r'))
    if key_type is not None:
        res = {key_type(k): v for k, v in res.items()}
    return res