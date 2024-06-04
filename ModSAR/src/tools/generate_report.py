""" Tool used to generate report for multiple experiments (listed by ckpt name, grouped by random seed).
"""

import os
import sys
import pandas as pd
import numpy as np

from datetime import datetime as dt
from glob import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_results_from_csv(path, metrics=None):
    """ Fetech recorded results from tensorboard.

        Args:
            path (str): csv logger file path
            metrics (list, optional): names of metrics, e.g., `test/N@10`, which should be aligned with metric names in tensorboard.
                all metrics will be returned if `metrics` is None. default=None.

        Returns: 
            dict: dictionary of (metric_name, record_values), e.g., {`valid/N@10`: [0.1, 0.5, 0.3]}.
            list: metric names, e.g., [`valid/N@10`, `valid/R@10`]
    """
    raise NotImplementedError


def get_results_from_tensorboard(path, metrics=None):
    """ Fetech recorded results from tensorboard.

        Args:
            path (str): tensorboard tfevent file path
            metrics (list, optional): names of metrics, e.g., `test/N@10`, which should be aligned with metric names in tensorboard.
                all metrics will be returned if `metrics` is None. default=None.

        Returns: 
            dict: dictionary of (metric_name, record_values), e.g., {`valid/N@10`: [0.1, 0.5, 0.3]}.
            list: metric names, e.g., [`valid/N@10`, `valid/R@10`]
    """
    event = EventAccumulator(path); event.Reload()
    all_metrics = event.Tags()['scalars']
    metrics = all_metrics if metrics is None else [m for m in metrics if m in all_metrics]
    info = defaultdict(list)
    for m in metrics:
        for e in event.Scalars(m):
            info[m].append(e.value)
    return info, metrics

def get_dataframe(info, metrics=None):
    """ Get two dataframes given info.

        Args:
            info (dict): dictionary of results, the structure should be ``{key: {metric1: value or list of values, metric2: value or list of values}}``.

        Returns:
            pandas.DataFrame: a pd.DataFrame where each row is a file, each columns is the testing value per metric
            pandas.DataFrame: a pd.DataFrame which is similar to the first one but grouped by random seeds, thus we also get std of metric 
    """

    def mean(values):
        """ Calculate mean for values or list of values
        """
        values = values.tolist()
        if type(values[0]) == list:
            lists = values
            max_len = max([len(l) for l in lists])
            res = []
            for i in range(max_len):
                res.append(np.mean([l[i] for l in lists if i < len(l)]))
            return res 
        else:
            return np.mean(values)

    def std(values):
        """ Calculate std for values or list of values
        """
        values = values.tolist()
        if type(values[0]) == list:
            lists = values
            max_len = max([len(l) for l in lists])
            res = []
            for i in range(max_len):
                res.append(np.std([l[i] for l in lists if i < len(l)]))
            return res 
        else:
            return np.std(values)

    data = defaultdict(list) 
    for i, key in enumerate(info):
        if i == 0:
            metrics = list(info[key].keys())
        data[key].extend([info[key][m] for m in metrics])

    if metrics is None:
        return None, None

    df1 = pd.DataFrame(data=data).T
    df1 = df1.rename(columns={i: m for i, m in enumerate(metrics)})
    df1 = df1.reset_index()

    df2 = df1.copy()
    df2['index'] = df2['index'].apply(lambda x: 'seed_'.join(x.split('seed_')[:-1]))
    df2 = df2.groupby('index').agg([mean, std]).reset_index()

    return df1, df2


def plot_train_valid_results(df1, df2, save_to=None):
    """ Plot metrics of interest per epoch according to dataframes returned from ``get_train_valid_results``

        Args:
            df1 (pd.DataFrame): a pd.DataFrame where each row is a file, each columns is the list of values according to epochs per metric
            df2 (pd.DataFrame): a pd.DataFrame which is similar to the first one but grouped by random seeds, thus we also get std of metric
            save_to (str): directory name to save reports
    """

    from matplotlib import pyplot as plt 

    if save_to is not None and not os.path.exists(save_to):
        os.makedirs(save_to)
    
    for m in df1.columns[1:]:
        plt.figure(figsize=(7, 4))
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(m.title(), fontsize=10)

        for i, row in df1.iterrows():
            plt.plot(row[m], label=row['index'], alpha=0.8)

        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_to, f"individual_{'_'.join(m.split('/'))}.jpg"), dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()

    for m in [c[0] for c in df2.columns[1:]]:
        plt.figure(figsize=(7, 4))
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(m.title(), fontsize=10)

        for i, row in df2.iterrows():
            mean, std = np.array(row[(m, 'mean')]), np.array(row[(m, 'std')]), 
            plt.plot(mean, label=row['index'][0], alpha=0.8)
            plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)

        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_to, f"grouped_{'_'.join(m.split('/'))}.jpg"), dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()



def get_test_results(dirnames, pattern="**/events.out.tfevents.*.2", metrics=None, save_to=None):
    """ Given pattern to retrieve all the results as two dataframes.
    
        Args:
            dirnames (dict): name of output directories, e.g., ``{alias1:bert4rec-tf/checkpoints/, alias2:bert4rec-pytorch/checkpoints}``.
            pattern (str): pattern used in ``glob`` to match paths, e.g., ``**/events.out.tfevents.*.2`` returns all file 
                           paths ``containing**/events.out.tfevents.*.2`` (which should be testing files).
            metrics (list): select metrics of interest
            save_to (str): directory name to save reports
            
        Returns:
            pandas.DataFrame: a pd.DataFrame where each row is a file, each columns is the testing value per metric
            pandas.DataFrame: a pd.DataFrame which is similar to the first one but grouped by random seeds, thus we also get std of metric
    """
    
    info = {}

    for dirname in dirnames: 
        dirpath = dirnames[dirname]
        for path in glob(os.path.join(dirpath, pattern), recursive=True):
            info_, metrics = get_results_from_tensorboard(path, metrics)
            key = dirname + '/' + path.replace(dirpath, '').split('lightning_logs')[0]
            info[key.replace('//', '/')] = {m: info_[m][-1] for m in metrics}

    df1, df2 = get_dataframe(info)

    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        df1.to_csv(os.path.join(save_to, 'individual_test.csv'), index=None)
        df2.to_csv(os.path.join(save_to, 'grouped_test.csv'), index=None)

    return df1, df2


def get_train_valid_results(dirnames, pattern="**/events.out.tfevents.*.1", metrics=None, save_to=None):
    """ Given pattern to retrieve all the results as two dataframes.
    
        Args:
            dirnames (dict): name of output directories, e.g., ``{alias1:bert4rec-tf/checkpoints/, alias2:bert4rec-pytorch/checkpoints}``.
            pattern (str): pattern used in ``glob`` to match paths, e.g., ``**/events.out.tfevents.*.1`` returns all file 
                           paths containing ``**/events.out.tfevents.*.1`` (which should be training files).
            metrics (list): select metrics of interest
            save_to (str): directory name to save reports
            
        Returns:
            pandas.DataFrame: a pd.DataFrame where each row is a file, each columns is the list of values according to epochs per metric
            pandas.DataFrame: a pd.DataFrame which is similar to the first one but grouped by random seeds, thus we also get std of metric
    """
    
    info = {}
    
    for dirname in dirnames: 
        dirpath = dirnames[dirname]
        for path in glob(os.path.join(dirpath, pattern), recursive=True):
            info_, metrics = get_results_from_tensorboard(path, metrics)
            key = dirname + '/' + path.replace(dirpath, '').split('lightning_logs')[0]
            info[key.replace('//', '/')] = {m: info_[m] for m in metrics}

    df1, df2 = get_dataframe(info)

    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        df1.to_csv(os.path.join(save_to, 'individual_train_valid.csv'), index=None)
        df2.to_csv(os.path.join(save_to, 'grouped_train_valid.csv'), index=None)

    return df1, df2

if __name__ == "__main__":
    # setting
    pd.set_option('display.max_colwidth', None)

    # parsing
    dirnames = {}
    for d in set(sys.argv[1:]):
        d = d.split(":")
        if len(d) == 1:
            dirnames[d[0]] = d[0]
        elif len(d) == 2:
            dirnames[d[0]] = d[1]

    # generate

    metrics = ['test/R@5', 'test/N@5', 'test/R@10', 'test/N@10', 'test/R@20', 'test/N@20']

    save_dir = os.path.join("reports", "_".join("_".join(list(dirnames.keys())).replace('../', '').split("/")))+"_"+dt.now().strftime('%Y-%m-%d-%H-%M-%S')
    df1, df2 = get_test_results(dirnames, save_to=save_dir, metrics=metrics)
    print('\nINDIVIDUAL TESTING RESULTS:\n', df1)
    print('\nGROUPED TESTING RESULTS:\n', df2)

    metrics = ['valid/R@5', 'valid/N@5', 'valid/R@10', 'valid/N@10', 'valid/R@20', 'valid/N@20', 'train/loss']

    df1, df2 = get_train_valid_results(dirnames, save_to=save_dir, metrics=metrics)

    print('\nSAVING TABLES & PLOTS to: \n', save_dir)    

    plot_train_valid_results(df1, df2, save_to=save_dir)