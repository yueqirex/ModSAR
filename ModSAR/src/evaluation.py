import ray
from ray import tune
from ray.tune import ExperimentAnalysis
from pprint import pprint
import argparse
import torch
from collections import Counter
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import pytorch_lightning as pl

import pandas as pd

import os 
import sys
import json
import glob
import pdb
import shutil
import yaml

from pytorch_lightning import Trainer, seed_everything
from design import modularized

from IPython.display import display
from copy import copy

def str2bool(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--results_dir', type=str, default=None, help='ray tune results parent dir')
parser.add_argument('--exp_name', type=str, default=None, help='experiment name tuned by ray')
parser.add_argument('--mode', type=str, default=None, help='mode of valid or test')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')
# special args
parser.add_argument('--with_config', type=str2bool, default=False)
parser.add_argument('--succinct', type=str2bool, default=True)
args = parser.parse_args()


def get_best_exp():
    assert args.exp_name is not None, "experiment name incorrect"
    print(f"testing best args from: {args.results_dir}/{args.exp_name}")
    # ray.init()
    analysis = tune.ExperimentAnalysis(f'{args.results_dir}/{args.exp_name}')

    # Retrieve the best trial based on a specific metric
    best_trial = analysis.get_best_trial(metric="recall", mode="max")

    # Get the best hyperparameters and metrics
    best_params = best_trial.config
    best_metrics = best_trial.metric_analysis
    best_trial_dir = best_trial.logdir

    print("Best Hyperparameters: ")
    pprint(best_params)
    print("Best Metrics: ")
    pprint(best_metrics)
    # print("Best results dir: ")
    # pprint(best_trial_dir)
    return best_trial_dir



def copy_best_trial_dir(test_dir = None):
    # set the path
    assert test_dir is not None, "test dir is none"
    # copy the best dir to dest path to access easily
    dst_dir = os.path.join(f'{args.results_dir}/{args.exp_name}/best_trial')
    if os.path.isdir(dst_dir):
        print('=== NOTE: best trial dir already exists ===')
        print('=== NOTE: best trial dir removed and ready to create a new one ===')
        shutil.rmtree(dst_dir)
        shutil.copytree(test_dir, dst_dir)
        print(f'=== NOTE: best trial dir copied sucessfully to {args.results_dir}/{args.exp_name}/best_trial ===')
    else:
        shutil.copytree(test_dir, dst_dir)
        print(f'=== NOTE: best trial dir copied sucessfully to {args.results_dir}/{args.exp_name}/best_trial ===')



def save_test(test_dir = None):
    # set the path
    assert test_dir is not None, "test dir is none"
    ckpt_dir = os.path.join(test_dir, 'output_lightning/csv_logs/version*/checkpoints/epoch*')
    # pdb.set_trace()
    assert len(glob.glob(ckpt_dir)) == 1, "more than 1 checkpoint found"
    ckpt_path = glob.glob(ckpt_dir)[0]
    # path = sys.argv[1] # e.g., ../version_0/checkpoints/epoch=100.ckpt
    test_result_path_ = os.path.join(test_dir, 'output_lightning/csv_logs/version*/checkpoints')
    assert len(glob.glob(test_result_path_)) == 1
    test_result_path = os.path.join(glob.glob(test_result_path_)[0], 'test_result.json')

    # if the test result already exists, skip
    if os.path.exists(test_result_path):
        print('The test result already exists.')
        return
    else:
        print(f'creating test results dir at: {test_result_path}')
    # set the trainer
    trainer = Trainer(default_root_dir=None, auto_select_gpus=True, accelerator=args.device, devices=1, deterministic=True)

    # convert ckpt to have consistent datadir for current machine
    ckpt = torch.load(ckpt_path)
    ds_name = ckpt['datamodule_hyper_parameters']['data_dir'].split('/')[-1]
    ckpt['datamodule_hyper_parameters']['data_dir'] = os.path.join(os.getcwd(), '../data/modularized', ds_name)
    converted_ckpt_path = os.path.join(glob.glob(test_result_path_)[0], 'converted_ckpt.ckpt')
    torch.save(ckpt, converted_ckpt_path)

    # load the model and data
    data = modularized.LitData.load_from_checkpoint(converted_ckpt_path)
    model = modularized.LitModel.load_from_checkpoint(converted_ckpt_path)
    

    # test
    res = trainer.test(model, datamodule=data)
    print(res)

    # save the result
    with open(test_result_path, "w") as f:
        json.dump(res[0], f, indent=2)
    print('=== NOTE: best test results saved successfully ===')

    # copy the best dir to dest path to access easily
    dst_dir = os.path.join(f'{args.results_dir}/{args.exp_name}/best_trial')
    shutil.copytree(test_dir, dst_dir)
    print('=== NOTE: best trial dir copied sucessfully (with test results) ===')




def compare_test_results():
    # aspect ls
    # aspect_ls = ['loss_type_ce', 'loss_type_bce', 'loss_type_bpr', 'local_attn', 'ssept']
    # dataset_ls = ['beauty', 'sports', 'video', 'yelp', 'ml-1m']
    # task_ls = ['ae', 'ar']
    aspect_ls = ['albert','llama','transfo_xl']
    task_ls = ['ae', 'ar']
    dataset_ls = ['beauty', 'sports', 'video', 'yelp', 'ml-1m']
    # results ls
    results_ls = []
    header_ls = ['Aspect', 'Dataset', 'Task']
    columns = ['test/Recall@1', 'test/Recall@5', 'test/Recall@10', 'test/Recall@20', 'test/Recall@50', 'test/Recall@100', \
               'test/NDCG@5', 'test/NDCG@10', 'test/NDCG@20', 'test/NDCG@50', 'test/NDCG@100']
    base_header_ls = copy(header_ls)
    for asp in aspect_ls:
        for ds in dataset_ls:
            for task in task_ls:
                # pdb.set_trace()
                lt_dir = os.path.join(args.results_dir, (asp + '_' + task +'_'+ ds), "best_trial/output_lightning_rerun/tb_logs/version*")
                if len(glob.glob(lt_dir))==0:
                    print('testing NA for: ', asp + '_' + task +'_'+ ds)
                    test_log = [-1] * len(columns)
                elif len(glob.glob(lt_dir))>1:
                    raise Exception(f'more than 1 version found AT: {lt_dir}')
                else:
                    # log_path = os.path.join(args.results_dir, (asp + '_' + task +'_'+ ds), "best_trial/output_lightning_rerun/tb_logs/version_0/metrics.csv")
                    log_path = os.path.join(glob.glob(lt_dir)[0], 'metrics.csv')
                    df_test = pd.read_csv(log_path)
                    if len(header_ls) == len(base_header_ls):
                        header_ls += columns
                        length = len(header_ls)
                    if 'test/Recall@20' not in df_test.columns:
                        print('testing unfinished for: ', asp + '_' + task +'_'+ ds)
                        test_log = [-1] * len(columns)
                    else:
                        test_log = df_test.iloc[[-1]][columns].values[0].tolist()
                        # pdb.set_trace()
                rs = [asp, ds, task] + test_log
                
                # read hps
                if args.with_config:
                    param_path = os.path.join(args.results_dir, (asp + '_' + task +'_'+ ds), "best_trial/params.json")
                    with open(param_path, 'r') as f:
                        params = json.load(f)
                    p_keys = sorted(list(params.keys()))
                    if len(header_ls)==length: header_ls += p_keys
                    p = [params[k] for k in p_keys]
                    rs += p

                results_ls.append(rs)
    # create pandas dataframe
    results_df = pd.DataFrame(data=results_ls, columns=header_ls)
    if args.with_config and args.succinct:
        results_df = results_df[base_header_ls+['model.dropout_prob', 'data.mask_prob', 'model.num_attention_heads', 'model.hidden_size']]
        results_df.to_csv(f'{args.results_dir}/test_results_with_config.csv')
    else:
        results_df.to_csv(f'{args.results_dir}/test_results.csv')
    display(results_df)
    print('=== NOTE: final test results saved to disk successfully ===')



def get_run_time(results_dir):
    analysis = ExperimentAnalysis(os.path.join(results_dir, "local_attn_ar_beauty"))
    # best_trial = analysis.get_best_trial(metric="recall", mode="max")
    print(analysis.stats())
    # best_params = best_trial.config
    # best_metrics = best_trial.metric_analysis
    # best_trial_dir = best_trial.logdir

    # print("Best Hyperparameters: ")
    # pprint(best_params)
    # print("Best Metrics: ")
    # pprint(best_metrics)



def ray_error_detection(results_dir):
    def check_errored_experiments(results_dir):
        # such as 'your_path_to_ModSAR/src/output_ray'
        analysis = ExperimentAnalysis(results_dir)
        num_exp = 0
        errored_experiments = []
        for trial in analysis.trials:
            num_exp += 1
            if trial.error_file: errored_experiments.append(trial)
        return errored_experiments, num_exp
    # Check for errored experiments
    # hf
    aspect_ls = ['transfo_xl']
    task_ls = ['ae', 'ar']
    dataset_ls = ['beauty', 'sports', 'video', 'yelp', 'ml-1m']
    # modularized
    # aspect_ls = ['ssept']
    # task_ls = ['ae', 'ar']
    # dataset_ls = ['beauty', 'sports']
    unfinished_exp_ls = []
    not_tuned_ls = []
    for asp in aspect_ls:
        for task in task_ls:
            for ds in dataset_ls:
                print(f'=== this is for {asp}_{task}_{ds} ===')
                exp_dir = os.path.join(f'{args.results_dir}', asp+'_'+task+'_'+ds)
                if not os.path.isdir(exp_dir): 
                    print(f'=== NOTE: not tuned for {asp}_{task}_{ds} ===')
                    not_tuned_ls.append(f'{asp}_{task}_{ds}')
                else:
                    analysis = ExperimentAnalysis(exp_dir)
                    status_ls = [trial.status for trial in analysis.trials]
                    status_count = dict(Counter(status_ls))
                    status_count
                    status_rate = {k: v/sum(list(status_count.values())) for k, v in status_count.items()}
                    print(status_count)
                    print(status_rate)
                    
                    if list(status_rate.items())[0][1]<1.0:
                        unfinished_exp_ls.append((asp+'_'+task+'_'+ds, status_rate))
    print('\n')
    for exp in not_tuned_ls:
        print('=== NOTE: Untuned/Errored exp: ===')
        print(exp)
    for info, exp in unfinished_exp_ls:
        print('=== NOTE: Unfinished/Errored exp: ===')
        print(info, exp)



def export_best_config():
    '''
        export best config as a new yaml -> best_trial dir
    '''
    splits = args.exp_name.split('_')
    task = splits[-2]
    ds = splits[-1]
    asp = args.exp_name.replace('_'+task, '').replace('_'+ds, '')
    if args.results_dir=='output_hf':
        std_config = os.path.join(f'configs/hugging_face/{asp}/{ds}_config_run.yaml')
    else:
        std_config = os.path.join(f'configs/{asp}/{ds}_config_run.yaml')
    with open(std_config, 'r') as file:
        config = yaml.safe_load(file)

    
    best_config = os.path.join(f'{args.results_dir}/{args.exp_name}/best_trial/output_lightning/csv_logs/version_0/hparams.yaml')
    with open(best_config, 'r') as file:
        best_config = yaml.safe_load(file)


    for k in best_config.keys():
        if k in ['batch_size', 'data_dir', 'item_sse_prob', \
                 'mask_prob', 'user_sse_prob']:
            config['data']['init_args'][k] = best_config[k]
        else:
            config['model']['init_args'][k] = best_config[k]

    # Create the first callback dictionary
    callback1 = {
        'class_path': 'pytorch_lightning.callbacks.ModelCheckpoint',
        'init_args': {
            'save_top_k': 5,
            'monitor': 'valid/Recall@10',
            'mode': 'max',
            'every_n_epochs': 5
        }
    }
    callback2 = {
        'class_path': 'pytorch_lightning.callbacks.early_stopping.EarlyStopping',
        'init_args': {
            'patience': 10,
            'monitor': 'valid/Recall@10',
            'verbose': False,
            'mode': 'max'
        }
    }

    logger2 = {
        'class_path': 'pytorch_lightning.loggers.TensorBoardLogger',
        'init_args': {
            'name': 'csv_logs',
            'save_dir': f'{args.results_dir}/{args.exp_name}/best_trial/output_lightning_rerun'
        }
    }

    logger1 = {
        'class_path': 'pytorch_lightning.loggers.CSVLogger',
        'init_args': {
            'name': 'tb_logs',
            'save_dir': f'{args.results_dir}/{args.exp_name}/best_trial/output_lightning_rerun'
        }
    }

    config['trainer']['callbacks'] = [callback1, callback2]
    config['trainer']['logger'] = [logger1, logger2]
    config['trainer']['max_epochs'] = 2000
    config['trainer']['devices'] = [0]
    config['data']['init_args']['data_dir'] = '../data/modularized/' + config['data']['init_args']['data_dir'].split('/')[-1]
    # Save the modified YAML file
    with open(f'{args.results_dir}/{args.exp_name}/best_trial/best_config.yaml', 'w') as file:
        yaml.safe_dump(config, file)
    print(f'=== NOTE: best config saved successfully for {args.exp_name} ===')
    




if __name__ == '__main__':
    seed_everything(args.seed)
    assert args.mode is not None, 'mode not specified correctly'
    if args.mode == 'detect_error':
        ray_error_detection(f'{args.results_dir}')
    elif args.mode == 'export_best_config':
        best_trial_dir = get_best_exp()
        copy_best_trial_dir(best_trial_dir)
        export_best_config()
    # generate summary dataframe
    elif args.mode == 'compare':
        compare_test_results()
    # detect run time
    elif args.mode == 'get_run_time':
        get_run_time(args.results_dir)
    # === DEPRECATED ===
    elif args.mode == 'val':
        # exp dir is the one for the current aspect to check (e.g. loss_type, ssept etc.)
        # best trial dir is the one with the best args of the current exp
        best_trial_dir = get_best_exp()
    elif args.mode == 'test':
        best_trial_dir = get_best_exp()
        save_test(best_trial_dir)
