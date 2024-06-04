import re
from ast import literal_eval
from unittest.mock import patch
from jsonargparse import ActionConfigFile, ArgumentParser, capture_parser, REMAINDER

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneCallback

from main import lightning_cli


re_tune_func = re.compile(r'^tune.(\w+)\((.+)\)$')

# def str2bool(s):
#     s = s.lower()
#     if s not in {'false', 'true'}:
#         raise ValueError('Not a valid boolean string')
#     return s == 'true'

def eval_tune_run_config(config):
    """Evaluates a string into a python statement, but only allowing tune.* functions."""
    if not config:
        return
    for key, val in config.items():
        val_match = re_tune_func.match(val)
        assert val_match and hasattr(tune, val_match.group(1))
        func = getattr(tune, val_match.group(1))
        print(key, val_match)
        args = literal_eval(val_match.group(2))
        if not isinstance(args, tuple):
            args = (args,)
        config[key] = func(*args)


def ray_tune_cli(lightning_cli):
    parser = ArgumentParser()
    # parser.add_argument('--resume', type=str, default='False')
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(CLIReporter, 'reporter')#, skip={'parameter_columns'})
    parser.add_subclass_arguments(TuneCallback, 'tune_callback', instantiate=False)
    parser.add_function_arguments(tune.run, 'run', skip={'run_or_experiment', 'progress_reporter'})#, 'config'
    parser.add_argument(
        'fit_args',
        nargs=REMAINDER,
        help='All arguments after the double dash "--" are forwarded to the LightningCLI-based function. It is optional, ignore following text.',
    )
    cfg = parser.parse_args()

    fit_parser = capture_parser(lightning_cli)
    subcommand = []
    if fit_parser._subcommands_action:
        fit_parser = fit_parser._subcommands_action._name_parser_map['fit']
        subcommand = ['fit']

    if len(cfg.fit_args) > 1:
        fit_cfg = fit_parser.parse_args(cfg.fit_args[1:])
    else:
        fit_cfg = fit_parser.get_defaults()

    callbacks = [cfg.tune_callback]
    callbacks += [fit_cfg.get('trainer.callbacks')] or []

    fit_cfg['trainer.callbacks'] = callbacks

    fit_dump = fit_parser.dump(fit_cfg)

    def fit_function(config):
        args = [f'--config={fit_dump}'] if fit_dump else []
        for key, val in config.items():
            args.append(f'--{key}={val}')
        with patch('sys.argv', [''] + subcommand + args):
            lightning_cli()

    cfg_init = parser.instantiate_classes(cfg)
    eval_tune_run_config(cfg_init.run.config)

    analysis = tune.run(
        fit_function,
        progress_reporter=cfg_init.reporter,
        **cfg_init.run.as_dict(),
        # resume=cfg.resume
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    ray_tune_cli(lightning_cli)