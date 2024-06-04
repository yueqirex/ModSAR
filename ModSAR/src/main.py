# main.py
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from design import *


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # add the arguments linking for the data and model
        LINKED_ARGS = ["max_position", "task_type", "eval_neg", "num_users", "num_items", "pad_token", "mask_token"]

        # set the shared arguments for the data module
        for arg in LINKED_ARGS:
            parser.link_arguments(f"model.init_args.{arg}", f"data.init_args.{arg}")

    def after_fit(self):
        # test after running
        self.trainer.test(self.model, datamodule=self.datamodule, ckpt_path='best')


def lightning_cli():
    MyLightningCLI(save_config_callback=None)


if __name__ == "__main__":
    lightning_cli()
