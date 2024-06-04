# main.py
import os 
import sys
import json

from pytorch_lightning import Trainer
from design import modularized


if __name__ == "__main__":
    # set the path
    path = sys.argv[1] # e.g., ../version_0/checkpoints/epoch=100.ckpt
    test_result_path = os.path.join('/'.join(path.split('/')[:-1]), 'test_result.json')

    # if the test result already exists, skip
    if os.path.exists(test_result_path):
        print('The test result already exists.')
        exit(0)

    # set the trainer
    trainer = Trainer(default_root_dir=None, auto_select_gpus=True, gpus=1)

    # load the model and data
    model = modularized.LitModel.load_from_checkpoint(path)
    data = modularized.LitData.load_from_checkpoint(path)

    # test
    res = trainer.test(model, datamodule=data)

    # save the result
    json.dump(res[0], open(test_result_path, "w"), indent=2)
