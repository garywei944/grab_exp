from math import pi
from os import times
from sched import scheduler
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger

import wandb

# from lightning.pytorch.profilers import PyTorchProfiler
import torch

import evaluate

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pickle

from glue_t5 import GLUET5Model
from glue_data_t2t import (
    GLUET2TDataModule,
    TASK_NAMES,
    TASK_VAL_NAMES,
    GLUE_TASK_TO_LABELS,
)

from cd2root import cd2root

cd2root()


def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)

    parser.add_lightning_class_args(GLUET2TDataModule, "data")
    # parser.add_lightning_class_args(GLUET5Model, "model")
    # parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")

    return parser.parse_args()


def main():
    wandb.init(project="t5-glue", name="eval", entity="grab")
    args = parse_args()

    model_name = args.data.model_name_or_path
    # task_name = args.model.task_name

    path = Path("checkpoints/google/t5-v1_1-small/glue/finetune")

    L.seed_everything(args.seed)

    dm = GLUET2TDataModule(**vars(args.data))
    trainer = L.Trainer(
        logger=[
            CSVLogger("logs", name=model_name),
        ],
        # callbacks=[
        #     Timer(),
        # ],
        # profiler="simple",
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # limit_val_batches=0.5,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )

    best_results = {}
    for ckpt in path.glob("*.ckpt"):
        # print(ckpt)

        step = int(ckpt.stem.split("=")[-1])
        # print(step)
        model = GLUET5Model.load_from_checkpoint(ckpt)
        trainer.validate(model, datamodule=dm)

        best_results[step] = model.best

        print(model.best)

        wandb.log(model.best, step=step)

    # save best results
    pickle.dump(best_results, open("best_results.pkl", "wb"))


if __name__ == "__main__":
    main()
