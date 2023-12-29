from os import times
from sched import scheduler
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    Timer,
    TQDMProgressBar,
    LearningRateMonitor,
)

# from lightning.pytorch.profilers import PyTorchProfiler
import torch

from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule, get_scheduler

import evaluate

from collections import defaultdict
from datetime import datetime

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
    timestamp = datetime.now().isoformat()
    args = parse_args()

    model_name = args.data.model_name_or_path
    # task_name = args.model.task_name

    L.seed_everything(args.seed)

    model = GLUET5Model.load_from_checkpoint(
        "checkpoints/google/t5-v1_1-small/glue/2023-12-28T18:12:26.650635/last.ckpt"
    )
    dm = GLUET2TDataModule(**vars(args.data))

    trainer = L.Trainer(
        logger=[
            CSVLogger("logs", name=model_name),
        ],
        callbacks=[
            Timer(),
        ],
        # profiler="simple",
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # limit_val_batches=0,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    trainer.validate(model, datamodule=dm)
    # trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
