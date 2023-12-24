import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
from torch import nn, Tensor

from transformers import AutoConfig, AutoModelForSequenceClassification
import datasets

import os
from pathlib import Path
from datetime import datetime
from attrs import define, field

from glue_dm import GLUEDataModule


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("data", "model.dm", apply_on="instantiate")


class GLUEModel(L.LightningModule):
    model_name_or_path: str
    task_name: str = "mrpc"
    learning_rate: float = 2e-5
    warmup_steps: int = 20
    weight_decay: float = 0.0
    eval_splits: list | None = None

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.config = AutoConfig.from_pretrained(
            self.dm.model_name_or_path,
            num_labels=self.dm.num_labels,
            finetuning_task=self.dm.task_name,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=self.config
        )
        self.metric = datasets.load_metric(
            "glue",
            self.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        )


def main():
    cli = CLI(GLUEModel, GLUEDataModule, seed_everything_default=42, run=False)

    # from datargs import parse, make_parser
    #
    # parser = make_parser(GLUEDataModule)
    # args = parser.parse_args()
    #
    # print(args)

    pass


if __name__ == "__main__":
    main()
