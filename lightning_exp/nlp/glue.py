from collections import defaultdict
import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    Timer,
    TQDMProgressBar,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
import torch
from torch import nn, Tensor
from torch_optimizer import Adafactor
from traitlets import default

from transformers import AutoConfig, AutoModelForSequenceClassification
import evaluate

import os
from pathlib import Path
from collections import defaultdict

from glue_dm import GLUEDataModule, GLUE_TASK_NUM_LABELS

from cd2root import cd2root

cd2root()


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
        parser.link_arguments("model.task_name", "data.task_name")


class GLUEModel(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

        self.num_labels = GLUE_TASK_NUM_LABELS[self.task_name]

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=self.config
        )
        self.metric = evaluate.load(
            "glue",
            self.task_name,
        )

        self.val_predictions = defaultdict(list)
        self.val_labels = defaultdict(list)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.num_labels > 1:
            predictions = torch.argmax(logits, dim=-1)
        else:
            predictions = logits.squeeze()

        self.val_predictions[dataloader_idx].append(predictions)
        self.val_labels[dataloader_idx].append(batch["labels"])

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": val_loss, "predictions": predictions, "labels": batch["labels"]}

    def on_validation_epoch_end(self) -> None:
        n_val = len(self.val_predictions)

        for idx in range(n_val):
            predictions = torch.cat(self.val_predictions[idx]).flatten()
            labels = torch.cat(self.val_labels[idx]).flatten()

            predictions = self.all_gather(predictions).flatten()
            labels = self.all_gather(labels).flatten()

            self.val_predictions[idx].clear()
            self.val_labels[idx].clear()

            results = self.metric.compute(predictions=predictions, references=labels)

            if n_val > 1:
                results = {f"{k}/{idx}": v for k, v in results.items()}

            self.log_dict(results, prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if all(nd not in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return Adafactor(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0,
            scale_parameter=True,
        )


def main():
    # from datargs import parse, make_parser
    #
    # parser = make_parser(GLUEDataModule)
    # args = parser.parse_args()
    #
    # print(args)

    # MODEL_NAME = "google/t5-v1_1-small"
    # TASK_NAME = "mnli"

    # L.seed_everything(42)

    # dm = GLUEDataModule(
    #     MODEL_NAME, TASK_NAME, train_batch_size=32, num_workers=4, load_from_disk=True
    # )
    # # dm.prepare_data()
    # # dm.setup()

    # model = GLUEModel(MODEL_NAME, num_labels=dm.num_labels, task_name=TASK_NAME)

    cli = CLI(GLUEModel, GLUEDataModule, run=False)

    trainer = L.Trainer(
        max_steps=250_000,
        check_val_every_n_epoch=None,
        val_check_interval=1000,
        logger=[
            TensorBoardLogger("lightning_logs", name=cli.config["model_name_or_path"]),
            WandbLogger(
                project=f"t5-glue-{cli.config['task_name']}",
                entity="grab",
                mode="online",
            ),
            CSVLogger("logs", name=cli.config["model_name_or_path"]),
        ],
        callbacks=[
            # EarlyStopping(monitor="val_loss"),
            # ModelCheckpoint(
            #     monitor="matthews_correlation",
            # ),
            Timer(),
        ],
        # fast_dev_run=True,
        # limit_train_batches=0.01,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
