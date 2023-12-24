import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import torch
from torch import nn, Tensor
from torch_optimizer import Adafactor

from transformers import AutoConfig, AutoModelForSequenceClassification
import evaluate

import os
from pathlib import Path

from glue_dm import GLUEDataModule

from cd2root import cd2root

cd2root()


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
        parser.link_arguments("model.task_name", "data.task_name")
        parser.link_arguments(
            "model.num_labels", "data.num_labels", apply_on="instantiate"
        )
        ...


class GLUEModel(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        num_labels: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.task_name = task_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=self.config
        )
        self.train_metric = evaluate.load(
            "glue",
            self.task_name,
        )
        self.val_metric = evaluate.load(
            "glue",
            self.task_name,
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.num_labels > 1:
            predictions = torch.argmax(logits, dim=-1)
        else:
            predictions = logits.squeeze()

        self.val_metric.add_batch(predictions=predictions, references=batch["labels"])

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        return {"loss": val_loss, "predictions": predictions, "labels": batch["labels"]}

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metric.compute(), prog_bar=True)

    # def validation_epoch_end(self, outputs):
    #     preds = torch.cat([x["preds"] for x in outputs])
    #     labels = torch.cat([x["labels"] for x in outputs])
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #
    #     self.log("val_loss", loss, prog_bar=True, on_epoch=True)
    #
    #     result = self.metric.compute(predictions=preds, references=labels)
    #     self.log_dict(result)
    #
    #     return loss

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

    MODEL_NAME = "google/t5-v1_1-small"
    TASK_NAME = "cola"

    L.seed_everything(42)

    dm = GLUEDataModule(MODEL_NAME, TASK_NAME)
    dm.prepare_data()
    dm.setup()

    model = GLUEModel(MODEL_NAME, num_labels=dm.num_labels, task_name=TASK_NAME)

    # cli = CLI(GLUEModel, GLUEDataModule, run=False)
    #
    trainer = L.Trainer(
        max_steps=250_000,
        check_val_every_n_epoch=None,
        val_check_interval=1000,
        logger=[
            TensorBoardLogger("lightning_logs", name=MODEL_NAME),
            WandbLogger(
                project="t5-glue",
                entity="grab",
                mode="offline",
            ),
        ],
        callbacks=[
            # L.pytorch.callbacks.EarlyStopping(monitor="val_loss"),
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="f1",
                save_top_k=1,
                save_last=True,
                filename="{epoch:02d}-{val_loss:.2f}",
            ),
            L.pytorch.callbacks.Timer(),
        ],
        # fast_dev_run=True,
    )
    trainer.fit(model, datamodule=dm)
    # trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
