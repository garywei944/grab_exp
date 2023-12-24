import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
from torch import nn, Tensor
from torch_optimizer import Adafactor

from transformers import AutoConfig, AutoModelForSequenceClassification
import evaluate

import os
from pathlib import Path
from datetime import datetime
from attrs import define, field

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
        task_name: str = "mrpc",
        num_labels: int = 2,
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
        self.metric = evaluate.load(
            "glue",
            self.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
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
            preds = torch.argmax(logits, dim=-1)
        else:
            preds = logits.squeeze()

        return {"loss": val_loss, "preds": preds, "labels": batch["labels"]}

    # def on_validation_epoch_end(self) -> None:
    #     print("on_validation_epoch_end")
    #     pass

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
        )


def main():
    # from datargs import parse, make_parser
    #
    # parser = make_parser(GLUEDataModule)
    # args = parser.parse_args()
    #
    # print(args)

    dm = GLUEDataModule("t5-small", "mrpc")
    dm.prepare_data()
    dm.setup()

    model = GLUEModel("t5-small", num_labels=dm.num_labels, task_name="mrpc")

    # cli = CLI(GLUEModel, GLUEDataModule, run=False)
    #
    trainer = L.Trainer(
        max_steps=2**18,
        precision="16-mixed"
        # fast_dev_run=True,
    )
    trainer.fit(model, datamodule=dm)
    # trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
