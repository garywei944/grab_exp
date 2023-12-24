# Adapted from https://colab.research.google.com/github/PytorchLightning/pytorch-lightning
# /blob/master/notebooks/04-transformers-text-classification.ipynb#scrollTo=6yuQT_ZQMpCg

from argparse import ArgumentParser
from datetime import datetime
from typing import Optional
import os

import datasets
import pytorch_lightning as pl
import torch
import transformers
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from glue_data import GLUEDataModule

from torch_optimizer import Adafactor


class GLUETransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        finetuning_task: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        # save the hyperparameters. Access by self.hparams.[variable name]
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels, finetuning_task=finetuning_task
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.metric = datasets.load_metric(
            "glue",
            self.hparams.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, dim=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v
                    for k, v in self.metric.compute(
                        predictions=preds, references=labels
                    ).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        result = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(result, prog_bar=True)

        return loss

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # get number of assigned gpus
            n_gpus = 0
            if type(self.hparams.gpus) is list:
                n_gpus = len(self.hparams.gpus)
            elif type(self.hparams.gpus) is str:
                temp = [int(x.strip()) for x in self.hparams.gpus.split(",")]
                n_gpus = len(temp)
                if len(temp) == 1 and temp[0] == -1:
                    n_gpus = torch.cuda.device_count()
            elif type(self.hparams.gpus) is int:
                n_gpus = self.hparams.gpus
                if n_gpus == -1:
                    n_gpus = torch.cuda.device_count()

            # Calculate total steps
            self.total_steps = (
                (
                    len(train_loader.dataset)
                    // (self.hparams.train_batch_size * max(1, n_gpus))
                )
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser


def parse_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_model_specific_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)
    return parser.parse_args()


def main():
    # args
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    parser.add_argument("--seed", type=int, default=42)
    args = parse_args(parser)

    # seed
    pl.seed_everything(args.seed)

    # data
    data_module = GLUEDataModule.from_argparse_args(args)
    data_module.prepare_data()
    data_module.setup("fit")

    # model
    model = GLUETransformer(
        num_labels=data_module.num_labels,
        eval_splits=data_module.eval_splits,
        finetuning_task=data_module.task_name,
        **vars(args),
    )

    # training
    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=True)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
