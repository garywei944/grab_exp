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

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from transformers.optimization import Adafactor, AdafactorSchedule, get_scheduler

import evaluate

from collections import defaultdict
from datetime import datetime

from glue_data import GLUEDataModule, GLUE_TASK_NUM_LABELS

from cd2root import cd2root

cd2root()


class GLUET5Model(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "prajjwal1/bert-tiny",
        task_name: str = "qnli",
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,  # Not used by Adafactor
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.save_hyperparameters()

        num_labels = GLUE_TASK_NUM_LABELS[task_name]

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=task_name,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
            # legacy=False,
        )

        assert task_name != "mnli", "MNLI is not supported yet"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=self.config
        )
        self.metric = evaluate.load("glue", task_name)

        self.best = defaultdict(lambda: float("-inf"))

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss = outputs.loss

        is_regression = self.task_name == "stsb"

        predictions = (
            outputs.logits.squeeze() if is_regression else outputs.logits.argmax(dim=-1)
        )
        labels = batch["labels"]

        self.metric.add_batch(
            predictions=self.all_gather(predictions).flatten(),
            references=self.all_gather(labels).flatten(),
        )

        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True)

        return {"loss": val_loss, "predictions": predictions, "labels": labels}

    def on_validation_epoch_end(self) -> None:
        results = self.metric.compute()

        self.log_dict(
            results,
            prog_bar=True,
        )

        for k, v in results.items():
            self.best[k] = max(self.best[k], v)
        print(results)
        print(self.best)

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

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
            scheduler = get_scheduler(
                "linear",
                # "cosine_with_restarts",
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.trainer.max_steps,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = get_scheduler(
                "linear",
                # "cosine_with_restarts",
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.trainer.max_steps,
            )
        else:
            raise NotImplementedError
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

        # return optimizer


def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-T", "--max_steps", type=int, default=2500)
    parser.add_argument(
        "-vi",
        "--val_interval",
        type=int,
        default=100,
    )

    parser.add_lightning_class_args(GLUEDataModule, "data")
    parser.add_lightning_class_args(GLUET5Model, "model")
    parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
    # parser.link_arguments("model.task_name", "data.task_name")

    return parser.parse_args()


def main():
    timestamp = datetime.now().isoformat()
    args = parse_args()

    model_name = args.model.model_name_or_path
    # task_name = args.model.task_name

    L.seed_everything(args.seed)

    model = GLUET5Model(**vars(args.model))
    dm = GLUEDataModule(**vars(args.data))

    trainer = L.Trainer(
        # max_steps=args.max_steps,
        # check_val_every_n_epoch=None,
        max_epochs=5,
        # val_check_interval=args.val_interval,
        gradient_clip_val=1.0,
        logger=[
            TensorBoardLogger("lightning_logs", name=model_name),
            # WandbLogger(
            #     project=f"t5-glue",
            #     name=model_name,
            #     entity="grab",
            #     mode="online",
            # ),
            CSVLogger("logs", name=model_name),
        ],
        callbacks=[
            # EarlyStopping(monitor="val_loss"),
            # ModelCheckpoint(
            #     # monitor="matthews_correlation",
            #     save_top_k=-1,
            #     dirpath=f"checkpoints/{model_name}/glue/{timestamp}",
            #     every_n_train_steps=1000,
            #     # save_weights_only=True,
            #     save_last=True,
            #     verbose=True,
            # ),
            LearningRateMonitor(logging_interval="step"),
            Timer(),
        ],
        # profiler="simple",
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # limit_val_batches=0,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    # trainer.validate(model, datamodule=dm)
    trainer.fit(
        model,
        datamodule=dm,
        # ckpt_path="checkpoints/google/t5-v1_1-small/glue/2023-12-28T18:12:26.650635/epoch=7-step=54000.ckpt",
    )


if __name__ == "__main__":
    main()
