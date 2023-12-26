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
from lightning.pytorch.profilers import PyTorchProfiler
import torch

from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.optimization import Adafactor, get_scheduler

import evaluate

from collections import defaultdict

from glue_data import GLUEDataModule, GLUE_TASK_NUM_LABELS

from cd2root import cd2root

cd2root()


class GLUEModel(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

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

        self.metric2 = (
            evaluate.load(
                "glue",
                "mnli",
            )
            if self.task_name == "mnli"
            else None
        )

        self.best = defaultdict(lambda: float("-inf"))

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.num_labels > 1:
            predictions = torch.argmax(logits, dim=-1)
        else:
            predictions = logits.squeeze()

        # Special handling for MNLI
        metric = self.metric if dataloader_idx == 0 else self.metric2
        metric.add_batch(
            predictions=self.all_gather(predictions).flatten(),
            references=self.all_gather(batch["labels"]).flatten(),
        )

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": val_loss, "predictions": predictions, "labels": batch["labels"]}

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            self.metric.compute(),
            prog_bar=True,
        )

        if self.metric2 is not None:
            results = self.metric2.compute()
            results = {f"{k}/mm": v for k, v in results.items()}
            self.log_dict(
                results,
                prog_bar=True,
            )

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
        # T5 hyperparameter from Google paper and
        # https://discuss.huggingface.co/t/t5-finetuning-tips/684/36
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        # scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=self.warmup_steps,
        #     num_training_steps=self.trainer.max_steps,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        # }

        return optimizer


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
    parser.add_lightning_class_args(GLUEModel, "model")
    parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
    parser.link_arguments("model.task_name", "data.task_name")

    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model.model_name_or_path
    task_name = args.model.task_name

    L.seed_everything(args.seed)

    model = GLUEModel(**vars(args.model))
    dm = GLUEDataModule(**vars(args.data))

    L.seed_everything(args.seed)
    trainer = L.Trainer(
        max_steps=args.max_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_interval,
        logger=[
            TensorBoardLogger("lightning_logs", name=model_name),
            WandbLogger(
                project=f"t5-glue-{task_name}",
                name=model_name,
                entity="grab",
                mode="online",
            ),
            CSVLogger("logs", name=model_name),
        ],
        callbacks=[
            # EarlyStopping(monitor="val_loss"),
            # ModelCheckpoint(
            #     monitor="matthews_correlation",
            # ),
            LearningRateMonitor(),
            Timer(),
        ],
        profiler=PyTorchProfiler(),
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    trainer.validate(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
