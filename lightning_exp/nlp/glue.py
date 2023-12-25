import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    Timer,
    TQDMProgressBar,
)
import torch
from torch_optimizer import Adafactor

from transformers import AutoConfig, AutoModelForSequenceClassification
import evaluate

from collections import defaultdict
from dataclasses import dataclass, field

from glue_data import GLUEDataModule, GLUE_TASK_NUM_LABELS

from cd2root import cd2root

cd2root()


@dataclass
class GLUEModel(L.LightningModule):
    model_name_or_path: str
    task_name: str
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    def __post_init__(self):
        super().__init__()

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


def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-T", "--max_steps", type=int, default=250_000)
    parser.add_argument(
        "-vi",
        "--val_interval",
        type=int,
        default=1000,
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

    trainer = L.Trainer(
        max_steps=args.max_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_interval,
        logger=[
            TensorBoardLogger("lightning_logs", name=model_name),
            WandbLogger(
                project=f"t5-glue-{task_name}",
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
            Timer(),
        ],
        # fast_dev_run=True,
        # limit_train_batches=0.01,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
