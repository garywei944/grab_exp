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

from glue_data_t2t import (
    GLUET2TDataModule,
    TASK_NAMES,
    TASK_VAL_NAMES,
    GLUE_TASK_TO_LABELS,
)

from cd2root import cd2root

cd2root()


class GLUET5Model(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/t5-v1_1-small",
        optimizer: str = "adafactor",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,  # Not used by Adafactor
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
            legacy=False,
        )

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name_or_path, config=self.config
        )
        self.metrics = [
            evaluate.load(
                "glue",
                task_name.split("_")[0],
            )
            for task_name in TASK_VAL_NAMES
        ]

        self.best = defaultdict(lambda: float("-inf"))

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        def string2label(s: str):
            if not is_regression:
                try:
                    return GLUE_TASK_TO_LABELS[true_task_name].index(s)
                except ValueError:
                    return -1
            else:
                try:
                    return float(s)
                except ValueError:
                    return -1.0

        true_task_name = TASK_VAL_NAMES[dataloader_idx].split("_")[0]
        # outputs = self(**batch)
        # val_loss, logits = outputs[:2]
        val_loss = self(**batch).loss

        is_regression = true_task_name == "stsb"

        # This is actually Greedy Search according to T5 paper
        tokens = self.model.generate(
            batch["input_ids"],
            max_length=10,
        )
        texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

        try:
            labels = self.tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
        except OverflowError:
            labels = batch["labels"].clone()
            labels[labels == -100] = self.tokenizer.pad_token_id
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        if batch_idx == 0:
            print("-" * 80)
            print("texts and labels")
            print(texts)
            print(labels)

        # The following is consistent with T5X implementation of evaluation
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/f1f16c0d77a7c3a6a21b5cf9f3e96adf26a71c60/t5/data/postprocessors.py#L28_L49
        predictions = [string2label(e) for e in texts]
        labels = [string2label(e) for e in labels]

        if batch_idx == 0:
            print("-" * 80)
            print("predictions and labels")
            print(predictions)
            print(labels)
        # Hack to fix -1 to be different with labels s.t. the code is compatible with
        # Huggingface evaluate
        if not is_regression:
            num_labels = len(GLUE_TASK_TO_LABELS[true_task_name])
            predictions = [
                (y_hat + 1) % num_labels if y_pred == -1 else y_pred
                for y_pred, y_hat in zip(predictions, labels)
            ]

        predictions = torch.tensor(predictions, device=self.device)
        labels = torch.tensor(labels, device=self.device)

        metric = self.metrics[dataloader_idx]
        metric.add_batch(
            predictions=self.all_gather(predictions).flatten(),
            references=self.all_gather(labels).flatten(),
        )

        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True)

        return {"loss": val_loss, "predictions": predictions, "labels": labels}

    def on_validation_epoch_end(self) -> None:
        results = {}
        for task_name, metric in zip(TASK_VAL_NAMES, self.metrics):
            result = metric.compute()
            for k, v in result.items():
                results[f"val/{task_name}-{k}"] = v

        self.log_dict(
            results,
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

        if self.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = get_scheduler(
                "constant_with_warmup",
                # "cosine_with_restarts",
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.trainer.max_steps,
            )
        elif self.optimizer == "adafactor":
            # T5 hyperparameter from Google paper and
            # https://discuss.huggingface.co/t/t5-finetuning-tips/684/36
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                relative_step=False,
                scale_parameter=True,
                warmup_init=False,
            )
            # scheduler = AdafactorSchedule(optimizer, initial_lr=self.learning_rate)
            # scheduler = AdafactorSchedule(optimizer)

            scheduler = get_scheduler(
                "constant_with_warmup",
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

    parser.add_lightning_class_args(GLUET2TDataModule, "data")
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
    dm = GLUET2TDataModule(**vars(args.data))

    trainer = L.Trainer(
        max_steps=args.max_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_interval,
        # gradient_clip_val=1.0,
        logger=[
            TensorBoardLogger("lightning_logs", name=model_name),
            WandbLogger(
                project=f"t5-glue",
                name=model_name,
                entity="grab",
                mode="online",
            ),
            CSVLogger("logs", name=model_name),
        ],
        callbacks=[
            # EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(
                # monitor="matthews_correlation",
                save_top_k=-1,
                dirpath=f"checkpoints/{model_name}/glue/{timestamp}",
                every_n_train_steps=1000,
                # save_weights_only=True,
                save_last=True,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            Timer(),
        ],
        profiler="simple",
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        limit_val_batches=0,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    # trainer.validate(model, datamodule=dm)
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path="checkpoints/google/t5-v1_1-small/glue/2023-12-28T18:12:26.650635/epoch=7-step=54000.ckpt",
    )


if __name__ == "__main__":
    main()
