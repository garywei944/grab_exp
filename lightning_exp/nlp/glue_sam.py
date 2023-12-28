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
from transformers.optimization import Adafactor, AdafactorSchedule, get_scheduler

import torch

from glue_data import GLUEDataModule
from glue import GLUEModel

from cd2root import cd2root

cd2root()

from experiments.sam import SAM


class GLUESAMModel(GLUEModel):
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        rho: float = 0.05,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_name=task_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.rho = rho

        self.save_hyperparameters()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        def closure():
            loss = self(**batch).loss
            self.manual_backward(loss)
            if self.optimizer == "adam":
                self.clip_gradients(
                    opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                )
            return loss

        loss = self(**batch).loss
        self.manual_backward(loss)

        opt.step(closure=closure)
        opt.zero_grad()

        sch = self.lr_schedulers()
        sch.step()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

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
            optimizer = SAM(optimizer_grouped_parameters, optimizer, rho=self.rho)
            scheduler = get_scheduler(
                # "cosine_with_restarts",
                "constant_with_warmup",
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.trainer.max_steps,
            )
        else:
            # T5 hyperparameter from Google paper and
            # https://discuss.huggingface.co/t/t5-finetuning-tips/684/36
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                # lr=self.learning_rate,
                relative_step=True,
                scale_parameter=True,
                warmup_init=True,
            )
            optimizer = SAM(optimizer_grouped_parameters, optimizer, rho=self.rho)
            # scheduler = AdafactorSchedule(optimizer, initial_lr=self.learning_rate)
            scheduler = AdafactorSchedule(optimizer)

            # scheduler = get_scheduler(
            #     "constant_with_warmup",
            #     optimizer=optimizer,
            #     num_warmup_steps=self.warmup_steps,
            #     num_training_steps=self.trainer.max_steps,
            # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


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
    parser.add_lightning_class_args(GLUESAMModel, "model")
    parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
    parser.link_arguments("model.task_name", "data.task_name")

    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model.model_name_or_path
    task_name = args.model.task_name

    L.seed_everything(args.seed)

    model = GLUESAMModel(**vars(args.model))
    dm = GLUEDataModule(**vars(args.data))

    name = f"sam-{model_name}"

    L.seed_everything(args.seed)
    trainer = L.Trainer(
        strategy="deepspeed_stage_3",
        max_steps=args.max_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_interval,
        logger=[
            TensorBoardLogger("lightning_logs", name=name),
            WandbLogger(
                project=f"t5-glue-{task_name}",
                name=name,
                entity="grab",
                mode="online",
            ),
            CSVLogger("logs", name=name),
        ],
        callbacks=[
            # EarlyStopping(monitor="val_loss"),
            # ModelCheckpoint(
            #     monitor="matthews_correlation",
            # ),
            LearningRateMonitor(logging_interval="step"),
            Timer(),
        ],
        profiler="simple",
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    trainer.validate(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
