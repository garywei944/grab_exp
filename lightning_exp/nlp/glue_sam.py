import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    Timer,
    TQDMProgressBar,
)
from transformers.optimization import Adafactor

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
            return loss

        loss = self(**batch).loss
        self.manual_backward(loss)
        opt.step(closure=closure)
        opt.zero_grad()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return SAM(
            self.model.parameters(),
            Adafactor,
            rho=self.rho,
            lr=self.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )


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
