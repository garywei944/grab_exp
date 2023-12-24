import os
from dataclasses import dataclass
from attrs import define

import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer, LightningDataModule


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


@define
class BoringDataModule(LightningDataModule):
    batch_size: int = 2

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        print("attrs post init")
        self.save_hyperparameters()

        self.test = "hahah"

        print(self.hparams)

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        # weights_summary=None,
    )
    trainer.fit(model, datamodule=BoringDataModule())


if __name__ == "__main__":
    run()
