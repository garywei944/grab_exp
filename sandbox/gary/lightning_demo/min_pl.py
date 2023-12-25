import lightning as L

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from transformers import HfArgumentParser

from attrs import define, field

# from attr import dataclass
from dataclasses import dataclass


@dataclass
class MNISTDataModuleArgs:
    a: int = 1000
    data_dir: str = "~/projects/grab_exp/data/external"


@dataclass
class Args:
    a: int = 1000


# MNIST lightning datamodule
class MNISTDataModule(L.LightningDataModule, MNISTDataModuleArgs):
    def __init__(self, args: MNISTDataModuleArgs):
        super().__init__()
        MNISTDataModuleArgs.__init__(self, **vars(args))

        self.save_hyperparameters(vars(args), ignore=["data_dir"])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # transforms
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # split dataset
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=32)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=32)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=32)
        return mnist_test


def main():
    args = MNISTDataModuleArgs(a=12345)
    dm = MNISTDataModule(args)

    print(dm.a)
    print(dm.hparams)

    # from datargs import make_parser
    #
    # parser = make_parser(MNISTDataModule)
    # args = parser.parse_args()

    # parser = HfArgumentParser((MNISTDataModuleArgs, Args))
    # margs, args = parser.parse_args_into_dataclasses()
    #
    # print(margs)
    print(args)


if __name__ == "__main__":
    main()
