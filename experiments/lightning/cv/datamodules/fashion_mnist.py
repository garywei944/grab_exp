from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from .cv_datamodule import CVDataModule


class FashionMNISTDataModule(CVDataModule):
    train_dataset: Dataset
    test_dataset: Dataset

    def __init__(
        self,
        data_dir: str = "data/external",
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
        num_workers: int = 1,
        shuffle: bool = True,
        sampler: Sampler = None,
        data_augmentation: str = "none",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.sampler = sampler

        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=0.28604060411453247,
                    std=0.3530242443084717,
                ),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = FashionMNIST(
            self.data_dir, train=True, transform=self.transform
        )
        self.test_dataset = FashionMNIST(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
