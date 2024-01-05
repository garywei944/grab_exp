import lightning as L
from torch.utils.data import DataLoader, Dataset, Sampler


class CVDataModule(L.LightningDataModule):
    dims: tuple[int, ...]
    num_classes: int
