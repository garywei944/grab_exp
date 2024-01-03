import lightning as L


class CVDataModule(L.LightningDataModule):
    dims: tuple[int, ...]
    num_classes: int
