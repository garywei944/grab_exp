from typing import Any

import lightning as L
import datasets
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    default_data_collator,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader

from pathlib import Path
from dict_hash import sha256

import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


GLUE_TASK_TO_KEYS = {
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_TASK_NUM_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "stsb": 1,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "ax": 3,
}

GLUE_COLUMNS = [
    "datasets_idx",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "start_positions",
    "end_positions",
    "labels",
]


class GLUEDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        pad_to_max_length: bool = True,
        max_length: int = 128,
        train_batch_size: int = 128,
        eval_batch_size: int = 1024,
        num_workers: int = 1,
        use_fast_tokenizer: bool = True,
        use_fp16: bool = False,
        data_path: str = "data/processed",
        load_from_disk: bool = True,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.pad_to_max_length = pad_to_max_length
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.use_fast_tokenizer = use_fast_tokenizer
        self.use_fp16 = use_fp16
        self.load_from_disk = load_from_disk

        self.save_hyperparameters(ignore=["num_workers", "data_path", "load_from_disk"])

        self.id = sha256(self.hparams)  # important to load cached processed data
        self.path = Path(data_path) / "glue" / self.id

        self.text_keys = GLUE_TASK_TO_KEYS[self.task_name]
        self.num_labels = GLUE_TASK_NUM_LABELS[self.task_name]
        self.is_regression = self.task_name == "stsb"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=self.use_fast_tokenizer
        )

        if self.pad_to_max_length:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8 if self.use_fp16 else None
            )

    def prepare_data(self) -> None:
        try:
            if not self.load_from_disk:
                raise FileNotFoundError
            datasets.load_from_disk(self.path)
        except FileNotFoundError:
            dataset = datasets.load_dataset("glue", self.task_name)
            dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                # num_proc=self.num_workers,
                remove_columns=dataset["train"].column_names,
            )

            dataset.save_to_disk(self.path)

    def setup(self, stage: str = None) -> None:
        dataset: DatasetDict = datasets.load_from_disk(self.path)

        self.train_dataset = dataset["train"]

        if self.task_name == "mnli":
            self.val_dataset = [
                dataset["validation_matched"],
                dataset["validation_mismatched"],
            ]
        else:
            self.val_dataset = dataset["validation"]

    def preprocess_function(self, examples: dict[str, Any]) -> BatchEncoding:
        # Either encode single sentence or sentence pairs
        if len(self.text_keys) > 1:
            texts = (
                examples[self.text_keys[0]],
                examples[self.text_keys[1]],
            )
        else:
            texts = (examples[self.text_keys[0]],)

        padding = "max_length" if self.pad_to_max_length else False
        result = self.tokenizer(
            *texts,
            padding=padding,
            max_length=self.max_length,
            truncation=True,
        )

        if "label" in examples:
            result["labels"] = examples["label"]

        return result

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if self.task_name == "mnli":
            return [
                DataLoader(
                    e,
                    batch_size=self.eval_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.num_workers,
                )
                for e in self.val_dataset
            ]
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.num_workers,
            )


if __name__ == "__main__":
    obj = GLUEDataModule(model_name_or_path="bert-base-uncased", task_name="mrpc")

    print(obj.hparams)

    # from datargs import parse, make_parser
    #
    # parser = make_parser(GLUEDataModule)
    # args = parser.parse_args()
    #
    # print(args)

    # parser = HfArgumentParser((GLUEDataModule,))
    # (args,) = parser.parse_args_into_dataclasses()
    #
    # print(args)
