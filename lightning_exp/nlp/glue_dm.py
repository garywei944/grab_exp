import lightning as L
import datasets
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    BatchEncoding,
    default_data_collator,
    DataCollatorWithPadding,
    HfArgumentParser,
)
from torch.utils.data import DataLoader

from attrs import define, field
import os

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


# class GLUEDataModule(L.LightningDataModule):
#     tokenizer: PreTrainedTokenizerBase
#     num_labels: int
#     dataset: DatasetDict
#     train_dataset: Dataset
#     eval_dataset: Dataset
#
#     def __init__(
#         self,
#         model_name_or_path: str = "",
#         task_name: str = "mrpc",
#         pad_to_max_length: bool = True,
#         max_seq_length: int = 128,
#         train_batch_size: int = 32,
#         eval_batch_size: int = 32,
#         num_workers: int = os.cpu_count(),
#         use_fast_tokenizer: bool = True,
#         use_fp16: bool = False,
#     ):
#         super().__init__()
#
#         self.model_name_or_path = model_name_or_path
#         self.task_name = task_name
#         self.pad_to_max_length = pad_to_max_length
#         self.max_seq_length = max_seq_length
#         self.train_batch_size = train_batch_size
#         self.eval_batch_size = eval_batch_size
#         self.num_workers = num_workers
#         self.use_fast_tokenizer = use_fast_tokenizer
#         self.use_fp16 = use_fp16
#
#         self.save_hyperparameters()
#
#         self.text_keys = TASK_TO_KEYS[self.task_name]
#         self.is_regression = self.task_name == "stsb"
#
#         if self.pad_to_max_length:
#             self.data_collator = default_data_collator
#         else:
#             self.data_collator = DataCollatorWithPadding(
#                 self.tokenizer, pad_to_multiple_of=8 if self.use_fp16 else None
#             )


@define
class GLUEDataModule(L.LightningDataModule):
    model_name_or_path: str
    task_name: str = "mrpc"
    max_seq_length: int = 128
    pad_to_max_length: bool = True
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = os.cpu_count()
    use_fast_tokenizer: bool = True
    use_fp16: bool = False

    tokenizer: PreTrainedTokenizerBase = field(init=False)
    dataset: DatasetDict = field(init=False)
    train_dataset: Dataset = field(init=False)
    eval_dataset: Dataset = field(init=False)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.text_keys = GLUE_TASK_TO_KEYS[self.task_name]
        self.is_regression = self.task_name == "stsb"
        self.num_labels = (
            1 if self.is_regression else GLUE_TASK_NUM_LABELS[self.task_name]
        )

        if self.pad_to_max_length:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8 if self.use_fp16 else None
            )

    def prepare_data(self) -> None:
        AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=self.use_fast_tokenizer
        )
        dataset = datasets.load_dataset("glue", self.task_name)
        dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=dataset["train"].column_names,
        )

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=self.use_fast_tokenizer
        )
        self.dataset = datasets.load_dataset("glue", self.task_name)

        # Hopefully Huggingface cache won't recompute this
        self.dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
            load_from_cache_file=True,
        )

        self.train_dataset = self.dataset["train"]
        self.eval_dataset = self.dataset[
            "validation_matched" if self.task_name == "mnli" else "validation"
        ]

    def preprocess_function(self, examples: dict[str, list]) -> BatchEncoding:
        # Either encode single sentence or sentence pairs
        if len(self.text_keys) > 1:
            texts = list(
                zip(
                    examples[self.text_keys[0]],
                    examples[self.text_keys[1]],
                )
            )
        else:
            texts = examples[self.text_keys[0]]

        padding = "max_length" if self.pad_to_max_length else False
        result = self.tokenizer.batch_encode_plus(
            *texts,
            max_length=self.max_seq_length,
            padding=padding,
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

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    obj = GLUEDataModule(model_name_or_path="bert-base-uncased", task_name="mrpc")

    # from datargs import parse, make_parser
    #
    # parser = make_parser(GLUEDataModule)
    # args = parser.parse_args()
    #
    # print(args)

    parser = HfArgumentParser((GLUEDataModule,))
    (args,) = parser.parse_args_into_dataclasses()

    print(args)
