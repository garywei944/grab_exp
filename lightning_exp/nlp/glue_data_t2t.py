from typing import Any

import lightning as L
import datasets
import torch
import numpy as np
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    default_data_collator,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    # T5ForConditionalGeneration,
)
from torch.utils.data import DataLoader

# from attrs import define, field
# from functools import cached_property
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

GLUE_TASK_TO_LABELS = {
    "cola": ["unacceptable", "acceptable"],
    "mnli": ["entailment", "neutral", "contradiction"],
    "mrpc": ["not_equivalent", "equivalent"],
    "qnli": ["entailment", "not_entailment"],
    "qqp": ["not_duplicate", "duplicate"],
    "rte": ["entailment", "not_entailment"],
    "sst2": ["negative", "positive"],
    "stsb": [None],
    "wnli": ["not_entailment", "entailment"],
}

TASK_NAMES = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
TASK_VAL_NAMES = [
    "cola",
    "mnli",
    "mnli_mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]


class GLUET2TEDataModule(L.LightningDataModule):
    train_dataset: Dataset
    val_datasets: Dataset | list[Dataset]

    def __init__(
        self,
        model_name_or_path: str,
        pad_to_max_length: bool = False,
        max_length: int = 128,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        num_workers: int = 1,
        use_fast_tokenizer: bool = True,
        use_fp16: bool = False,
        data_path: str = "data/processed",
        load_from_disk: bool = True,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=self.use_fast_tokenizer, legacy=False
        )

        if self.pad_to_max_length:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8
            )

    def prepare_data(self) -> None:
        try:
            if not self.load_from_disk:
                raise FileNotFoundError
            datasets.load_from_disk(str(self.path / "train"))
            for task_name in TASK_VAL_NAMES:
                datasets.load_from_disk(str(self.path / task_name))
            print(f"Loaded processed data from {self.path}")
        except FileNotFoundError:
            # Process train dataset
            train_dataset = make_glue_train_dataset(
                self.tokenizer,
                self.max_length,
                self.pad_to_max_length,
                self.num_workers,
            )

            train_dataset.save_to_disk(self.path / "train")

            for task_name in TASK_VAL_NAMES:
                if task_name == "mnli":
                    dataset = datasets.load_dataset("glue", task_name)[
                        "validation_matched"
                    ]
                elif task_name == "mnli_mm":
                    dataset = datasets.load_dataset("glue", "mnli")[
                        "validation_mismatched"
                    ]
                else:
                    dataset = datasets.load_dataset("glue", task_name)["validation"]
                dataset = dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.num_workers,
                    remove_columns=dataset.column_names,
                    load_from_cache_file=False,
                    fn_kwargs={
                        "task_name": task_name.split("_")[0],  # Hack only for mnli_mm
                        "tokenizer": self.tokenizer,
                        "max_length": self.max_length,
                        "pad_to_max_length": self.pad_to_max_length,
                    },
                )
                dataset.save_to_disk(self.path / task_name)

            print(f"Saved processed data to {self.path}")

    def setup(self, stage: str = None) -> None:
        self.train_dataset = datasets.load_from_disk(self.path / "train")
        self.val_datasets = [
            datasets.load_from_disk(self.path / task_name)
            for task_name in TASK_VAL_NAMES
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            # num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return [
            DataLoader(
                e,
                batch_size=self.eval_batch_size,
                collate_fn=self.data_collator,
                # num_workers=self.num_workers,
            )
            for e in self.val_datasets
        ]


def preprocess_function(
    examples: dict[str, Any],
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    pad_to_max_length: bool,
) -> BatchEncoding:
    """
    Prepare GLUE exactly follows Google T5 practice that add special tokens
    before trimming and padding.
    """
    task_keys = GLUE_TASK_TO_KEYS[task_name]
    task_labels = GLUE_TASK_TO_LABELS[task_name]

    texts = []
    for i in range(len(examples[task_keys[0]])):
        str2join = [task_name]
        for k in task_keys:
            str2join.extend((f"{k}:", examples[k][i]))
        texts.append(" ".join(str2join))

    padding = "max_length" if pad_to_max_length else False
    results = tokenizer(
        texts,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )

    is_regression = task_name == "stsb"

    if not is_regression:
        label = [task_labels[i] for i in examples["label"]]
    else:
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/f1f16c0d77a7c3a6a21b5cf9f3e96adf26a71c60/t5/data/preprocessors.py#L853
        # tf.as_string(tf.round(x['label'] * 5) / 5, precision=1)
        label = [str(np.round(e * 5) / 5) for e in examples["label"]]

    targets = tokenizer(
        label,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )

    results["labels"] = targets["input_ids"]

    # for k, v in targets.items():
    #     results[f"decoder_{k}"] = v

    # # Default behavior to pad label with -100 if pad to max length
    # if pad_to_max_length:
    #     labels = np.array(results["decoder_input_ids"])
    #     labels[labels == tokenizer.pad_token_id] = -100
    #     results["labels"] = labels.tolist()
    # else:
    #     results["labels"] = results["decoder_input_ids"]

    return results


def make_glue_train_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    pad_to_max_length: bool,
    num_workers: int = 1,
) -> Dataset:
    ds = []
    # for task_name in TASK_NAMES:
    for task_name in ["cola", "stsb"]:

        def func(examples: dict[str, Any]) -> BatchEncoding:
            return preprocess_function(
                examples,
                task_name,
                tokenizer,
                max_length,
                pad_to_max_length,
            )

        dataset = datasets.load_dataset("glue", task_name)["train"]
        dataset = dataset.map(
            func,
            batched=True,
            num_proc=num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )
        ds.append(dataset)
        # for i in range(1):
        #     tokens = tokenizer.convert_ids_to_tokens(
        #         dataset["train"][i]["decoder_input_ids"]
        #     )
        #     s = tokenizer.decode(dataset["train"][i]["decoder_input_ids"])
        #     # s = tokenizer.decode(dataset["train"][i]["labels"])
        #
        #     print(s)

    return datasets.concatenate_datasets(ds)


if __name__ == "__main__":
    from cd2root import cd2root

    cd2root()
    # obj = GLUET2TEDataModule(
    #     model_name_or_path="google/t5-v1_1-small", task_name="mrpc"
    # )
    #
    # obj.prepare_data()
    #
    # print(obj.hparams)
    # DATASETS = {}
    # names = {}
    # # for task_name in GLUE_TASK_TO_KEYS.keys():
    # for task_name in ['cola', 'stsb']:
    #     dataset = datasets.load_dataset("glue", task_name)
    #     if task_name != "stsb":
    #         label_list = dataset["train"].features["label"].names
    #     else:
    #         label_list = ["label"]
    #
    #     names[task_name] = label_list
    #
    # print(names)
    #     DATASETS[task_name] = GLUEDataConfig(
    #         task_name=task_name,
    #         keys=GLUE_TASK_TO_KEYS[task_name],
    #         labels=label_list,
    # #     )

    # @define
    # class C:
    #     x = field(type=list[int], factory=lambda: [20])
    #     # x = field(default=[20])
    #
    # i = C()
    # k = C()
    # i.x.append(42)
    # print(k.x)
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

    # make_combined_datasets()

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "google/t5-v1_1-small", use_fast=True, legacy=False
    # )
    # ds = make_glue_train_dataset(tokenizer, 128, True)
    #
    # # cola = ds["cola"]
    #
    # model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
    #
    # # model.eval()
    # model.train()
    # model.to("cuda")
    #
    # inputs = ds[:3]
    # inputs = {k: torch.tensor(v, device="cuda") for k, v in inputs.items()}
    #
    # result = model(**inputs)
    #
    # print(result)
    #
    # pass

    dm = GLUET2TEDataModule(
        "google/t5-v1_1-small",
        pad_to_max_length=True,
        max_length=128,
        train_batch_size=128,
        eval_batch_size=1024,
        num_workers=4,
        use_fast_tokenizer=True,
        use_fp16=False,
        data_path="data/processed",
        load_from_disk=False,
    )

    dm.prepare_data()
    dm.setup()

    print(dm.hparams)
    loader = dm.val_dataloader()
    for batch in loader:
        print(batch)
        break
