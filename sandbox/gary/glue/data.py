from transformers import (
    AutoTokenizer,
    BatchEncoding,
    default_data_collator,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    AutoConfig,
    T5ForConditionalGeneration,
)

import datasets

from attrs import define, field
from functools import cached_property

GLUE_TASK_TO_KEYS = {
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "stsb": ("sentence1", "sentence2"),
    # "wnli": ("sentence1", "sentence2"),
}


@define()
class GLUETaskDataset:
    task_name = field(type=str)
    keys = field(type=tuple[str])
    labels = field(type=list[str])

    @cached_property
    def num_labels(self):
        return len(self.labels)

    @cached_property
    def is_regression(self):
        return self.task_name == "stsb"


def main():
    # config = AutoConfig.from_pretrained(
    #     "google/t5-v1_1-small",
    #     num_labels=2,
    #     # finetuning_task="cola",
    # )
    DATASETS = {}
    for task_name in GLUE_TASK_TO_KEYS.keys():
        dataset = datasets.load_dataset("glue", task_name)
        if task_name != "stsb":
            label_list = dataset["train"].features["label"].names
        else:
            label_list = ["label"]
        DATASETS[task_name] = GLUETaskDataset(
            task_name=task_name,
            keys=GLUE_TASK_TO_KEYS[task_name],
            labels=label_list,
        )

    print(DATASETS)
    print(DATASETS["cola"])


if __name__ == "__main__":
    main()
