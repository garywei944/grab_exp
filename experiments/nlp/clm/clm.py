#!/usr/bin/env python
# coding: utf-8
import math
import os
import sys
from functools import partial
from itertools import chain
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from absl import logging

import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    CONFIG_MAPPING,
    AutoModelForCausalLM,
)
from transformers.training_args import OptimizerNames
from datasets import load_dataset
from accelerate import Accelerator

import torchopt
from torch.func import grad, grad_and_value, vmap, functional_call

from grablib import GraBSampler, BalanceType
from grablib.utils import EventTimer, pretty_time

# Change pwd to the project root directory
PROJECT_NAME = "GraB-lib"
PROJECT_PATH = Path(__file__).resolve()
while PROJECT_PATH.name != PROJECT_NAME:
    PROJECT_PATH = PROJECT_PATH.parent
os.chdir(PROJECT_PATH)
sys.path.insert(0, str(PROJECT_PATH))

from experiments.utils.func_helpers import make_func_params
from experiments.utils.arguments import GraBArguments

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: str | None = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: str | None = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    torch_dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        # if self.config_overrides is not None and (
        #     self.config_name is not None or self.model_name_or_path is not None
        # ):
        #     raise ValueError(
        #         "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
        #     )
        ...


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str | None = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: str | None = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: str | None = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: str | None = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: int | None = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


@torch.no_grad()
def train(
    train_loader,
    sampler,
    model,
    params,
    buffers,
    ft_compute_sample_grad_and_loss,
    optimizer,
    opt_state,
    completed_steps,
    progress_bar,
    checkpoint_path,
    checkpointing_steps,
    max_steps,
    device="cuda",
):
    # TODO: no resume or accumulate gradient
    losses = []
    for step, batch in enumerate(train_loader):
        # batch = {k: v.to(device) for k, v in batch.items()}
        ft_per_sample_grads, batch_loss = ft_compute_sample_grad_and_loss(
            params, buffers, dict(batch)
        )
        sampler.step(ft_per_sample_grads)
        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}
        updates, opt_state = optimizer.update(
            grads, opt_state, params=params
        )  # get updates
        params = torchopt.apply_updates(params, updates)  # update model parameters
        losses.append(batch_loss.float())

        progress_bar.update(1)
        completed_steps += 1

        # checkpointing
        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                torch.save(opt_state, checkpoint_path / f"{output_dir}_opt_state.pt")
                torch.save(
                    model.state_dict(), checkpoint_path / f"{output_dir}_model.pt"
                )
        if completed_steps >= max_steps:
            break

    return torch.cat(losses).mean(), completed_steps


@torch.no_grad()
def validate(eval_loader, model, device="cuda", disable_tqdm=False):
    losses = []
    for step, batch in tqdm(
        enumerate(eval_loader),
        total=len(eval_loader),
        desc="eval",
        leave=True,
        disable=disable_tqdm,
    ):
        # for step, batch in enumerate(eval_loader):
        # batch = {k: v.to(device) for k, v in batch.items()}
        for k, v in batch.items():
            b = v.size(0)
            break
        outputs = model(**batch)
        loss = outputs.loss
        losses.append(loss.repeat(b))
    losses = torch.cat(losses)

    eval_loss = losses.mean()
    try:
        perplexity = torch.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return eval_loss, perplexity


def main():
    # Parse the arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, GraBArguments)
    )
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    grab_args: GraBArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, grab_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            grab_args,
        ) = parser.parse_args_into_dataclasses()

    assert (
        training_args.gradient_accumulation_steps == 1
    ), "Please set gradient_accumulation_steps to 1."

    # Set up wandb
    model_name = (
        model_args.model_name_or_path.split("/")[-1]
        if model_args.model_name_or_path
        else model_args.config_name.split("/")[-1]
    )
    wandb.init(
        project=f"grab-{model_name}-{data_args.dataset_name}"
        if grab_args.wandb_project is None
        else grab_args.wandb_project,
        entity="grab",
        mode="online" if grab_args.use_wandb else "offline",
        config={
            **vars(model_args),
            **vars(data_args),
            **vars(training_args),
            **vars(grab_args),
        },
    )

    # Set up exp_id and checkpoint path
    exp_id = get_exp_id(model_args, data_args, training_args, grab_args, model_name)
    checkpoint_path = Path(training_args.output_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.set_verbosity(logging.INFO)
    device = training_args.device
    timer = EventTimer(device=device)
    accelerator = Accelerator()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        if model_args.config_overrides is not None:
            logging.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logging.info(f"New config: {config}")
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logging.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logging.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logging.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logging.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )

    model.to(device=device, dtype=dtype)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logging.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logging.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    train_dataset = train_dataset.with_format("torch")

    eval_dataset = lm_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    eval_dataset = eval_dataset.with_format("torch")

    # Gary: start to convert to functorch and train with GraB

    params, buffers = make_func_params(model)

    # Initiate sampler
    d = sum(p[1].numel() for p in model.named_parameters())
    logging.info(f"Number of training examples: n = {len(train_dataset):,}")
    logging.info(f"Number of parameters: d = {d:,}")

    # Load orders for FixedOrdering
    orders = None
    if grab_args.order_path is not None:
        orders = torch.load(grab_args.order_path)
        if len(orders.shape) == 2:
            orders = orders[-1].tolist()
        else:
            orders = orders.tolist()

    sampler = GraBSampler(
        train_dataset,
        params,
        batch_size=training_args.train_batch_size,
        # Random projection
        seed=training_args.seed,  # Only used for generating random projection
        # Probabilistic balance
        orders=orders,
        # Other specific
        dtype=dtype,
        device=device,
        timer=timer,
        record_herding=grab_args.record_grads,
        stale_mean_herding=False,
        cuda_herding=not grab_args.cpu_herding,
        record_norm=grab_args.record_grads,
        **vars(grab_args),
    )

    # Initiate data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=training_args.train_batch_size,
        sampler=sampler,
        persistent_workers=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=training_args.eval_batch_size,
        persistent_workers=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=training_args.eval_batch_size,
        persistent_workers=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )
    epochs = int(training_args.num_train_epochs)
    max_steps = epochs * len(train_loader)
    if training_args.max_steps > 0:
        max_steps = min(max_steps, training_args.max_steps)
    if epochs * len(train_loader) > max_steps:
        # TODO: Actually we don't know if any data is dropped by the loader
        epochs = math.ceil(max_steps / len(train_loader))
    warmup_steps = training_args.get_warmup_steps(max_steps)

    no_decay = ["bias", "LayerNorm.weight"]

    if training_args.optim in [OptimizerNames.ADAMW_HF, OptimizerNames.ADAMW_TORCH]:
        optimizer = torchopt.adamw(
            # lr=torchopt.schedule.linear_schedule(
            #     init_value=training_args.learning_rate,
            #     end_value=0.0,
            #     transition_steps=max_steps - warmup_steps,
            #     transition_begin=warmup_steps,
            # ),
            lr=get_lr_scheduler(
                training_args.learning_rate,
                warmup_steps,
                max_steps,
                training_args.learning_rate / 10,
            ),
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            weight_decay=training_args.weight_decay,
            mask={
                k: not any(nd in k for nd in no_decay)
                for k, p in model.named_parameters()
            },
            use_accelerated_op=True,
        )
    else:
        raise ValueError(f"Optimizer {training_args.optim} not supported.")

    model, train_loader, train_eval_loader, test_loader = accelerator.prepare(
        model, train_loader, train_eval_loader, test_loader
    )

    opt_state = optimizer.init(params)

    ft_compute_sample_grad_and_loss = get_func(model, next(iter(train_dataset)))

    # Record the norms
    df_grad_norms = pd.DataFrame()

    # loop over the dataloader multiple times
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {epochs}")
    logging.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logging.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.train_batch_size}"
    )
    logging.info(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    logging.info(f"  Total optimization steps = {max_steps}")
    progress_bar = tqdm(
        range(max_steps),
        disable=training_args.disable_tqdm,
        desc="Training",
        leave=True,
    )
    completed_steps = 0
    save_steps = (
        training_args.save_steps if training_args.save_strategy == "steps" else None
    )
    for epoch in range(0 if training_args.logging_first_step else 1, epochs + 1):
        logs = {
            "epoch": epoch,
            "step": epoch * len(train_dataset),
        }

        # Only evaluate before the first epoch
        if epoch != 0:
            # compute orders if we are using ntk balance
            assert (
                grab_args.balance_type != BalanceType.NTK_EIGEN.value
            ), "NTK balance is not supported for NLP tasks."

            with timer(f"train"):
                # perform training (single loop over the train dataloader)
                train_loss, completed_steps = train(
                    train_loader=train_loader,
                    sampler=sampler,
                    model=model,
                    params=params,
                    buffers=buffers,
                    ft_compute_sample_grad_and_loss=ft_compute_sample_grad_and_loss,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    completed_steps=completed_steps,
                    progress_bar=progress_bar,
                    checkpoint_path=checkpoint_path,
                    checkpointing_steps=save_steps,
                    max_steps=max_steps,
                    device=device,
                )

            # For Adaptive Mean Balance that need m and v from optimizer
            try:
                v = torch.cat([p.reshape(-1) for p in opt_state[0].nu])
                t = completed_steps
                v.div_(1 - training_args.adam_beta2**t)

                sampler.sorter.update_v(v)

                logging.info("-" * 20)
                logging.info(sampler.sorter.v)

                del v, t
            except (AttributeError, TypeError):
                pass
            logs.update(
                {
                    "train_loss": train_loss,
                    "train_time": timer["train"][-1],
                }
            )

            if grab_args.record_grads:
                grad_norms = sampler.sorter.grad_norms

                # Save the norms
                norm_mean = np.mean(grad_norms)
                norm_std = np.std(grad_norms)
                norm_max = np.max(grad_norms)
                df_grad_norms[epoch] = grad_norms

                herding = sampler.sorter.herding
                avg_grad_error = sampler.sorter.avg_grad_error

                # Update only after the first epoch
                logs.update(
                    {
                        "norm_mean": norm_mean,
                        "norm_std": norm_std,
                        "norm_max": norm_max,
                        "herding": herding,
                        "avg_grad_error": avg_grad_error,
                    }
                )
        with timer("val"):
            train_eval_loss, train_eval_ppl = validate(
                train_eval_loader, model, disable_tqdm=training_args.disable_tqdm
            )
            # perform validation (single loop over the validation dataloader)
            val_loss, val_ppl = validate(
                test_loader, model, disable_tqdm=training_args.disable_tqdm
            )

        logs.update(
            {
                "train_eval_loss": train_eval_loss,
                "train_eval_perplexity": val_ppl,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "val_time": timer["val"][-1],
            }
        )

        if epoch == 0:
            print(
                f"Before training | "
                f"train_eval loss: {train_eval_loss :.3f} "
                f"train_eval ppl: {train_eval_ppl:.3f} | "
                f"val loss: {val_loss :.3f} "
                f"val ppl: {val_ppl:.3f} | "
                f'val: {pretty_time(timer["val"][-1])}'
            )
        else:
            log_msg = (
                f"Epoch: {epoch} | "
                f"train loss: {train_loss :.3f} | "
                f"train_eval loss: {train_eval_loss :.3f} "
                f"ppl: {train_eval_ppl:.3f} | "
                f"val loss: {val_loss :.3f} "
                f"ppl: {val_ppl:.3f} | "
                f'train: {pretty_time(timer["train"][-1])} '
                f'val: {pretty_time(timer["val"][-1])}'
            )
            if grab_args.record_grads:
                log_msg += (
                    f" | norm_mean: {norm_mean:.2f} "
                    f"norm_std: {norm_std:.2f} "
                    f"norm_max: {norm_max:.2f} | "
                    f"herding: {herding:.2f} "
                    f"avg_grad_error: {avg_grad_error:.2f}"
                )
            print(log_msg)

        # save checkpoint
        if training_args.save_strategy == "epoch":
            checkpoint_name = exp_id + f"_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path / checkpoint_name)

        if epoch > 0:
            if grab_args.record_grads:
                # Save the grad norms
                df_grad_norms.describe().to_csv(
                    checkpoint_path / f"{exp_id}_{epochs}_grad_norms_proc.csv"
                )
            # Save the timer
            timer.save(checkpoint_path / f"{exp_id}_{epochs}_timer_proc.pt")
            timer.summary().to_csv(
                checkpoint_path / f"{exp_id}_{epochs}_timer_proc.csv"
            )

        # Save the orders
        if grab_args.record_orders:
            torch.save(
                torch.tensor(sampler.orders_history),
                checkpoint_path / f"{exp_id}_{epochs}_orders_proc.pt",
            )
        wandb.log(logs)

    print(torch.cuda.memory_summary())

    if grab_args.record_grads:
        print("-" * 50)
        print(df_grad_norms.describe())

    print("-" * 50)
    print("Timer:")
    print(timer.summary())
    peak_memory_allocated = (
        torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 / 1024
    )
    wandb.log(
        {
            "peak_gpu_mem": peak_memory_allocated,
            "total_train_time": sum(timer["train"]),
            "total_val_time": sum(timer["val"]),
        }
    )
    # wandb.finish()

    # Save the timer
    timer.save(checkpoint_path / f"{exp_id}_{epochs}_timer.pt")
    timer.summary().to_csv(checkpoint_path / f"{exp_id}_{epochs}_timer.csv")
    if grab_args.record_grads:
        # Save the grad norms
        df_grad_norms.describe().to_csv(
            checkpoint_path / f"{exp_id}_{epochs}_grad_norms.csv"
        )

    if grab_args.record_orders:
        # Save the orders
        torch.save(
            torch.tensor(sampler.orders_history),
            checkpoint_path / f"{exp_id}_{epochs}_orders.pt",
        )


# pure function
def compute_loss(model, params, buffers, kwargs):
    # vamp remove the first dimension, which is batch size, but bert requires it
    kwargs = {k: v.unsqueeze(0) for k, v in kwargs.items()}

    # Gary: BERT doesn't treat input_ids as named kwargs, so we need to
    # explicitly pass it in
    input_ids = kwargs["input_ids"]
    kwargs.pop("input_ids")
    out = functional_call(
        module=model,
        parameter_and_buffer_dicts=(params, buffers),
        args=(input_ids,),
        kwargs=kwargs,
    )

    return out.loss


def get_lr_scheduler(learning_rate, num_warmup_steps, num_training_steps, min_lr):
    def func(it: int):
        # 1) linear warmup for warmup_iters steps
        if it < num_warmup_steps:
            return learning_rate * it / num_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > num_training_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        # assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    return func


def get_func(model, batch: dict):
    return vmap(
        grad_and_value(partial(compute_loss, model)),
        in_dims=(None, None, {k: 0 for k in batch.keys()}),
        randomness="different",
    )  # the only argument of compute_loss is batched along the first axis


def get_exp_id(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    grab_args: GraBArguments,
    model_name: str,
):
    # Unique experiment name for checkpoints
    exp_id = (
        f"{data_args.dataset_name}_{model_name}_{grab_args.balance_type}"
        f"_{training_args.optim.value}_lr_{training_args.learning_rate}"
        f"_wd_{training_args.weight_decay}"
        f"_b_{training_args.train_batch_size}_seed_{training_args.seed}"
    )

    if grab_args.normalize_grads:
        exp_id += "_norm"
    if grab_args.random_projection:
        exp_id += f"_pi_{grab_args.random_projection_eps}"
    if grab_args.prob_balance:
        exp_id += f"_prob_{grab_args.prob_balance_c:.1f}"
    if grab_args.balance_type in [
        BalanceType.RECURSIVE_BALANCE,
        BalanceType.RECURSIVE_PAIR_BALANCE,
    ]:
        exp_id += f"_depth_{grab_args.depth}"
    if not grab_args.random_first_epoch:
        exp_id += "_no_rr"
    if grab_args.balance_type == BalanceType.EMA_BALANCE:
        exp_id += f"_ema_{grab_args.ema_decay}"

    return exp_id


if __name__ == "__main__":
    main()
