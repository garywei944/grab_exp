import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataModule,
    LanguageModelingTransformer,
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
model = LanguageModelingTransformer(pretrained_model_name_or_path="gpt2")
dm = LanguageModelingDataModule(
    batch_size=16,
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    tokenizer=tokenizer,
    num_workers=32,
)
trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

trainer.fit(model, dm)
