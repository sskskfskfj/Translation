from transformers import (
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
from data.huggingface_parallel_data import HuggingfaceParallelData

import torch
import wandb


model_name = "facebook/m2m100_1.2b"


class M2MTrainer:
    def __init__(self, model_name):
        wandb.init(project="translation", name="m2m100_1b")
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.dataset = HuggingfaceParallelData()
        self.output_dir = "./model/m2m100_1b"


    def train(self):
        dataset_dict = self.dataset.preprocess_dataset()
        
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['validation']
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            max_grad_norm=1.0,
            learning_rate=3e-5,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            report_to="wandb",
            run_name="m2m100_1b"
        )

        model = self.model
        model.config.use_cache = False

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model(self.output_dir)


if __name__ == "__main__":
    trainer = M2MTrainer(model_name=model_name)
    trainer.train()
