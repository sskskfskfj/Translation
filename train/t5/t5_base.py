from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from data.huggingface_parallel_data import HuggingfaceParallelData

import os
import dotenv
import wandb

dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


class T5Trainer:
    def __init__(self, model_name, output_dir):
        wandb.init(project="translation", name=model_name)
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
        self.model.config.use_cache = False
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True, token=HUGGINGFACE_TOKEN)
        self.output_dir = output_dir
        self.dataset = HuggingfaceParallelData(model_name = self.model_name)

    def train(self):
        dataset = self.dataset.preprocess_dataset()
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
        test_dataset = dataset['test']

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model, 
            padding="longest",
            label_pad_token_id=-100
        )

       
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_total_limit=5,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="linear",
            bf16=True,
            logging_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb",
            run_name="mt5_en_ko"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":
    data = HuggingfaceParallelData(model_name = "google/mt5-base")
    dataset = data.preprocess_dataset()
    print(len(dataset["train"]))
    # trainer = T5Trainer(model_name="google/mt5-base", output_dir="./model/mt5_en_ko")
    # trainer.train()