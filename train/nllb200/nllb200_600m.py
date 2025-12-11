from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from data.huggingface_parallel_data import HuggingfaceParallelData

import torch
import wandb
import os
import dotenv
import numpy as np
import sacrebleu


dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"


class NLLB200Trainer:
    def __init__(self, model_name, output_dir):
        wandb.init(project="translation", name=model_name)
        self.src_lang = "eng_Latn"
        self.tgt_lang = "kor_Hang"
        
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN).to(device)
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            src_lang=self.src_lang, 
            tgt_lang=self.tgt_lang, 
            token=HUGGINGFACE_TOKEN
        )
        self.model.config.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        self.dataset = HuggingfaceParallelData(model_name=model_name)
        self.output_dir = output_dir


    def train(self):
        dataset_dict = self.dataset.preprocess_dataset()
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['validation']

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model, 
            padding=True,
            label_pad_token_id=-100
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=3,

            metric_for_best_model="eval_loss",
            greater_is_better=True,
            load_best_model_at_end=True,
            bf16=True,
            overwrite_output_dir=True,

            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,

            max_grad_norm=0.8,
            learning_rate=5e-5,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,

            report_to="wandb",
            logging_steps=100,
            run_name=self.model_name
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


    # def test(self):
    #     test_dataset = self.dataset.preprocess_dataset()['test']
    #     pass


if __name__ == "__main__":
    trainer = NLLB200Trainer(model_name="facebook/nllb-200-distilled-600M", output_dir="model/nllb200_600m")
    trainer.train()