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
import numpy as np
from sacrebleu import BLEU
from rouge_score import rouge_scorer


model_name = "facebook/m2m100_1.2b"

# 전역 tokenizer 변수 (compute_metrics에서 사용)
_tokenizer = None


def compute_metrics(eval_pred):
    global _tokenizer
    
    predictions, labels = eval_pred
    
    # predictions는 logits 형태이므로 argmax로 토큰 ID로 변환
    predictions = np.argmax(predictions, axis=-1)
    
    # labels에서 -100 (padding)을 제외
    decoded_preds = []
    decoded_labels = []
    
    for pred, label in zip(predictions, labels):
        # padding과 -100 제거
        pred_tokens = [token for token in pred if token != _tokenizer.pad_token_id]
        label_tokens = [token for token in label if token != -100]
        
        # 디코딩
        decoded_pred = _tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_label = _tokenizer.decode(label_tokens, skip_special_tokens=True)
        
        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)
    
    # BLEU 계산 (sacrebleu는 references를 list of lists로 받음)
    bleu = BLEU()
    references = [[label] for label in decoded_labels]
    bleu_score = bleu.corpus_score(decoded_preds, references).score
    
    # ROUGE 계산
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge.score(label, pred)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    
    # 평균 계산
    num_samples = len(decoded_preds)
    rouge_scores['rouge1'] /= num_samples
    rouge_scores['rouge2'] /= num_samples
    rouge_scores['rougeL'] /= num_samples
    
    return {
        "bleu": float(bleu_score),
        "rouge1": float(rouge_scores['rouge1']),
        "rouge2": float(rouge_scores['rouge2']),
        "rougeL": float(rouge_scores['rougeL'])
    }


class M2MTrainer:
    def __init__(self, model_name, output_dir):
        wandb.init(project="translation", name=model_name)
        self.model_name = model_name
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.dataset = HuggingfaceParallelData(model_name=model_name)
        self.output_dir = output_dir


    def train(self):
        global _tokenizer
        _tokenizer = self.tokenizer
        
        dataset_dict = self.dataset.preprocess_dataset()
        
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['validation']
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            label_smoothing_factor=0.1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            max_grad_norm=0.8,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            weight_decay=0.02,
            logging_steps=50,
            logging_first_step=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            report_to="wandb",
            run_name=self.model_name
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
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":
    trainer = M2MTrainer(model_name=model_name, output_dir="./model/m2m100_1b")
    trainer.train()
