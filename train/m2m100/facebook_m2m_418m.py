from transformers import (
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
from data.huggingface_parallel_data import HuggingfaceParallelData
from sacrebleu import BLEU
from rouge_score import rouge_scorer

import torch
import wandb
import os
import dotenv
import numpy as np


dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


class M2MTrainer:
    def __init__(self, model_name, output_dir):
        wandb.init(project="translation", name=model_name)
        self.model_name = model_name
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
        self.model.config.use_cache = False
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

        self.dataset = HuggingfaceParallelData(model_name=model_name)
        self.output_dir = output_dir


    def train(self):
        dataset_dict = self.dataset.preprocess_dataset()
        
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['validation']
        

        compute_metrics = create_compute_metrics(self.tokenizer)
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
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

        trainer = Trainer(
            model=self.model,
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


def create_compute_metrics(tokenizer):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # logits → predicted ids
        if predictions.ndim == 3:
            preds = predictions.argmax(-1)
        else:
            preds = predictions
        
        # numpy array를 리스트로 변환하고 -100 제거
        pred_str = []
        for p in preds:
            # -100이 아닌 토큰만 필터링
            if isinstance(p, np.ndarray):
                p = p[p != -100]
            else:
                p = [token for token in p if token != -100]
            
            decoded = tokenizer.decode(p, skip_special_tokens=True)
            pred_str.append(decoded if decoded.strip() else " ")  # 빈 문자열 방지
        
        label_str = []
        for l in labels:
            # -100이 아닌 토큰만 필터링
            if isinstance(l, np.ndarray):
                l = l[l != -100]
            else:
                l = [token for token in l if token != -100]
            
            decoded = tokenizer.decode(l, skip_special_tokens=True)
            label_str.append(decoded if decoded.strip() else " ")  # 빈 문자열 방지
        
        # 샘플 수 확인
        n = len(pred_str)
        if n == 0:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }
        
        # BLEU 계산
        bleu = BLEU()
        references = [[l] for l in label_str]
        bleu_score = bleu.corpus_score(pred_str, references).score
        
        # ROUGE 계산
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        for pred, label in zip(pred_str, label_str):
            try:
                r = scorer.score(label, pred)
                rouge_scores['rouge1'] += r['rouge1'].fmeasure
                rouge_scores['rouge2'] += r['rouge2'].fmeasure
                rouge_scores['rougeL'] += r['rougeL'].fmeasure
            except Exception:
                # 빈 문자열 등으로 인한 오류 무시
                continue
        
        # 평균 계산
        for k in rouge_scores:
            rouge_scores[k] /= n
        
        return {
            "bleu": float(bleu_score),
            "rouge1": float(rouge_scores['rouge1']),
            "rouge2": float(rouge_scores['rouge2']),
            "rougeL": float(rouge_scores['rougeL'])
        }
    
    return compute_metrics



if __name__ == "__main__":
    model_name = "facebook/m2m100_418M"
    trainer = M2MTrainer(model_name=model_name, output_dir="./model/m2m100_418m")
    trainer.train()