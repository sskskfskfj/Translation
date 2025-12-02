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
import os
import dotenv
from sacrebleu import BLEU
from rouge_score import rouge_scorer


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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # model.generate()를 사용하는 compute_metrics 생성
        compute_metrics_fn = create_compute_metrics(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            eval_dataset=eval_dataset
        )
        
        # model=None으로 설정 - 모델이 labels로부터 decoder_input_ids를 자동 생성하도록 함
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,
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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_fn,
        )
        
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


def create_compute_metrics(model, tokenizer, device, eval_dataset):
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # labels에서 -100 (padding)을 제외하고 디코딩
        decoded_labels = []
        for label in labels:
            label_tokens = [token for token in label if token != -100]
            decoded_label = tokenizer.decode(label_tokens, skip_special_tokens=True)
            decoded_labels.append(decoded_label)
        
        # eval_dataset에서 input_ids를 가져와서 model.generate() 사용
        decoded_preds = []
        batch_size = 8  # 생성 시 배치 크기
        
        model.eval()
        tokenizer.src_lang = "en"
        tokenizer.tgt_lang = "ko"
        
        with torch.no_grad():
            for i in range(0, len(eval_dataset), batch_size):
                end_idx = min(i + batch_size, len(eval_dataset))
                batch_indices = list(range(i, end_idx))
                batch = eval_dataset.select(batch_indices)
                
                # 배치 데이터 준비
                input_ids_list = batch['input_ids']
                attention_mask_list = batch['attention_mask']
                
                # 텐서로 변환
                input_ids = torch.tensor(input_ids_list).to(device)
                attention_mask = torch.tensor(attention_mask_list).to(device)
                
                # model.generate()를 사용하여 번역 생성
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    num_beams=4,
                    forced_bos_token_id=tokenizer.get_lang_id("ko"),
                    do_sample=False
                )
                
                # 생성된 텍스트 디코딩
                for gen_id in generated_ids:
                    decoded_pred = tokenizer.decode(gen_id, skip_special_tokens=True)
                    decoded_preds.append(decoded_pred)
        
        model.train()
        
        # BLEU 계산
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
        if num_samples > 0:
            rouge_scores['rouge1'] /= num_samples
            rouge_scores['rouge2'] /= num_samples
            rouge_scores['rougeL'] /= num_samples
        
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