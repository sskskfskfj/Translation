from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

import pandas as pd
import os
import dotenv

dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class HuggingfaceParallelData:
    def __init__(
        self, 
        model_name: str,
        dataset_name="lemon-mint/korean_english_parallel_wiki_augmented_v1"
    ):
        self.dataset = load_dataset(dataset_name, split="train", token=HUGGINGFACE_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN, src_lang="eng_Latn", tgt_lang="kor_Hang")


    def preprocess_dataset(self) -> DatasetDict:
        df = self.dataset.to_pandas()
        filtered_df = df[df['score'] >= 0.88].reset_index(drop=True)
        filtered_df = filtered_df[filtered_df["english"].str.len() <= 1200].reset_index(drop=True)
        filtered_df.drop(columns = ["score"], axis=1, inplace=True)
        
        expanded_rows = []
        for _, row in filtered_df.iterrows():
            ko_text = str(row.get('korean', ''))
            en_text = str(row.get('english', ''))
            
            ko_splits = [part.strip() for part in ko_text.split('\n\n') if part.strip()]
            en_splits = [part.strip() for part in en_text.split('\n\n') if part.strip()]
            
            if len(ko_splits) != len(en_splits):
                continue
            
            for i in range(len(ko_splits)):
                min_length = 100
                if len(en_splits[i]) < min_length:
                    continue
                
                new_row = row.copy()
                if 'korean' in row:
                    new_row['korean'] = ko_splits[i]
                
                if 'english' in row:
                    new_row['english'] = en_splits[i]

                expanded_rows.append(new_row)
        
        expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        filtered_dataset = Dataset.from_pandas(expanded_df)
        
        train_test_split = filtered_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        temp_dataset = train_test_split['test']
        
        validation_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
        validation_dataset = validation_test_split['train']
        test_dataset = validation_test_split['test']
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset
        })

        tokenized_datasets = {}
        for split_key in dataset_dict.keys():
            tokenized_datasets[split_key] = dataset_dict[split_key].map(
                self.tokenize_function, 
                batched=True,
                remove_columns=dataset_dict[split_key].column_names,
                num_proc=4
            )

        return DatasetDict(tokenized_datasets)


    def tokenize_function(self, examples):
        tokenized_src = self.tokenizer(
            examples['english'], 
            padding=False, 
            truncation=True, 
            max_length=256
        )
        
        tokenized_tgt = self.tokenizer(
            examples['korean'], 
            padding=False, 
            truncation=True, 
            max_length=256
        )

        labels = tokenized_tgt['input_ids'].copy()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels = [[-100 if token == pad_token_id else token for token in label] for label in labels]
        
        return {
            'input_ids': tokenized_src['input_ids'],
            'attention_mask': tokenized_src['attention_mask'],
            'labels': labels 
        }


if __name__ == "__main__":
    hf_parallel_data = HuggingfaceParallelData(model_name="facebook/nllb-200-distilled-600M")
    dataset = hf_parallel_data.preprocess_dataset()
    print(dataset)

    # df = dataset["train"].to_pandas()   
    # sample_1100 = [text for text in df[df["english"].str.len() >= 1100]["english"]]
    # print(f"length of sample_1100: {len(sample_1100)}")

    # for sample in sample_1100:
    #     tokenized_length = hf_parallel_data.tokenizer(sample, return_tensors="pt")["input_ids"].size(1)
    #     if tokenized_length > 256:     
    #         print(tokenized_length)