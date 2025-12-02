from datasets import load_dataset, Dataset, DatasetDict
from transformers import M2M100Tokenizer

import pandas as pd


class HuggingfaceParallelData:
    def __init__(
        self, 
        model_name: str,
        dataset_name="lemon-mint/korean_english_parallel_wiki_augmented_v1"
    ):
        self.dataset = load_dataset(dataset_name, split="train")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)


    def preprocess_dataset(self) -> DatasetDict:
        df = self.dataset.to_pandas()
        filtered_df = df[df['score'] >= 0.9].reset_index(drop=True)
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
        self.tokenizer.src_lang = "en"
        tokenized_src = self.tokenizer(
            examples['english'], 
            padding='max_length', 
            truncation=True, 
            max_length=512
        )
        
        self.tokenizer.tgt_lang = "ko"
        tokenized_tgt = self.tokenizer(
            examples['korean'], 
            padding='max_length', 
            truncation=True, 
            max_length=512
        )
        
        # labels에서 padding token을 -100으로 변환 (loss 계산 시 무시됨)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels = tokenized_tgt['input_ids'].copy()
        labels = [[-100 if token == pad_token_id else token for token in label] for label in labels]
        
        return {
            'input_ids': tokenized_src['input_ids'],
            'attention_mask': tokenized_src['attention_mask'],
            'labels': labels
        }


if __name__ == "__main__":
    hf_parallel_data = HuggingfaceParallelData(model_name="facebook/m2m100_1.2b")

    dataset = hf_parallel_data.preprocess_dataset()
    print(dataset["train"][0])