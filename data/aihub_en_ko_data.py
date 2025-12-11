from datasets import load_dataset
from transformers import AutoTokenizer
import os
import dotenv


dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class AihubEnKoData:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            src_lang="eng_Latn", 
            tgt_lang="kor_Hang", 
            token=HUGGINGFACE_TOKEN
        )


    def load_aihub_en_ko_data(self):
        dataset = load_dataset("sskskfskfj/aihub_en_ko_dataset", token=HUGGINGFACE_TOKEN)
        return dataset["train"]


    """train 90%, test 5%, validation 5%"""
    def preprocess_aihub_en_ko_data(self):
        dataset = self.load_aihub_en_ko_data()
        dataset = dataset.train_test_split(test_size=0.1)
        test_valid_dataset = dataset["test"].train_test_split(test_size=0.5, seed=42)

        return dataset["train"]
        # train_dataset = dataset["train"].map(self.tokenize_function, batched=True, remove_columns=dataset["train"].column_names, num_proc=4)
        # test_dataset = test_valid_dataset["train"].map(self.tokenize_function, batched=True, remove_columns=test_valid_dataset["train"].column_names, num_proc=4)
        # validation_dataset = test_valid_dataset["test"].map(self.tokenize_function, batched=True, remove_columns=test_valid_dataset["test"].column_names, num_proc=4)

        # return train_dataset, validation_dataset, test_dataset


    def tokenize_function(self, examples):
        tokenized_src = self.tokenizer(examples["en"], padding="max_length", truncation=True, max_length=128)
        tokenized_tgt = self.tokenizer(examples["ko"], padding="max_length", truncation=True, max_length=128)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels = tokenized_tgt["input_ids"].copy()
        labels = [[-100 if token == pad_token_id else token for token in label] for label in labels]

        return {
            "input_ids": tokenized_src["input_ids"],
            "attention_mask": tokenized_src["attention_mask"],
            "labels": labels
        }


if __name__ == "__main__":
    aihub_en_ko_data = AihubEnKoData()
    # train_dataset, test_dataset, validation_dataset = aihub_en_ko_data.preprocess_aihub_en_ko_data()
    tokenizer = aihub_en_ko_data.tokenizer
    dataset = aihub_en_ko_data.preprocess_aihub_en_ko_data()

    print(dataset["en"][0])