from transformers import T5Tokenizer, T5ForConditionalGeneration

import os
import dotenv
import torch


dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "model/mt5_en_ko/checkpoint-1572"

model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, token=HUGGINGFACE_TOKEN).to(device)
tokenizer = T5Tokenizer.from_pretrained(checkpoint_path, token=HUGGINGFACE_TOKEN)


if __name__ == "__main__":
    prompt = """
    The dominant sequence transduction models are based on complex recurrent or
    convolutional neural networks that include an encoder and a decoder. The best
    performing models also connect the encoder and decoder through an attention
    mechanism. We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and convolutions
    entirely.     
    """

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids.to(device), max_length=256)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))