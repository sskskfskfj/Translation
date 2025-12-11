# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "model/nllb200_600m/checkpoint-6972"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang="eng_Latn", tgt_lang="kor_Hang")
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

if __name__ == "__main__":
    prompt = """
    The dominant sequence transduction models are based on complex recurrent or
    convolutional neural networks that include an encoder and a decoder. The best
    performing models also connect the encoder and decoder through an attention
    mechanism. We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and convolutions
    entirely.     
    """
    print(len(prompt))
    print(len(tokenizer.encode(prompt)))


    forced_bos_token_id = tokenizer.convert_tokens_to_ids("kor_Hang")
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**input_ids, max_length=256, forced_bos_token_id=forced_bos_token_id)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))