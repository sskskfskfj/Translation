from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint에서 모델과 토크나이저 로드
# facebook/m2m100_418m
checkpoint_original_path = "facebook/m2m100_418m"
checkpoint_path = "model/m2m100_418m/checkpoint-3145"
model = M2M100ForConditionalGeneration.from_pretrained(checkpoint_path, local_files_only=True).to(device)
tokenizer = M2M100Tokenizer.from_pretrained(checkpoint_path, local_files_only=True)

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    tokenized = tokenizer(text, return_tensors="pt")
    translated = model.generate(**tokenized.to(device), forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    return tokenizer.decode(translated[0], skip_special_tokens=True)


if __name__ == "__main__":
    en_text = """
    The dominant sequence transduction models are based on complex recurrent or
    convolutional neural networks that include an encoder and a decoder. The best
    performing models also connect the encoder and decoder through an attention
    mechanism. We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and convolutions
    entirely.     
    """

    ko_text = translate(en_text, src_lang="en", tgt_lang="ko")

    print(ko_text)