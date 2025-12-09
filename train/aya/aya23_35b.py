from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import os
import dotenv


dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_WRITE_TOKEN = os.getenv("HUGGINGFACE_WRITE_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "CohereLabs/aya-23-35B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.bfloat16,
).to(device)


def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=512)

    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


if __name__ == "__main__":

    prompt = f"""
    Translate the following English text into Korean.
    Output only the translated Korean text, with no explanations, no repetition, and no additional commentary.

    English text:
    """
    print(generate_text(prompt))