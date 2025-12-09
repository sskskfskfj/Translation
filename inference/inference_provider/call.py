from huggingface_hub import InferenceClient

import os
import dotenv


dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
client = InferenceClient(model = model_name, token = HF_TOKEN)
prompt = "인공지능이 뭐야"

completion = client.chat_completion(
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please respond in Korean. Keep your responses concise and within the token limit of 128 tokens."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens = 256
)

print(completion.choices[0].message.content)