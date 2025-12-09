from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from huggingface_hub import HfApi, login

import os
import dotenv


dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_WRITE_TOKEN")

# Hugging Face에 로그인
login(token=HUGGINGFACE_TOKEN)

# 업로드할 checkpoint 경로
checkpoint_path = "model/m2m100_418m/checkpoint-3145"

# Hub에 업로드할 모델 이름 (사용자명/모델명 형식)
# 예: "your-username/m2m100-418m-en-ko" 또는 "your-username/m2m100-418m-checkpoint-3145"
hub_model_id = "sskskfskfj/test-m2m-checkpoint"  # 여기를 실제 사용자명으로 변경하세요

print(f"Checkpoint를 로드하는 중: {checkpoint_path}")
model = M2M100ForConditionalGeneration.from_pretrained(checkpoint_path, local_files_only=True)
tokenizer = M2M100Tokenizer.from_pretrained(checkpoint_path, local_files_only=True)

print(f"Hub에 업로드하는 중: {hub_model_id}")
model.push_to_hub(
    hub_model_id,
    token=HUGGINGFACE_TOKEN,
    commit_message="Upload fine-tuned M2M100 418M checkpoint for en-ko translation"
)

tokenizer.push_to_hub(
    hub_model_id,
    token=HUGGINGFACE_TOKEN,
    commit_message="Upload tokenizer for fine-tuned M2M100 418M"
)


