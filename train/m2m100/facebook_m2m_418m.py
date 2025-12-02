from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from train.m2m100.facebook_m2m_1b import M2MTrainer

import torch
import wandb


model_name = "facebook/m2m100_418m"


if __name__ == "__main__":
    trainer = M2MTrainer(model_name=model_name, output_dir="./model/m2m100_418m")
    trainer.train()