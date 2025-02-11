import torch
import torch.nn as nn
import torch.nn.functional as F

from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM

## Config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
global_params = config.get('global_params')
lora_params = config.get('lora_params')
sft_params = config.get('sft_params')
rl_params = config.get('rl_params')

with open('configs/keys.yaml') as f:
    keys = yaml.safe_load(f)

# SFT Training
## Lora Config
peft_config = LoraConfig(
    r=lora_params.get('r'),
    lora_alpha=lora_params.get('lora_alpha'),
    lora_dropout=lora_params.get('lora_dropout'),
    target_modules=lora_params.get('target_modules'),
    task_type=lora_params.get('task_type'),
    init_lora_weights=lora_params.get('init_lora_weights'),
    bias=lora_params.get('bias'),
)

## Load model
tokenizer = AutoTokenizer.from_pretrained(global_params.get('model_id'))
model = AutoModelForCausalLM.from_pretrained(
    global_params.get('model_id'),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token = keys.get('huggingface_key')
)

tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id


# RL Training
## Reward functions
def correctness_reward(prediction):
    pass

def format_reward(prediction):
    pass

def total_reward(prediction):
    correctness = correctness_reward(prediction)
    formating = format_reward(prediction)

    return correctness + formating

## Model
class RLModel(nn.Module):

    def __init__(self):
        super(RLModel, self).__init__()

    def forward():
        pass




## RL
def reinforce(
        model,
        optimizer,
        input_data,
        training_epochs=1000,
        max_seq_length=2048,
):

