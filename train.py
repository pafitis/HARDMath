import torch
import torch.nn as nn
import torch.nn.functional as F

from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

import re
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.answer_extraction import extract_final_answer_allform, extract_boxed_content, extract_final_answer

from Levenshtein import distance
import numpy as np

## Config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

global_params = config.get('global_params')
lora_params = config.get('lora_params')
sft_params = config.get('sft_params')
rl_params = config.get('rl_params')

## Authentication
with open('configs/keys.yaml') as f:
    keys = yaml.safe_load(f)

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
# TODO ADD KEYARG INPUT FOR REWARD FUNC! WONT WORK OTHERWISE
def correctness_reward(prediction, ground_truth):
    # Extract the contents between <answer> ... </answer>
    start_point = prediction.find('<answer>') + len('<answer>')
    end_point = prediction.find('</answer>')
    final_prediction = prediction[start_point:end_point]

    # Will use Levenshtein distance (edit distance)
    # counts number of additions/removals/edits required
    # to match the desired string

    lev_distance = distance(final_prediction, ground_truth) # symmetric, order doesn't matter
    len_maxstring = max(len(final_prediction), len(ground_truth))

    # TODO:
    # I've observed that we should probably modify the distance
    # We should probably have 0 reward if required modifications are more than 4-5
    # and then normalise based on that?
    return 1.0 - (lev_distance / len_maxstring)

def formating_reward(prediction, eos_token = '<|eot_id|>'):
    '''
    The desired format should contain two main blocks
        <think> ... </think>
        \n
        <answer> $$\boxed{...}$$ </answer>

    Thoughts:
        1. The final answer should use \\boxed LaTeX
        2. The final answer should be enclosed within <answer> </answer>    
        3. There should be no \\boxed string within <think> </think>
        4. EOS should appear after </answer>

    Note:
        Compared re, .find, `in` approaches 
        `in` appears fastest, however, `in` requires at least 3 checks which in turn
        makes it slower than a single regex search

        Missing <> should be captured, as well as wrong order
        No extra chars before or after
    '''
    # Checks general formatting
    # Check if starts with <think> and ends with </answer>
    
    # Checks that \boxed does not appear in <think> </think>
        # (?:(?!\\boxed\{).)*
    # Checks that $$\boxed{...} appears in <answer> </answer>
    # incl. whitepsaces to match HARDMath format
        # \$\$\\boxed.*\}\$\$
    # Checks that thinking and answering parts are sep by new line
        # </think>\\n<answer>
    pattern = (
        r'^<think>(?:(?!\\boxed\{).)*</think>\n<answer>\$\$ \\boxed.*\} \$\$</answer>$'
    )
    return rl_params.get('reward_format') if re.match(pattern, prediction, re.DOTALL) else 0.0

def total_reward(prediction):
    correctness = correctness_reward(prediction)
    formating = formating_reward(prediction)

    return correctness + formating

## Model
class RLModel(nn.Module):

    def __init__(self):
        super(RLModel, self).__init__()

    def forward():
        pass

def prepare_sample(sample):
    question = sample.question
    answer = sample.extracted_answer

## RL
def reinforce(
        model,
        optimizer,
        input_data,
        training_epochs,
        max_seq_length,
):
    
    history = []

    for epoch in training_epochs:
        print(f'Progess: {np.round(100 * (epoch+1) / training_epochs, 2)}')

        for sample in input_data:
            current_question = sample['question']
            current_answer = sample['extracted_answer']

            training_args = GRPOConfig(
                output_dir='llama31b_instruct_GRPO',
                logging_steps=10,
                use_vllm=True, # TODO check if this causes issues
            )

            trainer = GRPOTrainer(
                model=global_params.get('model_id'),
                reward_funcs=[formating_reward, correctness_reward],
                argrs=training_args,
                train_dataset=
            )