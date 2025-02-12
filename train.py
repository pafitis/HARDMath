import torch
import torch.nn as nn
import torch.nn.functional as F

from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

import re
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

from utls.rl_utils import formating_reward_wrapper, correctness_reward_wrapper

import wandb

def main():

    ## Config
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)

    global_params = config.get('global_params')
    lora_params = config.get('lora_params')
    grpo_params = config.get('grpo_params')

    ## Authentication
    with open('configs/keys.yaml') as f:
        keys = yaml.safe_load(f)
    
    ## Weights and biases
    wandb.login = keys.get('wandb_key')
    run = wandb.init(
        project='Llama31B_instruct_GRPO',
        job_type='training',
        anonymous='allow',
    )

    ## Lora Config
    peft_config = LoraConfig(
        r=lora_params.get('r'),
        lora_alpha=lora_params.get('lora_alpha'),
        lora_dropout=lora_params.get('lora_dropout'),
        target_modules=lora_params.get('target_modules'),
        init_lora_weights=lora_params.get('init_lora_weights'),
        task_type=lora_params.get('task_type'),
        bias=lora_params.get('bias'),
    )

    ## GRPO Config
    grpo_config = GRPOConfig(
        output_dir=grpo_params.get('output_dir'),
        
        temperature=grpo_params.get('temperature'),
        learning_rate=grpo_params.get('learning_rate'),
        beta=grpo_params.get('beta'),

        num_train_epochs=grpo_params.get('num_train_epochs'),
        num_generations=grpo_params.get('num_generations'),
        per_device_train_batch_size=grpo_params.get('per_device_train_batch_size'),
        
        gradient_accumulation_steps=grpo_params.get('gradient_accumulation_steps'),
        gradient_checkpointing=grpo_params.get('client_checkpointing'),

        logging_steps=grpo_params.get('logging_steps'),
        
        push_to_hub=grpo_params.get('huggingface_model_dir'),
        push_to_hub_token=keys.get('huggingface_key'),
        bf16=grpo_params.get('bf16'),

        report_to='wandb', # TODO setup weights and biases
        
        # TODO: Check usage of vllm
        # use_vllm=grpo_params.get('use_vllm'),
    )

    ## Load model, tokenizer, data
    tokenizer = AutoTokenizer.from_pretrained(
        global_params.get('model_id'),
        padding_side='left',
        device="auto",
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        global_params.get('model_id'),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token = keys.get('huggingface_key'),
        quantization_config=bnb_config,
        device="auto"
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    training_dataset = load_dataset('pafitis/HARDMath_processed_validation')

    ## Set up GRPO Trainer
    GRPO_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[formating_reward_wrapper, correctness_reward_wrapper],
        train_dataset= training_dataset, 
        args=grpo_config,
        peft_config=peft_config,
    )

    wandb.finish()

if __name__ == '__main__':
    # TODO: Write main wrapper
    main()