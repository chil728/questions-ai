import os
import argparse
import datetime
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
import transfomers

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,   
)

from datasets import load_dataset
from huggingface_hub import snapshot_download

import wandb
from dotenv import load_dotenv

from config import (
    TrainingConfig,
    ModelConfig,
    DataConfig
)

def is_local_path(path) -> bool:
    pass

def download_model(model_name):
    pass

def initialize_tokenizer(model_name):
    pass

def process_datasets(train_ds, test_ds, tokenizer, data_config, accelerator):
    pass

def get_fsdp_config(gradient_checkpointing):
    pass

def get_training_configurations():
    pass

def save_model():
    pass

def initialize_wandb(accelerator, args, wb_token):
    pass

def setup_model_environment(accelerator, model_name, hf_token):
    pass

def create_training_args(args, wb_token, fsdp_configurations, is_distributed, batch_size):
    pass

def main(args):
    
    load_dotenv()
    training_config = TrainingConfig()
    model_config = ModelConfig()
    data_config = DataConfig()

    gradient_accumulation_steps = args.grad_accum_steps
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        
    )