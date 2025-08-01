"""
# 安装PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装最新版Transformers
pip install git+https://github.com/huggingface/transformers

# 安装其他依赖
pip install -U bitsandbytes accelerate peft datasets trl scipy einops sentencepiece

PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
CUDA_LAUNCH_BLOCKING = "1"
"""

import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import Trainer
import time
import accelerate  # 导入accelerate库

# 设置环境变量优化CUDA内存
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
torch.backends.cudnn.allow_tf32 = True

# 打印accelerate版本以便调试
print(f"Accelerate version: {accelerate.__version__}")

# 2. 量化配置 (4-bit精度)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 3. 加载模型 - 使用支持qwen3架构的模型名称
try:
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True  # 需要信任远程代码
    )
    print("成功加载DeepSeek-R1-0528-Qwen3-8B模型")
except Exception as e:
    print(f"加载DeepSeek-R1-0528-Qwen3-8B失败: {e}")
    print("改用较小的替代模型: deepseek-ai/deepseek-llm-1.3b")
    model_name = "deepseek-ai/deepseek-llm-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

# 确保填充token正确设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. 准备模型进行k位训练
model = prepare_model_for_kbit_training(model)

# 5. LoRA配置
lora_config = LoraConfig(
    r=16,              # 增加秩以提升模型质量
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 增加目标模块
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 显示可训练参数占比

# 6. 准备数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# 改进的数据预处理函数 - 动态填充
def preprocess_function(examples):
    # 截断文本并确保是字符串
    texts = [str(text) for text in examples["text"]]
    
    # 使用tokenizer进行标记化 - 使用动态填充
    tokenized = tokenizer(
        texts,
        max_length=512,  # 增加最大长度
        truncation=True,
        padding="longest",  # 使用动态填充
        return_tensors="pt"
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

# 应用预处理
dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=64,  # 增加批处理大小
    remove_columns=["text"]  # 移除原始文本列
)

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.1)  # 10%作为验证集
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 7. 训练参数配置
save_steps = 500
eval_steps = save_steps  # 设置评估步数等于保存步数

# 根据accelerate版本调整参数
training_args_kwargs = {
    "output_dir": "./results",
    "per_device_train_batch_size": 2,   # 减少批大小以适应8GB VRAM
    "per_device_eval_batch_size": 2,    # 添加评估批大小
    "gradient_accumulation_steps": 8,    # 增加梯度累积步数
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "fp16": False,                       # 使用bf16
    "bf16": True,                        # RTX 40系列支持bf16
    "save_steps": save_steps,
    "logging_steps": 50,                 # 更频繁地记录日志
    "optim": "paged_adamw_8bit",         # 防止内存溢出
    "report_to": "none",                 # 禁用wandb
    "max_grad_norm": 0.3,
    "gradient_checkpointing": True,      # 启用梯度检查点节省显存
    "warmup_ratio": 0.05,                # 预热比例
    "lr_scheduler_type": "cosine",       # 余弦学习率衰减
    "logging_dir": "./logs",
    "remove_unused_columns": True,
    "dataloader_num_workers": 0,         # Windows下必须设置为0
    "tf32": True,                        # 启用TF32加速
    "group_by_length": True,             # 按长度分组提高效率
    "eval_strategy": "steps",            # 添加评估策略以监控进度
    "eval_steps": eval_steps,            # 评估步数等于保存步数
    "load_best_model_at_end": True,      # 训练结束时加载最佳模型
    "metric_for_best_model": "loss",     # 使用损失作为评估指标
    "greater_is_better": False           # 损失越低越好
}

# 对于新版本的transformers，参数名已更改
if hasattr(TrainingArguments, "evaluation_strategy"):
    training_args_kwargs["evaluation_strategy"] = training_args_kwargs.pop("eval_strategy")
    training_args_kwargs["eval_steps"] = training_args_kwargs.pop("eval_steps")

training_args = TrainingArguments(**training_args_kwargs)

# 8. 创建自定义数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 用于因果语言建模
)

# 11. 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 添加评估数据集
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 12. 开始训练
try:
    # 训练前检查点
    checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-initial")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"初始模型已保存到: {checkpoint_dir}")
    
    # 开始训练
    print("开始训练...")
    train_result = trainer.train()
    
    # 保存最终模型
    trainer.save_model()
    print(f"训练完成! 最终模型已保存到: {training_args.output_dir}")
    
    # 记录训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n内存不足错误! 尝试以下解决方案:")
        print("1. 减少 per_device_train_batch_size")
        print("2. 增加 gradient_accumulation_steps")
        
        # 自动调整设置
        training_args.per_device_train_batch_size = 1
        training_args.gradient_accumulation_steps = 16
        
        print(f"调整后设置: batch_size={training_args.per_device_train_batch_size}, "
              f"accumulation_steps={training_args.gradient_accumulation_steps}")
        
        # 重新创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        print("使用调整后的设置重新开始训练...")
        trainer.train()
    else:
        # 保存崩溃时的模型状态
        crash_dir = os.path.join(training_args.output_dir, "crash-recovery")
        model.save_pretrained(crash_dir)
        tokenizer.save_pretrained(crash_dir)
        print(f"训练崩溃! 模型状态已保存到: {crash_dir}")
        raise e

# 13. 保存适配器权重
adapter_path = "lora_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA适配器已保存到: {adapter_path}")

# 14. 合并LoRA适配器到基础模型
from peft import PeftModel

print("开始合并LoRA适配器到基础模型...")

# 重新加载基础模型（不量化）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True  # 如果使用DeepSeek-R1需要此参数
)

# 合并适配器
merged_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = merged_model.merge_and_unload()

# 保存完整模型
full_model_path = "full_model"
merged_model.save_pretrained(full_model_path, safe_serialization=True)
tokenizer.save_pretrained(full_model_path)
print(f"完整模型已保存到: {full_model_path}")

# 15. 推理测试
def generate_response(prompt, max_new_tokens=200):
    # 加载完整模型
    model = AutoModelForCausalLM.from_pretrained(
        full_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(full_model_path)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # 确保设置正确的填充token
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试推理
prompt = input("輸入你的問題：")

response = generate_response(prompt)