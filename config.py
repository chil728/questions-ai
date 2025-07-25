class TrainingConfig:
    def __init__(self):
        self.lora_r = 8    # 4 ~ 32
        self.lora_alpha = 16
        self.lora_dropout = 0.1    # 0.1 ~ 0.3
        self.lora_target_modules = "all-linear"
        
        self.torch_dtype = "bf16"
        self.gradient_checkpointing = True
        self.use_flash_attntion = True
        self.seed = 666
        
        self.bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "llm_int8_skip_modules": None,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False
        }
        
        self.gpu_memory_threshold = 8 * 1024 * 1024 * 1024
        self.large_gpu_batch_size = 16
        
        self.max_memory = {0: "20GB"}
        self.offload_folder = "offload"
        
        self.optimizer_config = {
            "lr": 2e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01
        }
        
class ModelConfig:
    def __init__(self):
        self.cache_dir = './.cache'
        self.temp_model_dir = './model'
        self.model_max_shard_size = "2GB"
        
        self.safe_serializaion = True
        self.save_strategy = "no"
        self.save_merged_model = False
        
class DataConfig:
    def __init__(self):
        self.max_seq_length = 2048
        self.pad_to_max_length = True
        self.truncation = True