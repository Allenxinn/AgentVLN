from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class VLNTrainConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-2B"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2" 
    torch_dtype: str = "bfloat16" 
    vln_data_roots: List[str] = field(default_factory=list) 
    
    llava_video_data_root: Optional[str] = None
    llava_video_json_path: Optional[str] = None
    
    max_seq_length: int = 4096
    image_min_pixels: int = 256 * 28 * 28
    image_max_pixels: int = 1280 * 28 * 28

    freeze_llm: bool = False
    freeze_vision_encoder: bool = True 
    freeze_projector: bool = False
    llm_lr: float = 2e-5
    vision_encoder_lr: float = 2e-6
    projector_lr: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine" 
    warmup_ratio: float = 0.03
    warmup_steps: int = 0 
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    deepspeed: Optional[str] = None  
    local_rank: int = -1
    report_to: str = "wandb" 
    run_name: Optional[str] = None  
    
    max_history_frames: int = 8
    max_action_lookahead: int = 3
    distance_threshold: float = 3.0
    use_action_symbols: bool = False
    prefix_trainable: bool = True
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self):
        if self.bf16 and self.fp16:
            raise ValueError("Cannot enable both bf16 and fp16.")
