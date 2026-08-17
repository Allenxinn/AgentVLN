import os
os.environ["HF_ALLOW_VULNERABLE_TORCH_LOAD"] = "1"

import sys
import json
import logging
import argparse
from dataclasses import fields
from typing import Optional

import torch
import transformers
import transformers.utils.import_utils

if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers, "trainer") and hasattr(transformers.trainer, "check_torch_load_is_safe"):
    transformers.trainer.check_torch_load_is_safe = lambda: None

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    set_seed,
)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agentvln.datasets.vln_nav_dataset import VLNNavDataset
from agentvln.datasets.nav_dataset_config import NavDatasetConfig
from agentvln.datasets.vln_nav_collator import VLNDataCollator
from agentvln.train.train_config import VLNTrainConfig
from agentvln.train.trainer import VLNTrainer
from torch.utils.data import ConcatDataset
from agentvln.datasets.llava_video_dataset import LLaVAVideoDataset

logger = logging.getLogger(__name__)


def parse_args() -> VLNTrainConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL on VLN navigation data."
    )

    config = VLNTrainConfig()
    for f in fields(VLNTrainConfig):
        field_type = f.type
        default_val = getattr(config, f.name)

        if field_type == "bool" or field_type is bool:
            parser.add_argument(
                f"--{f.name}",
                type=lambda x: x.lower() in ("true", "1", "yes"),
                default=default_val,
                help=f"Default: {default_val}",
            )
        elif "Optional" in str(field_type):
            parser.add_argument(
                f"--{f.name}",
                type=str,
                default=default_val,
                help=f"Default: {default_val}",
            )
        elif f.name == "vln_data_roots":
            parser.add_argument(
                f"--{f.name}",
                nargs="+",
                default=[],
                help="List of VLN dataset root directories.",
            )
        else:
            actual_type = type(default_val) if default_val is not None else str
            parser.add_argument(
                f"--{f.name}",
                type=actual_type,
                default=default_val,
                help=f"Default: {default_val}",
            )

    args = parser.parse_args()
    config_dict = {}
    for f in fields(VLNTrainConfig):
        if hasattr(args, f.name):
            config_dict[f.name] = getattr(args, f.name)

    return VLNTrainConfig(**config_dict)


def setup_logging(local_rank: int):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if local_rank in (-1, 0) else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def setup_model_and_processor(config: VLNTrainConfig):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    logger.info(f"Loading model: {config.model_name_or_path}")
    logger.info(f"  torch_dtype: {config.torch_dtype}")
    logger.info(f"  attn_implementation: {config.attn_implementation}")
    processor = AutoProcessor.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        min_pixels=config.image_min_pixels,
        max_pixels=config.image_max_pixels,
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )
    
    _apply_freeze(model, config)
    
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
        
    _log_model_stats(model)

    return model, processor


def _apply_freeze(model, config: VLNTrainConfig):
    if config.freeze_vision_encoder:
        logger.info("Freezing vision encoder (model.visual.*)...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if "visual" in name and "merger" not in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"  Frozen {frozen_count} vision encoder parameters.")

    if config.freeze_projector:
        logger.info("Freezing projector (model.visual.merger.*)...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if "visual" in name and "merger" in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"  Frozen {frozen_count} projector parameters.")

    if config.freeze_llm:
        logger.info("Freezing LLM backbone...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"  Frozen {frozen_count} LLM parameters.")


def _log_model_stats(model):
    stats = {
        "vision_encoder": {"trainable": 0, "total": 0},
        "projector": {"trainable": 0, "total": 0},
        "llm": {"trainable": 0, "total": 0},
    }

    for name, param in model.named_parameters():
        numel = param.numel()
        if "visual" in name:
            if "merger" in name:
                key = "projector"
            else:
                key = "vision_encoder"
        else:
            key = "llm"

        stats[key]["total"] += numel
        if param.requires_grad:
            stats[key]["trainable"] += numel

    logger.info("=" * 60)
    logger.info("Model Parameter Statistics:")
    logger.info("-" * 60)
    total_trainable = 0
    total_all = 0
    for component, s in stats.items():
        trainable = s["trainable"]
        total = s["total"]
        total_trainable += trainable
        total_all += total
        pct = 100 * trainable / total if total > 0 else 0
        logger.info(
            f"  {component:20s}: {trainable:>12,} / {total:>12,} "
            f"({pct:5.1f}%) trainable"
        )
    logger.info("-" * 60)
    pct = 100 * total_trainable / total_all if total_all > 0 else 0
    logger.info(
        f"  {'TOTAL':20s}: {total_trainable:>12,} / {total_all:>12,} "
        f"({pct:5.1f}%) trainable"
    )
    logger.info("=" * 60)


def setup_dataset(config: VLNTrainConfig) -> torch.utils.data.Dataset:
    datasets = []
    if config.vln_data_roots:
        nav_config = NavDatasetConfig(
            max_history_frames=config.max_history_frames,
            max_action_lookahead=config.max_action_lookahead,
            distance_threshold=config.distance_threshold,
            use_action_symbols=config.use_action_symbols,
            prefix_trainable=config.prefix_trainable,
        )

        for root in config.vln_data_roots:
            json_path = os.path.join(root, "exploration_data.json")
            if not os.path.exists(json_path):
                 logger.warning(f"Metadata file not found: {json_path}. Skipping this root.")
                 continue

            logger.info(f"Loading VLN dataset from: {root}")
            ds = VLNNavDataset(
                json_path=json_path,
                data_root=root,
                config=nav_config,
                seed=config.seed,
            )
            datasets.append(ds)
            
    if config.llava_video_data_root and config.llava_video_json_path:
        logger.info(f"Loading LLaVA-Video dataset from: {config.llava_video_data_root}")
        video_ds = LLaVAVideoDataset(
            data_root=config.llava_video_data_root,
            json_path=config.llava_video_json_path,
            video_folder="",
        )
        datasets.append(video_ds)

    if not datasets:
        raise ValueError("No valid datasets loaded! Check your data paths.")

    combined_dataset = ConcatDataset(datasets)
    logger.info(f"Combined dataset size: {len(combined_dataset)} samples")
    
    return combined_dataset


def build_training_arguments(config: VLNTrainConfig) -> TrainingArguments:
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.llm_lr, 
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=config.remove_unused_columns,
        deepspeed=config.deepspeed,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.run_name,
        save_safetensors=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if config.gradient_checkpointing
        else None,
        resume_from_checkpoint=config.resume_from_checkpoint,
    )

    return training_args


def main():
    config = parse_args()
    setup_logging(config.local_rank)
    set_seed(config.seed)
    model, processor = setup_model_and_processor(config)
    train_dataset = setup_dataset(config)
    data_collator = VLNDataCollator(
        processor=processor,
        max_seq_length=config.max_seq_length,
        prefix_trainable=config.prefix_trainable,
    )
    training_args = build_training_arguments(config)
    trainer = VLNTrainer(
        train_config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
        processor=processor,
    )
    
    logger.info("Starting training...")
    
    resume_from_checkpoint = config.resume_from_checkpoint
    resume_disabled = (
        isinstance(resume_from_checkpoint, str)
        and resume_from_checkpoint.lower() in ["false", "0", "no"]
    )
    if resume_disabled:
        resume_from_checkpoint = None
        
    if os.path.isdir(config.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(config.output_dir)
        if last_checkpoint is not None and not resume_disabled:
            if resume_from_checkpoint is None or (
                isinstance(resume_from_checkpoint, str)
                and resume_from_checkpoint.lower() in ["true", "1", "yes"]
            ):
                resume_from_checkpoint = last_checkpoint

    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete!")
    logger.info(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
