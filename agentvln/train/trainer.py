import os
import math
import torch
from typing import Dict, List, Optional, Any, Tuple
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import has_length


class VLNTrainer(Trainer):
    def __init__(self, train_config=None, processor=None, **kwargs):
        self.train_config = train_config
        self.processor = processor
        super().__init__(**kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
            
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        config = self.train_config

        if config is None:
            return super().create_optimizer()
            
        vision_encoder_params = []
        projector_params = []
        llm_params = []
        
        vision_encoder_names = []
        projector_names = []
        llm_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "visual" in name:
                if "merger" in name:
                    projector_params.append(param)
                    projector_names.append(name)
                else:
                    vision_encoder_params.append(param)
                    vision_encoder_names.append(name)
            else:
                llm_params.append(param)
                llm_names.append(name)

        optimizer_grouped_parameters = []

        if llm_params:
            optimizer_grouped_parameters.append({
                "params": llm_params,
                "lr": config.llm_lr,
                "weight_decay": config.weight_decay,
                "name": "llm",
            })
            print(f"[Optimizer] LLM parameters: {len(llm_params)}, "
                  f"lr={config.llm_lr}")

        if vision_encoder_params:
            optimizer_grouped_parameters.append({
                "params": vision_encoder_params,
                "lr": config.vision_encoder_lr,
                "weight_decay": config.weight_decay,
                "name": "vision_encoder",
            })
            print(f"[Optimizer] Vision encoder parameters: "
                  f"{len(vision_encoder_params)}, lr={config.vision_encoder_lr}")

        if projector_params:
            optimizer_grouped_parameters.append({
                "params": projector_params,
                "lr": config.projector_lr,
                "weight_decay": config.weight_decay,
                "name": "projector",
            })
            print(f"[Optimizer] Projector parameters: "
                  f"{len(projector_params)}, lr={config.projector_lr}")

        if not optimizer_grouped_parameters:
            raise ValueError(
                "No trainable parameters found! Check freeze settings."
            )
            
        total_trainable = sum(
            p.numel() for group in optimizer_grouped_parameters 
            for p in group["params"]
        )
        total_all = sum(p.numel() for p in model.parameters())
        print(f"[Optimizer] Trainable parameters: {total_trainable:,} / "
              f"{total_all:,} ({100 * total_trainable / total_all:.2f}%)")
              
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            "betas": (config.adam_beta1, config.adam_beta2),
            "eps": config.adam_epsilon,
        }

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs
        )

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer=None
    ):
        return super().create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer or self.optimizer,
        )

    def _log_parameter_groups(self):
        if self.optimizer is None:
            return
        
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            lr = group.get("lr", "N/A")
            num_params = len(group["params"])
            total_elements = sum(p.numel() for p in group["params"])
            print(f"  [{name}] lr={lr}, params={num_params}, "
                  f"elements={total_elements:,}")
