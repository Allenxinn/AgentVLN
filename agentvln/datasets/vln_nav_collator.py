import os
import torch
import numpy as np
from typing import Dict, List, Any, Sequence
from PIL import Image

IGNORE_INDEX = -100


class VLNDataCollator:
    def __init__(self, processor, max_seq_length: int = 4096, 
                 prefix_trainable: bool = True):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_length = max_seq_length
        self.prefix_trainable = prefix_trainable

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        all_images = []
        all_image_grid_thw = []

        for feature in features:
            input_ids, labels, pixel_values, image_grid_thw = (
                self._process_single_sample(feature)
            )
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(
                torch.ones_like(input_ids, dtype=torch.long)
            )
            if pixel_values is not None:
                all_images.append(pixel_values)
            if image_grid_thw is not None:
                all_image_grid_thw.append(image_grid_thw)
        input_ids_padded = self._pad_sequence(
            batch_input_ids, self.tokenizer.pad_token_id
        )
        labels_padded = self._pad_sequence(batch_labels, IGNORE_INDEX)
        attention_mask_padded = self._pad_sequence(batch_attention_mask, 0)

        batch = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask_padded,
        }
        if all_images:
            batch["pixel_values"] = torch.cat(all_images, dim=0)
        if all_image_grid_thw:
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        return batch

    def _process_single_sample(self, feature: Dict[str, Any]):
        prompt = feature["prompt"]
        response = feature["response"]
        images_paths = feature["images"]
        trainable_ranges = feature.get("trainable_ranges", [])
        pil_images = []
        valid_image_count = 0
        for item in images_paths:
            if isinstance(item, Image.Image):
                # Already loaded (e.g. from LMDB)
                pil_images.append(item)
                valid_image_count += 1
            elif isinstance(item, str):
                if os.path.exists(item):
                    try:
                        img = Image.open(item).convert("RGB")
                        pil_images.append(img)
                        valid_image_count += 1
                    except Exception:
                        raise ValueError(f"Fail to load from path {item}")
                        pil_images.append(Image.new("RGB", (640, 480), (0, 0, 0)))
                        valid_image_count += 1
                else:
                    raise ValueError(f"Path is not exist: {item}")
                    pil_images.append(Image.new("RGB", (640, 480), (0, 0, 0)))
                    valid_image_count += 1
            else:
                raise ValueError(f"item is not Image or str, but got {type(item)}: {item}")
                pil_images.append(Image.new("RGB", (640, 480), (0, 0, 0)))
                valid_image_count += 1
        content_parts = self._build_content_parts(prompt, pil_images)

        messages = [
            {
                "role": "user",
                "content": content_parts,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=pil_images if pil_images else None,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        labels = self._build_labels(input_ids, text, response, trainable_ranges)

        return input_ids, labels, pixel_values, image_grid_thw

    def _build_content_parts(self, prompt: str, pil_images: List[Image.Image]) -> list:
        parts = []
        image_token = "<image>"
        img_idx = 0
        
        segments = prompt.split(image_token)
        for i, segment in enumerate(segments):
            if segment:
                parts.append({"type": "text", "text": segment})
            if i < len(segments) - 1 and img_idx < len(pil_images):
                parts.append({"type": "image", "image": pil_images[img_idx]})
                img_idx += 1

        return parts

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        full_text: str,
        response: str,
        trainable_ranges: List,
    ) -> torch.Tensor:
        labels = input_ids.clone()
        
        assistant_token_ids = self.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )
        
        response_start = self._find_subsequence(
            input_ids.tolist(), assistant_token_ids
        )
        
        if response_start >= 0:
            response_content_start = response_start + len(assistant_token_ids)
            labels[:response_content_start] = IGNORE_INDEX
        else:
            response_token_ids = self.tokenizer.encode(
                response, add_special_tokens=False
            )
            response_start = self._find_subsequence(
                input_ids.tolist(), response_token_ids
            )
            if response_start >= 0:
                labels[:response_start] = IGNORE_INDEX
            else:
                mid = len(labels) // 2
                labels[:mid] = IGNORE_INDEX

        if self.tokenizer.eos_token_id is not None:
            eos_positions = (input_ids == self.tokenizer.eos_token_id).nonzero(
                as_tuple=True
            )[0]
            if len(eos_positions) > 0:
                # Mask all EOS tokens EXCEPT the last one (assistant's end)
                for pos in eos_positions[:-1]:
                    labels[pos] = IGNORE_INDEX
                # Ensure the last EOS IS trainable (not masked)
                labels[eos_positions[-1]] = input_ids[eos_positions[-1]]

        return labels

    @staticmethod
    def _find_subsequence(sequence: list, subsequence: list) -> int:
        n, m = len(sequence), len(subsequence)
        last_found = -1
        for i in range(n - m + 1):
            if sequence[i:i + m] == subsequence:
                last_found = i
        return last_found

    @staticmethod
    def _pad_sequence(
        tensors: List[torch.Tensor], pad_value: int
    ) -> torch.Tensor:
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            padding = torch.full(
                (max_len - t.size(0),), pad_value, dtype=t.dtype, device=t.device
            )
            padded.append(torch.cat([t, padding]))
        return torch.stack(padded)
