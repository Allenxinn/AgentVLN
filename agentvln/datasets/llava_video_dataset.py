import os
import json
import cv2
import random
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset

class LLaVAVideoDataset(Dataset):
    def __init__(self, data_root: str, json_path: str, num_frames: int = 8, video_folder: str = ""):
        self.data_root = data_root
        self.video_folder = video_folder
        self.num_frames = num_frames
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        print(f"[LLaVAVideoDataset] Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        video_path = item.get('video')
        if not video_path:
             image_path = item.get('image')
             if image_path:
                 frames = self._load_image(image_path)
             else:
                 frames = [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.num_frames)]
        else:
            full_video_path = os.path.join(self.data_root, self.video_folder, video_path)
            frames = self._load_video(full_video_path)

        conversations = item.get('conversations', [])
        prompt = ""
        response = ""
        image_tokens = "<image>" * len(frames)
        
        for turn in conversations:
            role = turn.get('from', '')
            value = turn.get('value', '')
            
            if role == 'human':
                if '<video>' in value:
                    value = value.replace('<video>\n', '').replace('<video>', '')
                    prompt += image_tokens + "\n" + value
                else:
                    prompt += value
            elif role == 'gpt':
                response = value
                
        return {
            "prompt": prompt,
            "response": response,
            "images": frames,
            "trainable_ranges": None,
            "metadata": {
                "id": item.get('id', str(idx)),
                "video_path": video_path
            }
        }

    def _load_video(self, video_path: str) -> List[Image.Image]:
        if not os.path.exists(video_path):
            if not os.path.exists(video_path + ".mp4"):
                 print(f"Video not found: {video_path}")
                 return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.num_frames)]
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.num_frames)]
                
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                else:
                    frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
            
            cap.release()
            
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), (0, 0, 0)))
                
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.num_frames)]

    def _load_image(self, image_path: str) -> List[Image.Image]:
        full_path = os.path.join(self.data_root, image_path)
        try:
            img = Image.open(full_path).convert("RGB")
            return [img]
        except Exception:
            return [Image.new('RGB', (224, 224), (0, 0, 0))]
