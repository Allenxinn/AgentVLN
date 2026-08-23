import os
import lmdb
import cv2
import numpy as np
from typing import Dict, Optional


class LMDBImageStore: 
    DEFAULT_FORMATS = {
        'rgb': 'jpg',
        'topdown': 'jpg',
    }
    
    def __init__(
        self,
        output_base_path,
        map_size = 1024 * 1024 * 1024 * 100,
        image_formats = None,
        jpg_quality = 95
    ):
        self.output_base_path = output_base_path
        self.map_size = map_size
        self.jpg_quality = jpg_quality
        self._envs: Dict[str, lmdb.Environment] = {}
        
        self.image_formats = dict(self.DEFAULT_FORMATS)
        if image_formats:
            for k, v in image_formats.items():
                if k in self.DEFAULT_FORMATS and v.lower() in ('jpg', 'jpeg', 'png'):
                    self.image_formats[k] = 'jpg' if v.lower() in ('jpg', 'jpeg') else 'png'
        
        os.makedirs(output_base_path, exist_ok=True)
    
    def _get_env(self, scene_id) -> lmdb.Environment:
        if scene_id not in self._envs:
            lmdb_path = os.path.join(self.output_base_path, f"{scene_id}.lmdb")
            os.makedirs(lmdb_path, exist_ok=True)
            self._envs[scene_id] = lmdb.open(
                lmdb_path,
                map_size=self.map_size,
                create=True
            )
        return self._envs[scene_id]
    
    def _encode_image(self, image, fmt = 'jpg', is_rgb = False) -> bytes:
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if fmt == 'png':
            _, encoded = cv2.imencode('.png', image)
        else:
            _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality])
        
        return encoded.tobytes()
    
    def save_rgb(self, rgb_image, scene_id, task_id, step):
        env = self._get_env(scene_id)
        fmt = self.image_formats.get('rgb', 'jpg')
        key = f"{task_id}/rgb/step_{step:04d}".encode('utf-8')
        value = self._encode_image(rgb_image, fmt=fmt, is_rgb=True)
        
        with env.begin(write=True) as txn:
            txn.put(key, value)
    
    def save_topdown(self, topdown_map, scene_id, task_id, step):
        env = self._get_env(scene_id)
        fmt = self.image_formats.get('topdown', 'jpg')
        key = f"{task_id}/topdown/step_{step:04d}".encode('utf-8')
        value = self._encode_image(topdown_map, fmt=fmt, is_rgb=False)
        
        with env.begin(write=True) as txn:
            txn.put(key, value)
    
    def close(self):
        for scene_id, env in self._envs.items():
            env.close()
        self._envs.clear()
    
    def __del__(self):
        self.close()
    
    
    @staticmethod
    def open_read(lmdb_path) -> lmdb.Environment:
        return lmdb.open(lmdb_path, readonly=True, lock=False)
    
    @staticmethod
    def decode_image(data) -> np.ndarray:
        buf = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    @staticmethod
    def list_keys(lmdb_path):
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        keys = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                keys.append(key.decode('utf-8'))
        env.close()
        return keys
    
    @staticmethod
    def get_stats(lmdb_path) -> dict:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            stat = txn.stat()
        env.close()
        return stat
