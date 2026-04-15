import os
import threading
from typing import List, Union

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingClient:
    _local_model_cache = {}
    _local_model_lock = threading.Lock()
    _local_encode_locks = {}

    def __init__(self, model_name: str):
        load_dotenv()
        self.model_name = model_name
        normalized_name = model_name.replace('\\', '/').lower()
        # 支持模型名和本地目录两种写法，优先把 bge-m3 作为本地模型加载。
        self.use_local_bge = (
            normalized_name in {'bge-m3', 'baai/bge-m3'}
            or os.path.isdir(model_name)
        )
        self.remote_client = None if self.use_local_bge else self._set_up_remote_client()

    def _set_up_remote_client(self):
        kwargs = {'timeout': None, 'max_retries': 2}
        qwen_key = os.getenv('QWEN_API_KEY')
        if qwen_key:
            kwargs['api_key'] = qwen_key
            base_url = os.getenv('QWEN_BASE_URL')
            if base_url:
                kwargs['base_url'] = base_url
        else:
            kwargs['api_key'] = os.getenv('OPENAI_API_KEY')
        return OpenAI(**kwargs)

    @staticmethod
    def _resolve_local_device() -> str:
        configured_device = os.getenv('EMBEDDING_DEVICE')
        if configured_device:
            return configured_device

        try:
            import torch

            # 本地 bge-m3 优先使用 CUDA，建库和查询都能明显降低耗时。
            if torch.cuda.is_available():
                return 'cuda'
        except Exception:
            pass

        return 'cpu'

    @classmethod
    def _get_local_model(cls, model_name: str, device: str):
        cache_key = (model_name, device)
        if cache_key in cls._local_model_cache:
            return cls._local_model_cache[cache_key]

        with cls._local_model_lock:
            if cache_key not in cls._local_model_cache:
                from sentence_transformers import SentenceTransformer

                # 统一复用本地模型实例，避免重复加载后出现显存浪费或 meta tensor 相关问题。
                cls._local_model_cache[cache_key] = SentenceTransformer(model_name, device=device)
        return cls._local_model_cache[cache_key]

    @classmethod
    def _get_local_encode_lock(cls, model_name: str, device: str):
        cache_key = (model_name, device)
        with cls._local_model_lock:
            if cache_key not in cls._local_encode_locks:
                # 共享的 embedding 实例内部会复用 tokenizer 和缓冲区，
                # 同一个实例被多个线程同时 encode 时容易触发并发借用错误。
                cls._local_encode_locks[cache_key] = threading.Lock()
        return cls._local_encode_locks[cache_key]

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            raise ValueError('Input text cannot be empty.')

        if self.use_local_bge:
            device = self._resolve_local_device()
            local_model_name = self.model_name if os.path.isdir(self.model_name) else 'BAAI/bge-m3'
            model = self._get_local_model(local_model_name, device)
            encode_lock = self._get_local_encode_lock(local_model_name, device)
            with encode_lock:
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            if isinstance(embeddings, np.ndarray):
                return embeddings.astype(np.float32).tolist()
            return [np.asarray(item, dtype=np.float32).tolist() for item in embeddings]

        response = self.remote_client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]

