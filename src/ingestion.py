import json
import os
import pickle
import re
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from src.embedding_models import EmbeddingClient

try:
    import jieba
except ImportError:  # pragma: no cover - optional dependency
    jieba = None


def tokenize_for_bm25(text: str) -> List[str]:
    if not text:
        return []
    if jieba is not None:
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    return [token for token in re.split(r"\s+|(?<=[，。！？；])", text) if token.strip()]


class BM25Ingestor:
    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        tokenized_chunks = [tokenize_for_bm25(chunk) for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            text_chunks = [chunk["text"] for chunk in report_data["content"]["chunks"]]
            bm25_index = self.create_bm25_index(text_chunks)

            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(bm25_index, f)

        print(f"Processed {len(all_report_paths)} reports")


class VectorDBIngestor:
    def __init__(self, model: str = None, provider: str = None):
        load_dotenv()
        self.model = model or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
        self.provider = provider or "qwen"
        self.embedding_client = EmbeddingClient(self.model)
        self.hnsw_m = int(os.getenv("HNSW_M", "32"))
        self.hnsw_ef_construction = int(os.getenv("HNSW_EF_CONSTRUCTION", "80"))

    def _get_batch_size(self) -> int:
        model_lower = self.model.lower()
        if model_lower in {"bge-m3", "baai/bge-m3"}:
            return 16
        if "text-embedding-v3" in model_lower or "text-embedding-v4" in model_lower:
            return 10
        if "text-embedding-v1" in model_lower:
            return 25
        return 64

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
    def _get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            raise ValueError("Input text cannot be empty.")

        batch_size = self._get_batch_size()
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeddings.extend(self.embedding_client.encode(batch))
        return embeddings

    def _create_vector_db(self, embeddings: List[List[float]]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        # 当前主线使用 HNSW + 内积索引，适合中等规模中文检索实验。
        index = faiss.IndexHNSWFlat(dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.add(embeddings_array)
        return index

    def _process_report(self, report: dict):
        text_chunks = [chunk["text"] for chunk in report["content"]["chunks"]]
        embeddings = self._get_embeddings(text_chunks)
        return self._create_vector_db(embeddings)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")
