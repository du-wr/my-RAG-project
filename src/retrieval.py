import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv

from src.embedding_models import EmbeddingClient
from src.ingestion import tokenize_for_bm25
from src.reranking import CrossEncoderReranker, LLMReranker

_log = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document = None
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document = doc
                    document_path = path
                    break

        if document is None or document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        tokenized_query = tokenize_for_bm25(query)
        scores = bm25_index.get_scores(tokenized_query)
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        retrieval_results = []
        seen_pages = set()
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] in seen_pages:
                    continue
                seen_pages.add(parent_page["page"])
                retrieval_results.append({"distance": score, "page": parent_page["page"], "text": parent_page["text"], "chunk_type": "page"})
            else:
                retrieval_results.append({"distance": score, "page": chunk["page"], "text": chunk["text"], "chunk_type": chunk.get("type", "content"), "chunk_id": chunk.get("id")})
        return retrieval_results


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_model: str = None, provider: str = "openai"):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
        self.provider = provider
        self.hnsw_ef_search = int(os.getenv("HNSW_EF_SEARCH", "64"))
        self.all_dbs = self._load_dbs()
        self.embedding_client = EmbeddingClient(self.embedding_model)

    def _load_dbs(self):
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob("*.json"))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob("*.faiss")}

        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            with open(document_path, "r", encoding="utf-8") as f:
                document = json.load(f)
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
                if hasattr(vector_db, "hnsw"):
                    vector_db.hnsw.efSearch = self.hnsw_ef_search
            except Exception as exc:
                _log.error(f"Error reading vector DB for {document_path.name}: {exc}")
                continue
            all_dbs.append({"name": stem, "vector_db": vector_db, "document": document})
        return all_dbs

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            if report["document"]["metainfo"].get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        actual_top_n = min(top_n, len(chunks))

        embedding = self.embedding_client.encode(query)[0]
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)

        retrieval_results = []
        seen_pages = set()
        for distance, index in zip(distances[0], indices[0]):
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] in seen_pages:
                    continue
                seen_pages.add(parent_page["page"])
                retrieval_results.append({"distance": round(float(distance), 4), "page": parent_page["page"], "text": parent_page["text"], "chunk_type": "page"})
            else:
                retrieval_results.append({"distance": round(float(distance), 4), "page": chunk["page"], "text": chunk["text"], "chunk_type": chunk.get("type", "content"), "chunk_id": chunk.get("id")})
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        for report in self.all_dbs:
            document = report["document"]
            if document["metainfo"].get("company_name") == company_name:
                return [{"distance": 0.5, "page": page["page"], "text": page["text"]} for page in sorted(document["content"]["pages"], key=lambda p: p["page"])]
        raise ValueError(f"No report found with '{company_name}' company name.")


class HybridRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        bm25_db_dir: Path = None,
        provider: str = "openai",
        rerank_strategy: str = "llm",
        rerank_model: str = None,
        embedding_model: str = None,
        cross_encoder_model: str = None,
    ):
        load_dotenv()
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir, embedding_model=embedding_model, provider=provider)
        self.bm25_retriever = BM25Retriever(bm25_db_dir, documents_dir) if bm25_db_dir else None
        self.rerank_strategy = rerank_strategy
        self.cross_encoder_model = cross_encoder_model or os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")

        if rerank_strategy == "cross_encoder":
            self.reranker = CrossEncoderReranker(model=self.cross_encoder_model)
        else:
            self.reranker = LLMReranker(provider=provider, model=rerank_model)

    @staticmethod
    def _fuse_candidates_rrf(vector_results: List[Dict], bm25_results: List[Dict], rrf_k: int = 60) -> List[Dict]:
        # 粗排阶段只融合排序名次，不直接混合原始分数，避免 BM25 与向量分数不可比。
        merged = {}

        for source_name, results in (("vector", vector_results), ("bm25", bm25_results)):
            for rank, item in enumerate(results, start=1):
                key = (item["page"], item["text"])
                if key not in merged:
                    merged[key] = {
                        "page": item["page"],
                        "text": item["text"],
                        "distance": float(item["distance"]),
                        "rough_score": 0.0,
                        "sources": [],
                    }
                merged[key]["rough_score"] += 1.0 / (rrf_k + rank)
                merged[key]["distance"] = max(merged[key]["distance"], float(item["distance"]))
                if source_name not in merged[key]["sources"]:
                    merged[key]["sources"].append(source_name)

        fused_results = list(merged.values())
        fused_results.sort(key=lambda x: x["rough_score"], reverse=True)
        return fused_results

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages,
        )

        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=llm_reranking_sample_size,
                return_parent_pages=return_parent_pages,
            )
        else:
            bm25_results = []

        coarse_results = self._fuse_candidates_rrf(vector_results, bm25_results)[:llm_reranking_sample_size]

        if self.rerank_strategy == "cross_encoder":
            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=coarse_results,
                top_n=top_n,
            )
        else:
            # Qwen ????????????? JSON ????????????
            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=coarse_results,
                documents_batch_size=1,
                llm_weight=llm_weight,
            )
        return reranked_results[:top_n]
