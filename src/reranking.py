import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import src.reranking_prompts as reranking_prompts


class LLMReranker:
    def __init__(self, provider: str = "qwen", model: str | None = None):
        self.provider = provider
        self.model = model or ("qwen-max" if provider == "qwen" else "gpt-4o-mini-2024-07-18")
        self.llm = self._set_up_llm()
        self.system_prompt_rerank_single_block = reranking_prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = reranking_prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks

    def _set_up_llm(self):
        load_dotenv()
        kwargs = {"timeout": None, "max_retries": 2}
        if self.provider == "qwen":
            kwargs["api_key"] = os.getenv("QWEN_API_KEY")
            base_url = os.getenv("QWEN_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
        else:
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        return OpenAI(**kwargs)

    def _create_json_completion(self, system_prompt: str, user_prompt: str):
        return self.llm.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

    @staticmethod
    def _normalize_single_block_result(payload: dict) -> dict:
        reasoning = (
            payload.get("reasoning")
            or payload.get("reasonining")
            or payload.get("analysis")
            or "??????????"
        )
        relevance_score = payload.get("relevance_score", 0.0)
        try:
            relevance_score = float(relevance_score)
        except (TypeError, ValueError):
            relevance_score = 0.0
        return {"relevance_score": relevance_score, "reasoning": reasoning}

    @classmethod
    def _normalize_multiple_blocks_result(cls, payload: dict, block_count: int) -> dict:
        if isinstance(payload.get("block_rankings"), list):
            normalized = [
                cls._normalize_single_block_result(item if isinstance(item, dict) else {})
                for item in payload["block_rankings"]
            ]
        else:
            numbered_keys = [key for key in payload.keys() if str(key).lower().startswith("block")]
            numbered_keys.sort(key=lambda key: int("".join(ch for ch in str(key) if ch.isdigit()) or 0))
            normalized = [
                cls._normalize_single_block_result(payload[key] if isinstance(payload[key], dict) else {})
                for key in numbered_keys
            ]

        while len(normalized) < block_count:
            normalized.append(
                {
                    "relevance_score": 0.0,
                    "reasoning": "?????????????????????????",
                }
            )
        return {"block_rankings": normalized[:block_count]}

    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'???"{query}"\n\n??????\n"""\n{retrieved_document}\n"""'
        completion = self._create_json_completion(self.system_prompt_rerank_single_block, user_prompt)
        content = completion.choices[0].message.content or '{}'
        payload = json.loads(content)
        return self._normalize_single_block_result(payload)

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join(
            [f'Block {i + 1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)]
        )
        user_prompt = (
            f'???"{query}"\n\n'
            f"????????\n{formatted_blocks}\n\n"
            f"????? {len(retrieved_documents)} ??? JSON ?????"
        )
        completion = self._create_json_completion(self.system_prompt_rerank_multiple_blocks, user_prompt)
        content = completion.choices[0].message.content or '{}'
        payload = json.loads(content)
        return self._normalize_multiple_blocks_result(payload, len(retrieved_documents))

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight

        if documents_batch_size == 1:
            def process_single_doc(doc):
                ranking = self.get_rank_for_single_block(query, doc["text"])
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + vector_weight * doc["distance"],
                    4,
                )
                return doc_with_score

            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
        else:
            def process_batch(batch):
                texts = [doc["text"] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                block_rankings = rankings.get("block_rankings", [])

                while len(block_rankings) < len(batch):
                    block_rankings.append(
                        {
                            "relevance_score": 0.0,
                            "reasoning": "?????????????????????????",
                        }
                    )

                results = []
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + vector_weight * doc["distance"],
                        4,
                    )
                    results.append(doc_with_score)
                return results

            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            all_results = [item for batch in batch_results for item in batch]

        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results


class CrossEncoderReranker:
    _model_cache = {}

    def __init__(self, model: str | None = None):
        load_dotenv()
        self.model_name = model or os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.device = self._resolve_device()
        self.model = self._get_model(self.model_name, self.device)

    @staticmethod
    def _resolve_device() -> str:
        configured_device = os.getenv("CROSS_ENCODER_DEVICE")
        if configured_device:
            return configured_device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass

        return "cpu"

    @classmethod
    def _get_model(cls, model_name: str, device: str):
        cache_key = (model_name, device)
        if cache_key not in cls._model_cache:
            from sentence_transformers import CrossEncoder

            cls._model_cache[cache_key] = CrossEncoder(model_name, device=device)
        return cls._model_cache[cache_key]

    def rerank_documents(self, query: str, documents: list, top_n: int = 5):
        if not documents:
            return []

        sentence_pairs = [(query, document["text"]) for document in documents]
        scores = self.model.predict(sentence_pairs, show_progress_bar=False)
        scores = np.asarray(scores, dtype=np.float32)

        reranked_results = []
        for document, score in zip(documents, scores):
            item = document.copy()
            item["cross_encoder_score"] = round(float(score), 6)
            item["combined_score"] = round(float(score), 6)
            reranked_results.append(item)

        reranked_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return reranked_results[:top_n]
