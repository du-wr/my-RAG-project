import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ingestion import BM25Ingestor, VectorDBIngestor
from src.mineru_parsing import MinerUImporter
from src.questions_processing import QuestionsProcessor
from src.text_splitter import TextSplitter

load_dotenv()

# 默认优先使用已经下载到 D 盘的本地 bge-m3，避免重复联机下载大模型。
DEFAULT_LOCAL_BGE_M3_PATH = r'D:\AI_Cache\modelscope\BAAI\bge-m3'


def _get_env_or_default(*env_keys: str, default: str) -> str:
    """按优先级读取环境变量，找不到时返回默认值。"""
    for env_key in env_keys:
        value = os.getenv(env_key)
        if value:
            return value
    return default


def _resolve_answering_model(config_env_key: str, default_model: str) -> str:
    """统一解析问答模型。

    优先级说明：
    1. 每个配置专属环境变量，例如 QWEN_MAX_LLM_TOP5_MODEL。
    2. 全局环境变量 QWEN_CHAT_MODEL，可一键覆盖所有配置。
    3. 代码中的默认模型兜底。
    """
    return _get_env_or_default(config_env_key, "QWEN_CHAT_MODEL", default=default_model)


def _get_env_int(env_key: str, default: int) -> int:
    """读取整数环境变量，解析失败时回退到默认值。"""
    raw_value = os.getenv(env_key)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


@dataclass
class PipelinePaths:
    root_path: Path
    subset_path: Path
    questions_file_path: Path
    answers_file_path: Path
    debug_data_path: Path
    merged_reports_path: Path
    reports_markdown_path: Path
    databases_path: Path
    documents_dir: Path
    vector_db_dir: Path
    bm25_db_path: Path


@dataclass
class RunConfig:
    use_bm25_db: bool = True
    parent_document_retrieval: bool = True
    llm_reranking: bool = False
    reranking_strategy: str = "llm"
    llm_reranking_sample_size: int = 24
    top_n_retrieval: int = 8
    parallel_requests: int = 3
    api_provider: str = "qwen"
    answering_model: str = "qwen-max"
    embedding_model: str = DEFAULT_LOCAL_BGE_M3_PATH
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    config_suffix: str = ""
    full_context: bool = False
    enable_rule_shortcuts: bool = True


class Pipeline:
    def __init__(self, root_path: Path, run_config: RunConfig):
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path) -> PipelinePaths:
        debug_data_path = root_path / "debug_data"
        databases_path = root_path / "databases"
        return PipelinePaths(
            root_path=root_path,
            subset_path=root_path / "subset.csv",
            questions_file_path=root_path / "questions.json",
            answers_file_path=root_path / f"answers{self.run_config.config_suffix}.json",
            debug_data_path=debug_data_path,
            merged_reports_path=debug_data_path / "02_merged_reports",
            reports_markdown_path=debug_data_path / "03_reports_markdown",
            databases_path=databases_path,
            documents_dir=databases_path / "chunked_reports",
            vector_db_dir=databases_path / "vector_dbs",
            bm25_db_path=databases_path / "bm25_dbs",
        )

    def _convert_json_to_csv_if_needed(self):
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.subset_path
        if json_path.exists() and not csv_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pd.DataFrame(data).to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _reset_output_dirs(self, *paths: Path):
        for path in paths:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    def chunk_reports(self):
        # 统一在页级重建结果上切块，保证后续检索和页码溯源一致。
        text_splitter = TextSplitter()
        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            None,
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")
    def create_vector_dbs(self):
        vdb_ingestor = VectorDBIngestor(
            model=self.run_config.embedding_model,
            provider=self.run_config.api_provider,
        )
        vdb_ingestor.process_reports(self.paths.documents_dir, self.paths.vector_db_dir)
        print(f"Vector databases created in {self.paths.vector_db_dir}")

    def create_bm25_db(self):
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(self.paths.documents_dir, self.paths.bm25_db_path)
        print(f"BM25 databases created in {self.paths.bm25_db_path}")

    def process_mineru_reports(self, source_dir: Path):
        print("Starting MinerU reports processing pipeline...")
        self._reset_output_dirs(
            self.paths.merged_reports_path,
            self.paths.reports_markdown_path,
            self.paths.documents_dir,
            self.paths.vector_db_dir,
        )
        if self.run_config.use_bm25_db:
            self._reset_output_dirs(self.paths.bm25_db_path)

        print("Step 1: Importing MinerU content_list outputs...")
        importer = MinerUImporter(self.paths.subset_path)
        imported = importer.import_reports(Path(source_dir), self.paths.merged_reports_path)
        print(f"Imported {imported} MinerU reports into {self.paths.merged_reports_path}")

        print("Step 2: Chunking imported reports...")
        self.chunk_reports()

        print("Step 3: Creating vector databases...")
        self.create_vector_dbs()

        if self.run_config.use_bm25_db:
            print("Step 4: Creating BM25 databases...")
            self.create_bm25_db()

        print("MinerU reports processing pipeline completed successfully!")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter:02d}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def _build_questions_processor(self) -> QuestionsProcessor:
        # 批处理问答与交互问答共用同一套 QuestionsProcessor，
        # 这样配置、提示词和输出结构都保持一致。
        return QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            bm25_db_dir=self.paths.bm25_db_path if self.run_config.use_bm25_db else None,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            reranking_strategy=self.run_config.reranking_strategy,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            embedding_model=self.run_config.embedding_model,
            cross_encoder_model=self.run_config.cross_encoder_model,
            full_context=self.run_config.full_context,
            enable_rule_shortcuts=self.run_config.enable_rule_shortcuts,
        )

    def process_questions(self):
        processor = self._build_questions_processor()
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        processor.process_all_questions(output_path=output_path, submission_file=False)
        debug_output_path = output_path.with_name(output_path.stem + "_debug" + output_path.suffix)
        print(f"Debug answers saved to {debug_output_path}")

    def answer_single_question(self, question_text: str, schema: str | None = None) -> dict:
        processor = self._build_questions_processor()
        return processor.answer_single_question(question_text, schema=schema)


configs = {
    "qwen_default": RunConfig(
        use_bm25_db=True,
        parent_document_retrieval=True,
        llm_reranking=False,
        api_provider="qwen",
        answering_model=_resolve_answering_model("QWEN_DEFAULT_MODEL", "qwen-max"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_BGE_M3_PATH),
        parallel_requests=_get_env_int("QUESTION_PARALLEL_REQUESTS", 3),
        config_suffix="_qwen_default",
    ),
    "qwen_turbo": RunConfig(
        use_bm25_db=True,
        parent_document_retrieval=True,
        llm_reranking=False,
        api_provider="qwen",
        answering_model=_resolve_answering_model("QWEN_TURBO_MODEL", "qwen-turbo"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_BGE_M3_PATH),
        parallel_requests=_get_env_int("QUESTION_PARALLEL_REQUESTS", 3),
        config_suffix="_qwen_turbo",
    ),
    "qwen_max_rerank": RunConfig(
        use_bm25_db=True,
        parent_document_retrieval=True,
        llm_reranking=True,
        reranking_strategy="llm",
        llm_reranking_sample_size=24,
        top_n_retrieval=8,
        api_provider="qwen",
        answering_model=_resolve_answering_model("QWEN_MAX_RERANK_MODEL", "qwen-max"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_BGE_M3_PATH),
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
        parallel_requests=_get_env_int("QUESTION_PARALLEL_REQUESTS", 3),
        config_suffix="_qwen_max_rerank",
    ),
    "qwen_plus_rerank": RunConfig(
        use_bm25_db=True,
        parent_document_retrieval=True,
        llm_reranking=True,
        reranking_strategy="llm",
        llm_reranking_sample_size=18,
        top_n_retrieval=8,
        api_provider="qwen",
        answering_model=_resolve_answering_model("QWEN_PLUS_RERANK_MODEL", "qwen-plus"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_BGE_M3_PATH),
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
        parallel_requests=_get_env_int("QUESTION_PARALLEL_REQUESTS", 3),
        config_suffix="_qwen_plus_rerank",
    ),
    "qwen_max_llm_top5": RunConfig(
        use_bm25_db=True,
        parent_document_retrieval=True,
        llm_reranking=True,
        reranking_strategy="llm",
        llm_reranking_sample_size=5,
        top_n_retrieval=5,
        api_provider="qwen",
        answering_model=_resolve_answering_model("QWEN_MAX_LLM_TOP5_MODEL", "qwen-max"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_BGE_M3_PATH),
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
        parallel_requests=_get_env_int("QUESTION_PARALLEL_REQUESTS", 3),
        config_suffix="_qwen_max_llm_top5",
    ),
    "qwen_max_ce_top5": RunConfig(
        use_bm25_db=True,
        # 交叉编码器精排对长页文本开销较大，这里改为 chunk 级检索并缩小粗排候选数。
        parent_document_retrieval=False,
        llm_reranking=True,
        reranking_strategy="cross_encoder",
        llm_reranking_sample_size=10,
        top_n_retrieval=5,
        api_provider="qwen",
        answering_model=_resolve_answering_model("QWEN_MAX_CE_TOP5_MODEL", "qwen-max"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_BGE_M3_PATH),
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
        parallel_requests=_get_env_int("QUESTION_PARALLEL_REQUESTS", 3),
        config_suffix="_qwen_max_ce_top5",
    ),
    "qwen_max_ce_top5_rag_only": RunConfig(
        use_bm25_db=True,
        # 纯 RAG 基线沿用当前交叉编码器配置，但显式关闭规则捷径，便于评估真实检索问答能力。
        parent_document_retrieval=False,
        llm_reranking=True,
        reranking_strategy="cross_encoder",
        llm_reranking_sample_size=10,
        top_n_retrieval=5,
        api_provider="qwen",
        answering_model=_resolve_answering_model("QWEN_MAX_CE_TOP5_RAG_ONLY_MODEL", "qwen-max"),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_BGE_M3_PATH),
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
        parallel_requests=_get_env_int("QUESTION_PARALLEL_REQUESTS", 3),
        config_suffix="_qwen_max_ce_top5_rag_only",
        enable_rule_shortcuts=False,
    ),
}



