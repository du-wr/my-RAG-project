import concurrent.futures
import json
import os
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from src.api_requests import APIProcessor
from src.embedding_models import EmbeddingClient
from src.retrieval import HybridRetriever, VectorRetriever


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = "./vector_dbs",
        documents_dir: Union[str, Path] = "./documents",
        bm25_db_dir: Optional[Union[str, Path]] = None,
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        reranking_strategy: str = "llm",
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        embedding_model: str = "text-embedding-3-large",
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        full_context: bool = False,
        enable_rule_shortcuts: bool = False,
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.bm25_db_dir = Path(bm25_db_dir) if bm25_db_dir else None
        self.subset_path = Path(subset_path) if subset_path else None

        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.reranking_strategy = reranking_strategy
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context
        self.enable_rule_shortcuts = enable_rule_shortcuts

        self.answer_details = []
        self._lock = threading.Lock()
        self.companies_df = None
        self._company_alias_lookup = None
        self._company_embedding_client = None
        self._company_embedding_vectors = None
        self.question_kind_model = (os.getenv("QUESTION_KIND_MODEL") or "").strip() or None
        self.question_kind_confidence_threshold = self._get_env_float("QUESTION_KIND_CONFIDENCE_THRESHOLD", 0.75)
        self._question_kind_cache = {}

    @staticmethod
    def _get_env_float(env_key: str, default: float) -> float:
        """读取浮点环境变量，解析失败时回退到默认值。"""
        raw_value = os.getenv(env_key)
        if not raw_value:
            return default
        try:
            return float(raw_value)
        except ValueError:
            return default

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_subset(self):
        if self.companies_df is None:
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            self.companies_df = pd.read_csv(self.subset_path, encoding="utf-8-sig")

    def _format_retrieval_results(self, retrieval_results: List[Dict]) -> str:
        if not retrieval_results:
            return ""

        context_parts = []
        for result in retrieval_results:
            context_parts.append(f'Text retrieved from page {result["page"]}:\n"""\n{result["text"]}\n"""')
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        self._load_subset()
        matching_rows = self.companies_df[self.companies_df["company_name"] == company_name]
        company_sha1 = "" if matching_rows.empty else matching_rows.iloc[0]["sha1"]
        return [{"pdf_sha1": company_sha1, "page_index": page} for page in pages_list]

    def _validate_page_references(
        self,
        claimed_pages: list,
        retrieval_results: list,
        min_pages: int = 1,
        max_pages: int = 8,
    ) -> list:
        claimed_pages = claimed_pages or []
        retrieved_pages = [result["page"] for result in retrieval_results]
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]

        if len(validated_pages) < min_pages:
            for page in retrieved_pages:
                if page not in validated_pages:
                    validated_pages.append(page)
                if len(validated_pages) >= min_pages:
                    break

        return validated_pages[:max_pages]

    def _normalize_company_text(self, text: str) -> str:
        normalized = (text or "").strip().lower()
        normalized = re.sub(r"[\s　:：,，.。()（）【】\[\]《》<>\-—_/]+", "", normalized)
        for suffix in [
            "股份有限公司",
            "有限责任公司",
            "集团股份有限公司",
            "集团有限公司",
            "股份公司",
            "有限公司",
            "集团",
            "公司",
        ]:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized

    def _is_subsequence_match(self, short_text: str, long_text: str) -> bool:
        if not short_text or not long_text or len(short_text) > len(long_text):
            return False

        index = 0
        for char in long_text:
            if char == short_text[index]:
                index += 1
                if index == len(short_text):
                    return True
        return False

    def _generate_company_aliases(self, company_name: str, source_filename: str) -> List[str]:
        aliases = {company_name}
        source_stem = Path(source_filename or "").stem
        if source_stem:
            aliases.add(source_stem)
            aliases.add(source_stem.split("：")[0])
            aliases.add(source_stem.split(":")[0])
            aliases.add(source_stem.replace("年度报告", ""))
            aliases.add(source_stem.replace("年年度报告", ""))

        cleaned_aliases = set()
        for alias in aliases:
            alias = (alias or "").strip()
            if not alias:
                continue
            cleaned_aliases.add(alias)
            cleaned_aliases.add(self._normalize_company_text(alias))
        return sorted(cleaned_aliases, key=len, reverse=True)

    def _build_company_alias_lookup(self) -> List[Dict[str, object]]:
        if self._company_alias_lookup is not None:
            return self._company_alias_lookup

        self._load_subset()
        alias_entries = []
        for _, row in self.companies_df.iterrows():
            company_name = str(row.get("company_name", "") or "").strip()
            source_filename = str(row.get("source_filename", "") or "").strip()
            if not company_name:
                continue
            alias_entries.append(
                {
                    "company_name": company_name,
                    "aliases": self._generate_company_aliases(company_name, source_filename),
                    "normalized_company_name": self._normalize_company_text(company_name),
                }
            )
        alias_entries.sort(key=lambda item: len(item["normalized_company_name"]), reverse=True)
        self._company_alias_lookup = alias_entries
        return alias_entries

    def _get_company_candidate_text(self, question_text: str) -> str:
        question_text = (question_text or "").strip()
        for delimiter in ["的", "：", ":", "，", ",", "中"]:
            if delimiter in question_text:
                return question_text.split(delimiter)[0]
        return question_text

    def _get_company_embedding_client(self) -> EmbeddingClient:
        if self._company_embedding_client is None:
            self._company_embedding_client = EmbeddingClient(self.embedding_model)
        return self._company_embedding_client

    def _build_company_embedding_vectors(self) -> Dict[str, List[float]]:
        if self._company_embedding_vectors is not None:
            return self._company_embedding_vectors

        alias_lookup = self._build_company_alias_lookup()
        embedding_client = self._get_company_embedding_client()
        alias_texts = []
        alias_owners = []
        for entry in alias_lookup:
            company_name = entry["company_name"]
            for alias in entry["aliases"]:
                if not alias:
                    continue
                alias_texts.append(alias)
                alias_owners.append(company_name)

        vectors = embedding_client.encode(alias_texts)
        embedding_lookup = {}
        for alias_text, company_name, vector in zip(alias_texts, alias_owners, vectors):
            embedding_lookup[alias_text] = {"company_name": company_name, "vector": vector}

        self._company_embedding_vectors = embedding_lookup
        return embedding_lookup

    def _resolve_companies_with_fuzzy_match(self, question_text: str) -> List[str]:
        candidate_text = self._normalize_company_text(self._get_company_candidate_text(question_text))
        if not candidate_text:
            return []

        scored_matches = []
        for entry in self._build_company_alias_lookup():
            company_name = entry["company_name"]
            alias_scores = []
            for alias in entry["aliases"]:
                normalized_alias = self._normalize_company_text(alias)
                if not normalized_alias:
                    continue
                alias_scores.append(SequenceMatcher(None, normalized_alias, candidate_text).ratio())
            if alias_scores:
                scored_matches.append((max(alias_scores), company_name))

        scored_matches.sort(reverse=True)
        if not scored_matches:
            return []

        best_score, best_company = scored_matches[0]
        if best_score < 0.62:
            return []
        return [best_company]

    def _resolve_companies_with_vector_similarity(self, question_text: str) -> List[str]:
        candidate_text = self._get_company_candidate_text(question_text)
        if not candidate_text.strip():
            return []

        embedding_client = self._get_company_embedding_client()
        question_vector = embedding_client.encode(candidate_text)[0]
        embedding_lookup = self._build_company_embedding_vectors()

        company_scores = {}
        for alias_data in embedding_lookup.values():
            company_name = alias_data["company_name"]
            alias_vector = alias_data["vector"]
            score = sum(left * right for left, right in zip(question_vector, alias_vector))
            company_scores[company_name] = max(company_scores.get(company_name, -1.0), score)

        if not company_scores:
            return []

        ranked_scores = sorted(company_scores.items(), key=lambda item: item[1], reverse=True)
        best_company, best_score = ranked_scores[0]
        second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else -1.0
        if best_score < 0.45 or (best_score - second_score) < 0.02:
            return []
        return [best_company]

    def _extract_companies_from_subset(self, question_text: str) -> List[str]:
        self._load_subset()
        remaining_text = question_text or ""
        normalized_question = self._normalize_company_text(remaining_text)
        found_companies = []

        # 先按公司全名做直接匹配，命中时优先返回，避免后续模糊匹配误伤。
        for company in sorted(self.companies_df["company_name"].dropna().unique(), key=len, reverse=True):
            if company and company in remaining_text and company not in found_companies:
                found_companies.append(company)
                remaining_text = remaining_text.replace(company, "", 1)

        if found_companies:
            return found_companies

        # 再尝试别名、归一化文本和子序列匹配，兼容简称、全称与不同提问写法。
        matched_companies = []
        for entry in self._build_company_alias_lookup():
            company_name = entry["company_name"]
            aliases = entry["aliases"]
            normalized_company_name = entry["normalized_company_name"]

            exact_alias_hit = any(alias and alias in remaining_text for alias in aliases if alias != normalized_company_name)
            normalized_alias_hit = any(
                alias and alias in normalized_question
                for alias in aliases
                if alias == self._normalize_company_text(alias)
            )
            subsequence_hit = self._is_subsequence_match(normalized_company_name, normalized_question)

            if (exact_alias_hit or normalized_alias_hit or subsequence_hit) and company_name not in matched_companies:
                matched_companies.append(company_name)

        if matched_companies:
            return matched_companies

        # 最后回退到模糊匹配与向量相似度匹配。
        fuzzy_matched = self._resolve_companies_with_fuzzy_match(question_text)
        if fuzzy_matched:
            return fuzzy_matched

        return self._resolve_companies_with_vector_similarity(question_text)

    def _get_document_by_company_name(self, company_name: str) -> Optional[dict]:
        for path in self.documents_dir.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if doc["metainfo"].get("company_name") == company_name:
                return doc
        return None

    def _build_retriever(self):
        if self.llm_reranking:
            return HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                bm25_db_dir=self.bm25_db_dir,
                provider=self.api_provider,
                rerank_strategy=self.reranking_strategy,
                rerank_model=self.answering_model,
                embedding_model=self.embedding_model,
                cross_encoder_model=self.cross_encoder_model,
            )
        return VectorRetriever(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            embedding_model=self.embedding_model,
            provider=self.api_provider,
        )

    def _retrieve_context_for_company(self, retriever, company_name: str, question: str) -> List[Dict]:
        if self.full_context:
            return retriever.retrieve_all(company_name)
        return retriever.retrieve_by_company_name(
            company_name=company_name,
            query=question,
            llm_reranking_sample_size=self.llm_reranking_sample_size,
            top_n=self.top_n_retrieval,
            return_parent_pages=self.return_parent_pages,
        )

    def _build_interactive_queries(self, company_name: str, question: str) -> List[str]:
        """为交互检索构造多个查询变体，提升不同问法下的召回率。"""
        question = (question or '').strip()
        if not question:
            return []

        queries = [question]
        candidate_variants = []

        # 优先尝试“公司名 + 的 + 问题主体”的常见结构，抽出公司名后的问题部分。
        if "的" in question:
            candidate_variants.append(question.split("的", 1)[1].strip())

        stripped_question = question
        alias_entry = next(
            (entry for entry in self._build_company_alias_lookup() if entry['company_name'] == company_name),
            None,
        )

        if alias_entry is not None:
            for alias in sorted(alias_entry['aliases'], key=len, reverse=True):
                alias = (alias or '').strip()
                if not alias:
                    continue
                if alias in stripped_question:
                    stripped_question = stripped_question.replace(alias, ' ', 1)

        candidate_variants.append(stripped_question)

        for variant in candidate_variants:
            simplified_question = re.sub('[\s:\uFF1A,\uFF0C\u3002\uFF1B;\uFF1F?\u7684]+', '', (variant or '')).strip()
            simplified_question = re.sub('[\u3002\uFF1B;\uFF1F?.]+$', '', simplified_question).strip()
            simplified_question = re.sub(r'\s+', ' ', simplified_question).strip()
            if simplified_question and simplified_question not in queries:
                queries.append(simplified_question)

            keyword_question = simplified_question
            for filler in ["请问", "一下", "情况", "相关情况", "总计是多少", "是多少", "是什么"]:
                keyword_question = keyword_question.replace(filler, ' ')
            keyword_question = re.sub(r'\s+', ' ', keyword_question).strip(' :\uFF1A,\uFF0C\u3002\uFF1B;\uFF1F?')
            if len(keyword_question) >= 4 and keyword_question not in queries:
                queries.append(keyword_question)

        return queries


    @staticmethod
    def _get_result_score(result: Dict) -> float:
        for key in ['combined_score', 'rough_score', 'cross_encoder_score', 'relevance_score', 'distance']:
            value = result.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _is_table_value_question(self, question: str) -> bool:
        """识别更依赖表格块而非正文段落的数值型问题。"""
        question = (question or '').strip()
        table_keywords = [
            '账面价值', '期末账面价值', '账面余额', '期末余额', '公允价值', '减值准备', '资产情况',
            '负债情况', '使用权资产', '固定资产', '无形资产', '项目', '合计', '单位：',
        ]
        return any(keyword in question for keyword in table_keywords)

    def _rank_interactive_result(self, result: Dict, prefer_table_chunks: bool) -> float:
        score = self._get_result_score(result)
        if prefer_table_chunks and result.get('chunk_type') in {'mineru_table', 'serialized_table'}:
            score += 1000.0
        return score

    def _merge_retrieval_results(self, result_groups: List[List[Dict]], top_n: int, prefer_table_chunks: bool = False) -> List[Dict]:
        """合并多轮检索结果并去重，再按重排分数截取前 top_n 条。"""
        merged = {}
        for group in result_groups:
            for item in group:
                key = (item.get('page'), item.get('text'))
                score = self._rank_interactive_result(item, prefer_table_chunks)
                existing = merged.get(key)
                if existing is None or score > self._rank_interactive_result(existing, prefer_table_chunks):
                    merged[key] = item

        results = list(merged.values())
        results.sort(key=lambda item: self._rank_interactive_result(item, prefer_table_chunks), reverse=True)
        return results[:top_n]

    def _retrieve_context_for_company_interactive(self, retriever, company_name: str, question: str) -> List[Dict]:
        """交互问答优先提升召回覆盖率，因此会使用更激进的查询扩展。"""
        if self.full_context:
            return retriever.retrieve_all(company_name)

        interactive_sample_size = max(self.llm_reranking_sample_size, 20)
        interactive_top_n = max(self.top_n_retrieval, 12)
        query_variants = self._build_interactive_queries(company_name, question)
        prefer_table_chunks = self._is_table_value_question(question)
        return_parent_pages = self.return_parent_pages and not prefer_table_chunks
        result_groups = []

        for variant in query_variants:
            results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=variant,
                llm_reranking_sample_size=interactive_sample_size,
                top_n=interactive_top_n,
                return_parent_pages=return_parent_pages,
            )
            if results:
                result_groups.append(results)

        return self._merge_retrieval_results(result_groups, interactive_top_n, prefer_table_chunks=prefer_table_chunks)

    def _resolve_schema(
        self,
        question_text: str,
        schema: Optional[str],
        extracted_companies: Optional[List[str]] = None,
    ) -> str:
        """统一处理显式 schema 与自动推断的回退逻辑。"""
        if schema and schema not in {"auto", "general"}:
            return schema
        return self.infer_schema_for_question(question_text, extracted_companies=extracted_companies)


    @staticmethod
    def _clean_field_value(value: str) -> Optional[str]:
        """清洗基础字段抽取结果，去掉表格分隔符和明显的表头噪声。"""
        value = (value or "").strip()
        if not value:
            return None

        value = re.sub(r"^[|｜:：\-\s]+", "", value)
        value = re.sub(r"[|｜\s]+$", "", value)
        value = value.strip(" ：:；;，,。 ")

        invalid_values = {
            "股票代码",
            "股票简称",
            "变更前股票代码",
            "变更前股票简称",
            "法定代表人",
            "公司法定代表人",
            "公司网址",
            "公司网站",
            "网址",
            "网站",
            "电子邮箱",
            "电子信箱",
            "办公地址",
            "注册地址",
        }
        if not value or value in invalid_values:
            return None
        return value

    @classmethod
    def _extract_adjacent_table_value(cls, line: str, labels: List[str]) -> Optional[str]:
        """处理表格行，优先抽取字段右侧相邻单元格的值。"""
        cells = [cell.strip() for cell in re.split(r"[|｜]", line)]
        if len(cells) < 2:
            return None

        for index, cell in enumerate(cells):
            if cell in labels:
                for candidate in cells[index + 1:]:
                    cleaned = cls._clean_field_value(candidate)
                    if cleaned is not None:
                        return cleaned
        return None

    def _match_field(self, text: str, labels: List[str]) -> Optional[str]:
        """兼容普通文本和表格文本的基础字段抽取。"""
        for line in re.split(r"[\n\r]+", text or ""):
            if not any(label in line for label in labels):
                continue

            table_value = self._extract_adjacent_table_value(line, labels)
            if table_value is not None:
                return table_value

        for label in labels:
            pattern = rf"{re.escape(label)}[\s:：]*([^\n\r\t；;，,|｜]+)"
            match = re.search(pattern, text or "")
            if not match:
                continue

            value = self._clean_field_value(match.group(1))
            if value is not None:
                return value
        return None

    def _try_basic_info_shortcut(self, company_name: str, question: str) -> Optional[dict]:
        doc = self._get_document_by_company_name(company_name)
        if doc is None:
            return None

        field_map = {
            "法定代表人": ["法定代表人", "公司法定代表人"],
            "公司网址": ["公司网址", "公司网站", "网址", "互联网地址", "国际互联网网址"],
            "电子邮箱": ["电子邮箱", "电子信箱", "电子邮件", "联系邮箱", "投资者关系邮箱"],
            "股票简称": ["股票简称"],
            "股票代码": ["股票代码"],
            "办公地址": ["办公地址"],
            "注册地址": ["注册地址"],
        }

        target_field = None
        for field_name in field_map:
            short_name = field_name.replace("公司", "")
            if field_name in question or short_name in question:
                target_field = field_name
                break

        if target_field is None:
            if "网址" in question or "网站" in question:
                target_field = "公司网址"
            elif "邮箱" in question or "信箱" in question or "邮件" in question:
                target_field = "电子邮箱"

        if target_field is None:
            return None

        pages = doc["content"]["pages"]
        candidate_pages = list(pages[:15])

        for page in pages:
            if any(label in page["text"] for label in field_map[target_field]):
                if page not in candidate_pages:
                    candidate_pages.append(page)

        for page in candidate_pages:
            value = self._match_field(page["text"], field_map[target_field])
            if value:
                return {
                    "step_by_step_analysis": (
                        f"1. 问题属于基础信息字段抽取。2. 我先定位“{target_field}”可能出现的页面。"
                        f"3. 在第 {page['page']} 页发现与该字段直接对应的文本。4. 字段名与问题一致，没有使用相近字段替代。"
                        f"5. 因此可以直接抽取该字段值作为答案。"
                    ),
                    "reasoning_summary": f"第 {page['page']} 页存在“{target_field}”的直接字段证据，可直接抽取答案。",
                    "relevant_pages": [page["page"]],
                    "final_answer": value,
                    "references": self._extract_references([page["page"]], company_name),
                }

        return None

    @staticmethod
    def _resolve_numeric_unit_multiplier(local_text: str, target_field: str) -> float:
        """根据字段类型和局部文本判断单位倍率，避免对每股指标误乘页面单位。"""
        if target_field == "基本每股收益":
            return 1.0

        if "亿元" in local_text:
            return 100000000.0
        if "万元" in local_text:
            return 10000.0
        if "千元" in local_text:
            return 1000.0
        return 1.0

    def _extract_numeric_from_text(self, text: str, labels: List[str], target_field: str) -> Optional[float]:
        """尽量在字段附近抽取数值，并只按局部单位做换算。"""
        for label in labels:
            pattern = rf"([^\n\r]*{re.escape(label)}[^\n\r]*)"
            match = re.search(pattern, text or "")
            if not match:
                continue

            local_line = match.group(1)
            number_match = re.search(r"(-?\(?[\d,]+(?:\.\d+)?\)?)", local_line)
            if not number_match:
                continue

            raw = number_match.group(1).replace(",", "").strip()
            negative = raw.startswith("(") and raw.endswith(")")
            raw = raw.strip("()")

            try:
                value = float(raw)
            except ValueError:
                continue

            # 只在字段附近和页眉单位提示中判断倍率，避免被页面内其他表格单位误导。
            unit_window_start = max(0, match.start() - 160)
            unit_window_end = min(len(text or ""), match.end() + 80)
            local_unit_text = (text or "")[unit_window_start:unit_window_end]
            value *= self._resolve_numeric_unit_multiplier(local_unit_text, target_field)

            if negative:
                value = -value

            return int(value) if value.is_integer() else value

        return None

    def _try_numeric_shortcut(self, company_name: str, question: str) -> Optional[dict]:
        doc = self._get_document_by_company_name(company_name)
        if doc is None:
            return None

        field_map = {
            "营业收入": ["营业收入"],
            "归属于上市公司股东的净利润": ["归属于上市公司股东的净利润", "归母净利润"],
            "研发投入": ["研发投入"],
            "基本每股收益": ["基本每股收益"],
        }

        target_field = next((key for key in field_map if key in question), None)
        if target_field is None:
            return None

        for page in doc["content"]["pages"][:20]:
            value = self._extract_numeric_from_text(page["text"], field_map[target_field], target_field)
            if value is None:
                continue

            return {
                "step_by_step_analysis": (
                    f"1. 问题属于数值字段抽取。2. 我先定位“{target_field}”在财务指标页中的位置。"
                    f"3. 在第 {page['page']} 页找到与该字段直接对应的数值。4. 根据页面中的单位信息完成必要换算。"
                    f"5. 因此可以直接返回该数值。"
                ),
                "reasoning_summary": f"第 {page['page']} 页直接给出了“{target_field}”数值，可按页面单位提取。",
                "relevant_pages": [page["page"]],
                "final_answer": value,
                "references": self._extract_references([page["page"]], company_name),
            }

        return None

    @staticmethod
    def _set_answer_to_na(answer_dict: dict, reason: str) -> dict:
        """将不可信的回答统一降级为 N/A，避免脏答案继续污染比较题和最终输出。"""
        answer_dict = dict(answer_dict or {})
        answer_dict["answer_type"] = "na"
        answer_dict["final_answer"] = "N/A"
        answer_dict["answer_unit"] = None
        answer_dict["unit_basis"] = None
        summary = str(answer_dict.get("reasoning_summary") or "").strip()
        answer_dict["reasoning_summary"] = reason if not summary else f"{summary}；{reason}"
        return answer_dict

    @staticmethod
    def _is_money_metric(question: str) -> bool:
        """判断问题是否属于金额类指标，便于统一换算为元。"""
        money_keywords = [
            "营业收入", "净利润", "归属于上市公司股东的净利润", "归母净利润", "研发投入",
            "资产", "负债", "现金流", "账面价值", "余额", "金额", "数额", "总额", "总计",
        ]
        question = question or ""
        return any(keyword in question for keyword in money_keywords)

    @staticmethod
    def _is_ratio_or_per_share_metric(question: str) -> bool:
        """判断是否属于比例类或每股类指标，这类指标不应被统一换算为元。"""
        ratio_keywords = [
            "每股收益", "基本每股收益", "稀释每股收益", "比例", "占比", "增长率",
            "毛利率", "净利率", "资产负债率", "同比", "百分比", "百分点",
        ]
        question = question or ""
        return any(keyword in question for keyword in ratio_keywords)

    @staticmethod
    def _normalize_answer_unit(answer_unit) -> Optional[str]:
        """统一单位写法，减少后续单位比较时的分支数量。"""
        if answer_unit is None:
            return None

        text = str(answer_unit).strip()
        if not text or text.upper() == "N/A":
            return None

        text = text.replace("人民币", "").replace(" ", "")
        text = text.rstrip("。；;，,）)")
        aliases = {
            "元人民币": "元",
            "万元人民币": "万元",
            "亿元人民币": "亿元",
            "千元人民币": "千元",
            "百分比": "%",
            "百分数": "%",
            "百分点数": "百分点",
        }
        return aliases.get(text, text)

    @staticmethod
    def _extract_first_company_name(text: str) -> Optional[str]:
        """从文本中提取一个较短的中文人名或公司名片段，用于字段题脏值清洗。"""
        candidates = re.findall(r"[\u4e00-\u9fff]{2,6}", text or "")
        invalid_fragments = {
            "主管会计工作负责人", "会计机构负责人", "财务报表", "公司负责人",
            "单位负责人", "负责人", "签名并盖章", "报告期内",
        }
        for candidate in candidates:
            if candidate in invalid_fragments:
                continue
            if "负责人" in candidate or "报表" in candidate or "签名" in candidate:
                continue
            return candidate
        return None

    def _normalize_numeric_text_answer_strict(self, question: str, raw_value: str) -> Optional[dict]:
        """更严格地将文本数值答案规范化为单个可比较数值。"""
        text = (raw_value or "").strip()
        if not text or text == "N/A":
            return None

        if len(re.findall(r"20\d{2}", text)) >= 2:
            return None

        matches = list(
            re.finditer(r"(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|千元|元/股|元|%|百分点|个百分点|个)?", text)
        )
        if len(matches) != 1:
            return None

        match = matches[0]
        number_text = match.group(1).replace(",", "")
        try:
            value = float(number_text)
        except ValueError:
            return None

        answer_dict = {
            "final_answer": value,
            "answer_type": "number",
            "answer_unit": self._normalize_answer_unit(match.group(2) or None),
            "unit_basis": None,
        }
        return self._normalize_numeric_answer_strict(question, answer_dict)

    def _normalize_numeric_answer_strict(self, question: str, answer_dict: dict) -> Optional[dict]:
        """统一校验数值题的值与单位，确保单值、单位可靠且便于后续比较。"""
        final_answer = answer_dict.get("final_answer")
        if isinstance(final_answer, bool) or isinstance(final_answer, list):
            return None

        try:
            value = float(final_answer)
        except (TypeError, ValueError):
            return None

        answer_unit = self._normalize_answer_unit(answer_dict.get("answer_unit"))
        unit_basis = str(answer_dict.get("unit_basis") or "").strip() or None

        if self._is_money_metric(question):
            if answer_unit == "亿元":
                value *= 100000000
                answer_unit = "元"
                unit_basis = unit_basis or "原文单位为亿元，已换算为元"
            elif answer_unit == "万元":
                value *= 10000
                answer_unit = "元"
                unit_basis = unit_basis or "原文单位为万元，已换算为元"
            elif answer_unit == "千元":
                value *= 1000
                answer_unit = "元"
                unit_basis = unit_basis or "原文单位为千元，已换算为元"
            elif answer_unit in {"元", None}:
                if answer_unit == "元" and not unit_basis:
                    unit_basis = "原文单位为元"
            else:
                return None
        else:
            if self._is_ratio_or_per_share_metric(question) and answer_unit in {"亿元", "万元", "千元"}:
                return None
            if answer_unit == "个百分点":
                answer_unit = "百分点"
            if answer_unit and not unit_basis:
                unit_basis = f"原文单位为{answer_unit}"

        if float(value).is_integer():
            value = int(value)

        return {
            "final_answer": value,
            "answer_type": "number",
            "answer_unit": answer_unit,
            "unit_basis": unit_basis,
        }

    def _normalize_name_answer_strict(self, question: str, final_answer) -> Optional[str]:
        """更严格地校验字段题，过滤表头、签名栏和格式不合法的脏值。"""
        if final_answer is None:
            return None

        text = str(final_answer).strip()
        if not text or text == "N/A":
            return None

        text = re.sub(r"\s+", " ", text).strip(" ：:|;；，,。.\"'“”‘’")
        if text in {"股票代码", "股票简称", "公司网址", "法定代表人", "注册地址", "办公地址", "电子信箱", "电子邮箱"}:
            return None

        if "股票代码" in question:
            match = re.search(r"\b\d{6}\b", text)
            return match.group(0) if match else None

        if "网址" in question or "网站" in question:
            match = re.search(r"(https?://[^\s]+|www\.[^\s]+)", text, flags=re.IGNORECASE)
            return match.group(1).rstrip("。；;，,") if match else None

        if "邮箱" in question or "电子信箱" in question:
            match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
            return match.group(0) if match else None

        if "法定代表人" in question:
            if any(token in text for token in ["负责人", "签名", "财务报表", "会计机构", "主管会计", "董事会"]):
                candidate = self._extract_first_company_name(text)
                return candidate if candidate and len(candidate) <= 4 else None
            candidate = self._extract_first_company_name(text)
            if candidate and len(candidate) <= 4:
                return candidate
            if re.fullmatch(r"[\u4e00-\u9fff·]{2,8}", text):
                return text
            return None

        if "地址" in question:
            if len(text) < 6 or any(token in text for token in ["签名", "财务报表", "负责人"]):
                return None
            return text

        if len(text) > 80:
            return None
        return text

    def _get_numeric_comparison_unit(self, question: str, answer_dict: dict) -> Optional[str]:
        """提取可用于程序比较的标准单位键，不可比较时返回 None。"""
        final_answer = answer_dict.get("final_answer")
        if not isinstance(final_answer, (int, float)) or isinstance(final_answer, bool):
            return None

        answer_unit = self._normalize_answer_unit(answer_dict.get("answer_unit"))
        if self._is_money_metric(question):
            return "元" if answer_unit == "元" else None

        if answer_unit is None:
            return "unitless"
        if self._is_ratio_or_per_share_metric(question):
            return answer_unit if answer_unit in {"元/股", "%", "百分点", "unitless"} else None
        return answer_unit

    def _normalize_numeric_text_answer(self, question: str, raw_value: str) -> Optional[dict]:
        """将模型返回的文本数值答案规范化为单个数值，并尽量补充单位信息。"""
        text = (raw_value or "").strip()
        if not text or text == "N/A":
            return None

        # 同时出现多个年份时，通常表示模型返回了多期文本，不再冒险自动抽取。
        if len(re.findall(r"20\d{2}", text)) >= 2:
            return None

        matches = list(
            re.finditer(r"(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|千元|元/股|元|%|个百分点|个)?", text)
        )
        if len(matches) != 1:
            return None

        match = matches[0]
        number_text = match.group(1).replace(",", "")
        unit_text = match.group(2) or ""
        try:
            value = float(number_text)
        except ValueError:
            return None

        unit_basis = None
        answer_unit = None
        if self._is_money_metric(question):
            if unit_text == "亿元":
                value *= 100000000
                answer_unit = "元"
                unit_basis = "原文单位为亿元，已换算为元"
            elif unit_text == "万元":
                value *= 10000
                answer_unit = "元"
                unit_basis = "原文单位为万元，已换算为元"
            elif unit_text == "千元":
                value *= 1000
                answer_unit = "元"
                unit_basis = "原文单位为千元，已换算为元"
            else:
                answer_unit = "元" if unit_text in {"元", ""} else unit_text
                if unit_text == "元":
                    unit_basis = "原文单位为元"
        else:
            answer_unit = unit_text or None
            if unit_text:
                unit_basis = f"原文单位为{unit_text}"

        if value.is_integer():
            value = int(value)

        return {
            "final_answer": value,
            "answer_type": "number",
            "answer_unit": answer_unit,
            "unit_basis": unit_basis,
        }

    def _normalize_name_answer(self, question: str, final_answer) -> Optional[str]:
        """对字段题做轻量规则校验，过滤明显的签名栏、表头和脏文本。"""
        if final_answer is None:
            return None

        text = str(final_answer).strip()
        if not text or text == "N/A":
            return None

        if "股票代码" in question:
            match = re.search(r"\b\d{6}\b", text)
            return match.group(0) if match else None

        if "网址" in question or "网站" in question:
            match = re.search(r"(https?://[^\s]+|www\.[^\s]+)", text, flags=re.IGNORECASE)
            return match.group(1).rstrip("。；;，,") if match else None

        if "法定代表人" in question:
            if any(token in text for token in ["负责人", "签名", "财务报表", "会计机构", "主管会计"]):
                return self._extract_first_company_name(text)
            candidate = self._extract_first_company_name(text)
            return candidate or text

        return text

    def _normalize_boolean_answer(self, final_answer):
        """将布尔题返回结果规整为 True/False/N/A。"""
        if isinstance(final_answer, bool):
            return final_answer

        text = str(final_answer or "").strip().lower()
        if not text or text == "n/a":
            return "N/A"

        positive_tokens = ["true", "是", "已披露", "披露了", "存在", "有", "已实施", "已回购"]
        negative_tokens = ["false", "否", "未披露", "不存在", "没有", "无", "未实施", "未回购", "不适用"]

        if any(token in text for token in positive_tokens) and not any(token in text for token in negative_tokens):
            return True
        if any(token in text for token in negative_tokens):
            return False
        return "N/A"

    def _validate_answer_dict(self, question: str, schema: str, answer_dict: dict) -> dict:
        """统一答案校验层：对不同题型做最小必要校验与归一化。"""
        answer_dict = dict(answer_dict or {})
        final_answer = answer_dict.get("final_answer")

        if final_answer == "N/A" or final_answer is None:
            answer_dict["answer_type"] = "na"
            answer_dict["answer_unit"] = None
            answer_dict["unit_basis"] = None
            return answer_dict

        if schema == "number":
            if isinstance(final_answer, bool) or isinstance(final_answer, list):
                return self._set_answer_to_na(answer_dict, "数值题返回结果类型不合法，已降级为 N/A")

            if isinstance(final_answer, str):
                normalized = self._normalize_numeric_text_answer_strict(question, final_answer)
            else:
                normalized = self._normalize_numeric_answer_strict(question, answer_dict)
            if normalized is None:
                return self._set_answer_to_na(answer_dict, "数值题未能稳定归一化为单个数值，已降级为 N/A")
            answer_dict.update(normalized)
            return answer_dict

        if schema == "name":
            normalized_name = self._normalize_name_answer_strict(question, final_answer)
            if normalized_name is None:
                return self._set_answer_to_na(answer_dict, "字段题返回值疑似表头或签名栏噪声，已降级为 N/A")
            answer_dict["final_answer"] = normalized_name
            answer_dict["answer_type"] = "text"
            return answer_dict

        if schema == "boolean":
            normalized_boolean = self._normalize_boolean_answer(final_answer)
            if normalized_boolean == "N/A":
                return self._set_answer_to_na(answer_dict, "布尔题结果无法稳定判定真值，已降级为 N/A")
            answer_dict["final_answer"] = normalized_boolean
            answer_dict["answer_type"] = "boolean"
            return answer_dict

        if schema == "names":
            if isinstance(final_answer, list):
                cleaned = [str(item).strip() for item in final_answer if str(item).strip()]
                answer_dict["final_answer"] = cleaned if cleaned else "N/A"
                answer_dict["answer_type"] = "list" if cleaned else "na"
                return answer_dict
            return self._set_answer_to_na(answer_dict, "列表题未返回列表结构，已降级为 N/A")

        return answer_dict

    def get_answer_for_company(self, company_name: str, question: str, schema: str) -> dict:
        # 先尝试规则捷径，命中后可以直接绕开高成本的检索与问答调用。
        if self.enable_rule_shortcuts and schema == "name":
            shortcut = self._try_basic_info_shortcut(company_name, question)
            if shortcut is not None:
                shortcut = self._validate_answer_dict(question, schema, shortcut)
                shortcut["route"] = "rule"
                shortcut["status"] = "answered" if shortcut.get("final_answer") != "N/A" else "insufficient_evidence"
                shortcut["resolved_company_names"] = [company_name]
                return shortcut

        if self.enable_rule_shortcuts and schema == "number":
            shortcut = self._try_numeric_shortcut(company_name, question)
            if shortcut is not None:
                shortcut = self._validate_answer_dict(question, schema, shortcut)
                shortcut["route"] = "rule"
                shortcut["status"] = "answered" if shortcut.get("final_answer") != "N/A" else "insufficient_evidence"
                shortcut["resolved_company_names"] = [company_name]
                return shortcut

        retriever = self._build_retriever()

        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages,
            )

        if not retrieval_results:
            raise ValueError("No relevant context found")

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_interactive_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            companies=[company_name],
            model=self.answering_model,
            question_kind_hint=schema,
        )
        answer_dict = self._validate_answer_dict(question, schema, answer_dict)
        self.response_data = self.openai_processor.response_data

        validated_pages = self._validate_page_references(answer_dict.get("relevant_pages", []), retrieval_results)
        answer_dict["relevant_pages"] = validated_pages
        answer_dict["references"] = self._extract_references(validated_pages, company_name)
        answer_dict["route"] = "rag"
        answer_dict["status"] = "answered" if answer_dict.get("final_answer") != "N/A" else "insufficient_evidence"
        answer_dict["resolved_company_names"] = [company_name]
        return answer_dict

    def process_question(self, question: str, schema: str):
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)

        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        if len(extracted_companies) == 1:
            return self.get_answer_for_company(extracted_companies[0], question, schema)
        return self.process_comparative_question(question, extracted_companies, schema)

    def _infer_schema_with_rules(self, question: str) -> str:
        """基于稳定关键词的兜底题型判断。"""
        question = (question or '').strip()
        if not question:
            raise ValueError('Question cannot be empty.')

        normalized_question = self._normalize_company_text(question)

        names_keywords = [
            '都有谁', '分别是谁', '哪些人员', '哪些人', '名单', '成员', '董事', '监事',
            '高级管理人员', '管理人员', '管理层', '核心技术人员', '主要客户', '主要供应商',
            '前五名客户', '前五名供应商', '有哪些', '列出',
        ]
        boolean_keywords = ['是否', '有没有', '有无', '是不是', '是否存在', '是否披露', '披露了', '披露过']
        number_keywords = [
            '多少', '金额', '数额', '总额', '总计', '账面价值', '期末账面价值', '期末余额',
            '期末价值', '营业收入', '净利润', '归母净利润', '每股收益', '研发投入', '资产', '负债',
            '现金流', '净额', '占比', '比例', '增长率', '下降', '增加', '减少',
        ]
        name_keywords = ['谁', '哪家', '哪个公司', '法定代表人', '公司网址', '网址', '网站', '股票代码', '股票简称', '办公地址', '注册地址', '邮箱']

        if any(keyword in question for keyword in names_keywords):
            return 'names'

        if any(keyword in question for keyword in boolean_keywords):
            return 'boolean'

        if any(keyword in question for keyword in number_keywords):
            return 'number'

        if any(keyword in question for keyword in name_keywords):
            return 'name'

        if '更高' in question or '更低' in question or '最高' in question or '最低' in question:
            return 'name'

        if any(keyword in normalized_question for keyword in ['年报', '报告', '情况']):
            return 'number'

        return 'name'

    def _classify_question_kind_with_model(
        self,
        question: str,
        extracted_companies: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """使用轻量模型做题型分类，低成本地产生 question_kind_hint。"""
        if not self.question_kind_model:
            return None

        cache_key = (question, tuple(extracted_companies or []))
        if cache_key in self._question_kind_cache:
            return self._question_kind_cache[cache_key]

        try:
            classification = self.openai_processor.classify_question_kind(
                question=question,
                companies=extracted_companies or [],
                model=self.question_kind_model,
            )
        except Exception:
            return None

        result = {
            "predicted_kind": classification.get("predicted_kind"),
            "confidence": float(classification.get("confidence", 0.0) or 0.0),
            "reasoning_summary": classification.get("reasoning_summary", ""),
        }
        self._question_kind_cache[cache_key] = result
        return result

    def infer_schema_for_question(self, question: str, extracted_companies: Optional[List[str]] = None) -> str:
        """优先使用轻量模型做题型分类，低置信度时回退到关键词规则。"""
        question = (question or '').strip()
        if not question:
            raise ValueError('Question cannot be empty.')

        extracted_companies = extracted_companies or []
        if len(extracted_companies) > 1:
            return 'comparative'

        classification = self._classify_question_kind_with_model(
            question=question,
            extracted_companies=extracted_companies,
        )
        if classification is not None:
            predicted_kind = classification.get("predicted_kind")
            confidence = classification.get("confidence", 0.0)
            if predicted_kind and confidence >= self.question_kind_confidence_threshold:
                return predicted_kind

        return self._infer_schema_with_rules(question)

    def answer_single_question(self, question_text: str, schema: Optional[str] = None) -> dict:
        # 交互问答默认保留查询扩展；批处理会显式关闭该模式，避免调用量失控。
        resolved_companies = self._extract_companies_from_subset(question_text) if self.new_challenge_pipeline else []
        resolved_kind = self._resolve_schema(question_text, schema, extracted_companies=resolved_companies)

        if not resolved_companies:
            return {
                'question_text': question_text,
                'resolved_kind': resolved_kind,
                'resolved_companies': [],
                'route': 'unknown',
                'status': 'company_not_matched',
                'value': None,
                'error': 'No company name found in the question.',
            }

        if len(resolved_companies) > 1:
            comparative_answer = self.process_comparative_question(question_text, resolved_companies, schema='comparative')
            detail = {
                'step_by_step_analysis': comparative_answer.get('step_by_step_analysis'),
                'reasoning_summary': comparative_answer.get('reasoning_summary'),
                'relevant_pages': comparative_answer.get('relevant_pages'),
                'response_data': getattr(self, 'response_data', None),
            }
            return {
                'question_text': question_text,
                'resolved_kind': 'comparative',
                'resolved_companies': resolved_companies,
                'route': comparative_answer.get('route', 'comparative'),
                'status': comparative_answer.get('status', 'answered' if comparative_answer.get('final_answer') != 'N/A' else 'insufficient_evidence'),
                'value': comparative_answer.get('final_answer'),
                'references': comparative_answer.get('references', []),
                'detail': detail,
            }

        company_name = resolved_companies[0]
        return self._answer_single_company_question(
            company_name=company_name,
            question_text=question_text,
            resolved_kind=resolved_kind,
            use_interactive_retrieval=True,
        )

    def _answer_single_company_question(
        self,
        company_name: str,
        question_text: str,
        resolved_kind: str,
        use_interactive_retrieval: bool,
    ) -> dict:
        """统一封装单公司问题回答，便于批处理与交互链共享主逻辑。"""
        if not use_interactive_retrieval:
            answer_dict = self.get_answer_for_company(company_name, question_text, resolved_kind)
            return {
                'question_text': question_text,
                'resolved_kind': answer_dict.get('answer_type', resolved_kind),
                'resolved_companies': [company_name],
                'route': answer_dict.get('route'),
                'status': answer_dict.get('status'),
                'value': answer_dict.get('final_answer'),
                'answer_unit': answer_dict.get('answer_unit'),
                'unit_basis': answer_dict.get('unit_basis'),
                'references': answer_dict.get('references', []),
                'detail': {
                    'step_by_step_analysis': answer_dict.get('step_by_step_analysis'),
                    'reasoning_summary': answer_dict.get('reasoning_summary'),
                    'relevant_pages': answer_dict.get('relevant_pages', []),
                    'answer_unit': answer_dict.get('answer_unit'),
                    'unit_basis': answer_dict.get('unit_basis'),
                    'response_data': getattr(self, 'response_data', None),
                },
            }

        retriever = self._build_retriever()
        retrieval_results = self._retrieve_context_for_company_interactive(retriever, company_name, question_text)
        if not retrieval_results:
            return {
                'question_text': question_text,
                'resolved_kind': resolved_kind,
                'resolved_companies': [company_name],
                'route': 'rag',
                'status': 'insufficient_evidence',
                'value': 'N/A',
                'answer_unit': None,
                'unit_basis': None,
                'references': [],
                'detail': {
                    'reasoning_summary': '未检索到可直接支持答案的相关上下文。',
                    'relevant_pages': [],
                    'answer_unit': None,
                    'unit_basis': None,
                },
            }

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_interactive_answer_from_rag_context(
            question=question_text,
            rag_context=rag_context,
            companies=[company_name],
            model=self.answering_model,
            question_kind_hint=resolved_kind,
        )
        answer_dict = self._validate_answer_dict(question_text, resolved_kind, answer_dict)
        self.response_data = self.openai_processor.response_data

        validated_pages = self._validate_page_references(answer_dict.get('relevant_pages', []), retrieval_results)
        references = self._extract_references(validated_pages, company_name)
        final_answer = answer_dict.get('final_answer')
        status = 'answered' if final_answer != 'N/A' else 'insufficient_evidence'

        return {
            'question_text': question_text,
            'resolved_kind': answer_dict.get('answer_type', resolved_kind),
            'resolved_companies': [company_name],
            'route': 'rag',
            'status': status,
            'value': final_answer,
            'answer_unit': answer_dict.get('answer_unit'),
            'unit_basis': answer_dict.get('unit_basis'),
            'references': references,
            'detail': {
                'step_by_step_analysis': answer_dict.get('step_by_step_analysis'),
                'reasoning_summary': answer_dict.get('reasoning_summary'),
                'relevant_pages': validated_pages,
                'answer_unit': answer_dict.get('answer_unit'),
                'unit_basis': answer_dict.get('unit_basis'),
                'response_data': self.response_data,
            },
        }

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict.get("step_by_step_analysis"),
                "reasoning_summary": answer_dict.get("reasoning_summary"),
                "relevant_pages": answer_dict.get("relevant_pages"),
                "answer_unit": answer_dict.get("answer_unit"),
                "unit_basis": answer_dict.get("unit_basis"),
                "response_data": getattr(self, "response_data", None),
                "self": ref_id,
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count

        if print_stats and total_questions:
            print("\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count / total_questions) * 100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count / total_questions) * 100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count / total_questions) * 100:.1f}%)\n")

        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count,
        }

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        import traceback

        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"

        with self._lock:
            self.answer_details[question_index] = {"error_traceback": tb, "self": error_ref}

        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")

        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

        return {
            "question": question_text,
            "schema": schema,
            "answer": None,
            "error": f"{type(err).__name__}: {error_message}",
            "answer_details": {"$ref": error_ref},
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        question_text = question_data.get("text") if self.new_challenge_pipeline else question_data.get("question")
        schema = question_data.get("kind") if self.new_challenge_pipeline else question_data.get("schema")

        try:
            if self.new_challenge_pipeline:
                resolved_companies = self._extract_companies_from_subset(question_text)
                if len(resolved_companies) == 1:
                    resolved_kind = self._resolve_schema(question_text, schema, extracted_companies=resolved_companies)
                    answer_dict = self._answer_single_company_question(
                        company_name=resolved_companies[0],
                        question_text=question_text,
                        resolved_kind=resolved_kind,
                        use_interactive_retrieval=False,
                    )
                else:
                    answer_dict = self.answer_single_question(question_text, schema=schema)
                if answer_dict.get("error"):
                    raise ValueError(answer_dict["error"])
                detail_ref = self._create_answer_detail_ref(answer_dict.get("detail", {}), question_index)
                return {
                    "question_text": question_text,
                    "kind": answer_dict.get("resolved_kind", schema),
                    "value": answer_dict.get("value"),
                    "answer_unit": answer_dict.get("answer_unit"),
                    "unit_basis": answer_dict.get("unit_basis"),
                    "references": answer_dict.get("references", []),
                    "route": answer_dict.get("route"),
                    "status": answer_dict.get("status"),
                    "resolved_company_names": answer_dict.get("resolved_companies", []),
                    "answer_details": {"$ref": detail_ref},
                }

            answer_dict = self.process_question(question_text, schema)
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            return {
                "question": question_text,
                "schema": schema,
                "answer": answer_dict.get("final_answer"),
                "route": answer_dict.get("route"),
                "status": answer_dict.get("status"),
                "resolved_company_names": answer_dict.get("resolved_company_names", []),
                "answer_details": {"$ref": detail_ref},
            }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def process_questions_list(
        self,
        questions_list: List[dict],
        output_path: str = None,
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ) -> dict:
        total_questions = len(questions_list)
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions
        processed_questions = []

        with tqdm(total=total_questions, desc="Processing questions") as pbar:
            for i in range(0, total_questions, max(1, self.parallel_requests)):
                batch = questions_with_index[i:i + max(1, self.parallel_requests)]
                # 批处理层面额外限制并发，避免与检索/重排内部线程叠加后瞬时过载。
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self.parallel_requests)) as executor:
                    batch_results = list(executor.map(self._process_single_question, batch))
                processed_questions.extend(batch_results)
                if output_path:
                    self._save_progress(
                        processed_questions,
                        output_path,
                        submission_file=submission_file,
                        team_email=team_email,
                        submission_name=submission_name,
                        pipeline_details=pipeline_details,
                    )
                pbar.update(len(batch_results))

        statistics = self._calculate_statistics(processed_questions, print_stats=True)
        return {"questions": processed_questions, "answer_details": self.answer_details, "statistics": statistics}

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        submission_answers = []
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = [] if value == "N/A" else [
                {"pdf_sha1": ref["pdf_sha1"], "page_index": ref["page_index"] - 1}
                for ref in q.get("references", [])
            ]

            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            reasoning_process = None
            if answer_details_ref.startswith("#/answer_details/"):
                index = int(answer_details_ref.split("/")[-1])
                if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                    reasoning_process = self.answer_details[index].get("step_by_step_analysis")

            item = {"question_text": question_text, "kind": kind, "value": value, "references": references}
            if reasoning_process:
                item["reasoning_process"] = reasoning_process
            submission_answers.append(item)
        return submission_answers

    def _save_progress(
        self,
        processed_questions: List[dict],
        output_path: Optional[str],
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ):
        if not output_path:
            return

        statistics = self._calculate_statistics(processed_questions)
        result = {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics,
        }
        output_file = Path(output_path)
        debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)

        with open(debug_file, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        if submission_file:
            submission = {
                "answers": self._post_process_submission_answers(processed_questions),
                "team_email": team_email,
                "submission_name": submission_name,
                "details": pipeline_details,
            }
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(
        self,
        output_path: str = "questions_with_answers.json",
        team_email: str = "",
        submission_name: str = "",
        submission_file: bool = False,
        pipeline_details: str = "",
    ):
        return self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details,
        )

    @staticmethod
    def _detect_comparative_mode(question: str) -> str:
        """识别比较题的可程序化类型。"""
        question = question or ""
        if "哪些公司同时" in question or "哪家公司同时" in question or "同时披露了" in question:
            return "boolean_all_of"

        numeric_compare_keywords = ["更高", "更低", "最高", "最低", "更大", "更小", "更多", "更少"]
        if "哪家" in question or any(keyword in question for keyword in numeric_compare_keywords):
            return "numeric_rank"

        return "llm"

    @staticmethod
    def _is_higher_better_question(question: str) -> bool:
        return any(keyword in (question or "") for keyword in ["更高", "最高", "更大", "更多", "更强"])

    @staticmethod
    def _is_lower_better_question(question: str) -> bool:
        return any(keyword in (question or "") for keyword in ["更低", "最低", "更小", "更少"])

    @staticmethod
    def _build_boolean_comparative_sub_question(target_company: str, question: str) -> str:
        """将“哪些公司同时……”类比较题改写为单公司的布尔问题。"""
        question = question or ""
        if "中，" in question:
            tail = question.split("中，", 1)[1]
        elif "中," in question:
            tail = question.split("中,", 1)[1]
        else:
            tail = question

        tail = re.sub(r"^哪些公司", f"{target_company}是否", tail)
        tail = re.sub(r"^哪家公司", f"{target_company}是否", tail)
        tail = tail.strip()
        if not tail.startswith(target_company):
            return f"{target_company}是否{tail}"
        return tail

    def _build_programmatic_numeric_comparative_answer(
        self,
        question: str,
        companies: List[str],
        individual_answers: Dict[str, dict],
        references: List[dict],
    ) -> dict:
        """对数值高低比较题优先使用程序比较，避免再让模型比较已经标准化的数值。"""
        comparable_items = []
        relevant_pages = []
        invalid_comparable_count = 0

        for company in companies:
            answer_dict = individual_answers.get(company, {})
            final_answer = answer_dict.get("final_answer")
            comparison_unit = self._get_numeric_comparison_unit(question, answer_dict)
            if isinstance(final_answer, bool) or not isinstance(final_answer, (int, float)):
                continue
            if comparison_unit is None:
                invalid_comparable_count += 1
                continue
            comparable_items.append((company, float(final_answer), answer_dict, comparison_unit))
            relevant_pages.extend(answer_dict.get("relevant_pages", []))

        if not comparable_items:
            return {
                "step_by_step_analysis": "1. 问题要求在多家公司之间比较数值。2. 逐一检查各公司的单公司答案。3. 当前没有足够的可比数值结果。4. 因此无法形成有效比较，只能返回 N/A。",
                "reasoning_summary": "可比数值不足，无法完成程序比较。",
                "relevant_pages": [],
                "answer_type": "na",
                "final_answer": "N/A",
                "answer_unit": None,
                "unit_basis": None,
                "references": references,
            }

        comparable_units = {item[3] for item in comparable_items}
        if invalid_comparable_count > 0 or len(comparable_units) != 1:
            return {
                "step_by_step_analysis": "1. 问题要求在多家公司之间比较同一指标。2. 我先检查各公司答案是否都是可比较的标准化数值。3. 当前可比较结果存在单位不一致或缺少标准单位。4. 为避免错误比较，本题返回 N/A。",
                "reasoning_summary": "可比较数值的单位口径不一致，无法安全完成程序比较。",
                "relevant_pages": sorted(set(relevant_pages))[:8],
                "answer_type": "na",
                "final_answer": "N/A",
                "answer_unit": None,
                "unit_basis": None,
                "references": references,
            }

        if self._is_lower_better_question(question):
            target_value = min(item[1] for item in comparable_items)
        else:
            target_value = max(item[1] for item in comparable_items)

        winners = [item for item in comparable_items if item[1] == target_value]
        if len(winners) != 1:
            return {
                "step_by_step_analysis": "1. 问题要求比较多家公司数值。2. 程序比较发现存在并列或无法唯一确定结果。3. 按保守策略不输出并列公司的主观选择。4. 因此返回 N/A。",
                "reasoning_summary": "比较结果存在并列，无法唯一确定答案。",
                "relevant_pages": sorted(set(relevant_pages))[:8],
                "answer_type": "na",
                "final_answer": "N/A",
                "answer_unit": None,
                "unit_basis": None,
                "references": references,
            }

        winner_company, winner_value, winner_answer, _ = winners[0]
        comparison_direction = "最低" if self._is_lower_better_question(question) else "最高"
        return {
            "step_by_step_analysis": (
                f"1. 问题要求比较多家公司数值并找出{comparison_direction}者。"
                f"2. 我先收集各公司的单公司标准化数值答案。"
                f"3. 排除 N/A 或非数值结果后，对剩余公司直接做程序比较。"
                f"4. 最终唯一结果为 {winner_company}，对应数值为 {winner_value}。"
            ),
            "reasoning_summary": f"程序比较可比数值后，{winner_company} 的结果为{comparison_direction}。",
            "relevant_pages": sorted(set(winner_answer.get("relevant_pages", [])))[:8],
            "answer_type": "text",
            "final_answer": winner_company,
            "answer_unit": None,
            "unit_basis": None,
            "references": references,
        }

    def _build_programmatic_boolean_comparative_answer(
        self,
        companies: List[str],
        individual_answers: Dict[str, dict],
        references: List[dict],
    ) -> dict:
        """对“哪些公司同时……”这类布尔筛选题优先使用程序汇总。"""
        matched_companies = []
        relevant_pages = []

        for company in companies:
            answer_dict = individual_answers.get(company, {})
            if answer_dict.get("final_answer") is True:
                matched_companies.append(company)
                relevant_pages.extend(answer_dict.get("relevant_pages", []))

        if not matched_companies:
            return {
                "step_by_step_analysis": "1. 问题要求筛选同时满足条件的公司。2. 我先查看各公司的单公司布尔判断结果。3. 当前没有公司被稳定判定为满足条件。4. 因此返回 N/A。",
                "reasoning_summary": "没有公司被明确判定为同时满足条件。",
                "relevant_pages": sorted(set(relevant_pages))[:8],
                "answer_type": "na",
                "final_answer": "N/A",
                "answer_unit": None,
                "unit_basis": None,
                "references": references,
            }

        return {
            "step_by_step_analysis": (
                "1. 问题要求筛选同时满足条件的公司。"
                "2. 我先将比较题改写为单公司的布尔问题。"
                "3. 再收集每家公司的 True/False 结果。"
                f"4. 最终筛选出满足条件的公司：{', '.join(matched_companies)}。"
            ),
            "reasoning_summary": f"程序汇总单公司布尔结果后，筛选出 {len(matched_companies)} 家符合条件的公司。",
            "relevant_pages": sorted(set(relevant_pages))[:8],
            "answer_type": "list",
            "final_answer": matched_companies,
            "answer_unit": None,
            "unit_basis": None,
            "references": references,
        }

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        individual_answers = {}
        aggregated_references = []
        comparative_mode = self._detect_comparative_mode(question)

        def build_company_sub_question(target_company: str) -> str:
            metric_text = question
            for company in sorted(companies, key=len, reverse=True):
                if company == target_company:
                    continue
                metric_text = metric_text.replace(company, "")

            metric_text = re.sub('[\u3001,\uFF0C\u548C\u53CA\u4E0E]\s*', '', metric_text)
            metric_text = re.sub('.*?(\u54EA\u5BB6|\u8C01\u7684)', '', metric_text)
            metric_text = re.sub('\u5982\u679C.*$', '', metric_text).strip()
            metric_text = re.sub('\u66F4\u9AD8|\u66F4\u4F4E|\u6700\u9AD8|\u6700\u4F4E|\u66F4\u5927|\u66F4\u5C0F|\u66F4\u5F3A|\u66F4\u591A|\u66F4\u5C11', '', metric_text)
            metric_text = metric_text.strip(' :\uFF1A,\uFF0C\u3002\uFF1B;\uFF1F? ')

            if not metric_text:
                return f"{target_company}的该指标是多少？"
            return f"{target_company}的{metric_text}是多少？"

        def process_company_question(company: str):
            if comparative_mode == "boolean_all_of":
                sub_question = self._build_boolean_comparative_sub_question(company, question)
                sub_schema = "boolean"
            else:
                sub_question = build_company_sub_question(company)
                sub_schema = "number"
            answer_dict = self.get_answer_for_company(company_name=company, question=sub_question, schema=sub_schema)
            return company, answer_dict

        comparative_workers = max(1, min(len(companies), 3))
        with concurrent.futures.ThreadPoolExecutor(max_workers=comparative_workers) as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company
                for company in companies
            }
            for future in concurrent.futures.as_completed(future_to_company):
                company, answer_dict = future.result()
                individual_answers[company] = answer_dict
                aggregated_references.extend(answer_dict.get("references", []))

        unique_refs = {}
        for ref in aggregated_references:
            unique_refs[(ref.get("pdf_sha1"), ref.get("page_index"))] = ref
        unique_references = list(unique_refs.values())

        if comparative_mode == "numeric_rank":
            comparative_answer = self._build_programmatic_numeric_comparative_answer(
                question=question,
                companies=companies,
                individual_answers=individual_answers,
                references=unique_references,
            )
            comparative_answer["route"] = "comparative_programmatic"
            comparative_answer["status"] = "answered" if comparative_answer.get("final_answer") != "N/A" else "insufficient_evidence"
            comparative_answer["resolved_company_names"] = companies
            return comparative_answer

        if comparative_mode == "boolean_all_of":
            comparative_answer = self._build_programmatic_boolean_comparative_answer(
                companies=companies,
                individual_answers=individual_answers,
                references=unique_references,
            )
            comparative_answer["route"] = "comparative_programmatic"
            comparative_answer["status"] = "answered" if comparative_answer.get("final_answer") != "N/A" else "insufficient_evidence"
            comparative_answer["resolved_company_names"] = companies
            return comparative_answer

        comparison_context_parts = []
        for company in companies:
            answer_dict = individual_answers.get(company, {})
            comparison_context_parts.append(
                f"公司：{company}\n"
                f"候选答案：{answer_dict.get('final_answer', 'N/A')}\n"
                f"相关页码：{answer_dict.get('relevant_pages', [])}\n"
                f"推理摘要：{answer_dict.get('reasoning_summary', '')}"
            )
        comparison_context = "\n\n---\n\n".join(comparison_context_parts)

        comparative_answer = self.openai_processor.get_interactive_answer_from_rag_context(
            question=question,
            rag_context=comparison_context,
            companies=companies,
            model=self.answering_model,
            question_kind_hint="comparative",
        )
        comparative_answer = self._validate_answer_dict(question, "comparative", comparative_answer)
        self.response_data = self.openai_processor.response_data
        comparative_answer["references"] = unique_references
        comparative_answer["route"] = "comparative"
        comparative_answer["status"] = "answered" if comparative_answer.get("final_answer") != "N/A" else "insufficient_evidence"
        comparative_answer["resolved_company_names"] = companies
        return comparative_answer
