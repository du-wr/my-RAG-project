import concurrent.futures
import json
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

        # ?????????????????????????????
        for company in sorted(self.companies_df["company_name"].dropna().unique(), key=len, reverse=True):
            if company and company in remaining_text and company not in found_companies:
                found_companies.append(company)
                remaining_text = remaining_text.replace(company, "", 1)

        if found_companies:
            return found_companies

        # ?????????????????????????????????
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

        # ???????????????????????
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
        """????????????????????????????"""
        question = (question or '').strip()
        if not question:
            return []

        queries = [question]
        candidate_variants = []

        # ????????????? + ? + ?????????????????????
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
            simplified_question = re.sub(r'^[的\s:?,?]+', '', (variant or '')).strip()
            simplified_question = re.sub(r'[???.]$', '', simplified_question).strip()
            simplified_question = re.sub(r'\s+', ' ', simplified_question).strip()
            if simplified_question and simplified_question not in queries:
                queries.append(simplified_question)

            keyword_question = simplified_question
            for filler in ["请问", "一下", "情况", "相关情况", "总计是多少", "是多少", "是什么"]:
                keyword_question = keyword_question.replace(filler, ' ')
            keyword_question = re.sub(r'\s+', ' ', keyword_question).strip(' ?,?.??')
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
        """???????????????????????????"""
        question = (question or '').strip()
        table_keywords = [
            '??????', '??????', '????', '????', '????', '????', '??????',
            '??????', '????', '????', '??', '??', '????', '??:?',
        ]
        return any(keyword in question for keyword in table_keywords)

    def _rank_interactive_result(self, result: Dict, prefer_table_chunks: bool) -> float:
        score = self._get_result_score(result)
        if prefer_table_chunks and result.get('chunk_type') in {'mineru_table', 'serialized_table'}:
            score += 1000.0
        return score

    def _merge_retrieval_results(self, result_groups: List[List[Dict]], top_n: int, prefer_table_chunks: bool = False) -> List[Dict]:
        """??????????????????????????????"""
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
        """?????????????????????????????"""
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


    def _match_field(self, text: str, labels: List[str]) -> Optional[str]:
        for label in labels:
            pattern = rf"{re.escape(label)}[\s:：]*([^\n\r\t；;，,]+)"
            match = re.search(pattern, text)
            if not match:
                continue

            value = match.group(1).strip(" ：:；;，,。 ")
            if value:
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

    def _extract_numeric_from_text(self, text: str, labels: List[str]) -> Optional[float]:
        for label in labels:
            pattern = rf"{re.escape(label)}[^\n\r]*?(-?\(?[\d,]+(?:\.\d+)?\)?)"
            match = re.search(pattern, text)
            if not match:
                continue

            raw = match.group(1).replace(",", "").strip()
            negative = raw.startswith("(") and raw.endswith(")")
            raw = raw.strip("()")

            try:
                value = float(raw)
            except ValueError:
                continue

            if "万元" in text:
                value *= 10000
            elif "亿元" in text:
                value *= 100000000

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
            value = self._extract_numeric_from_text(page["text"], field_map[target_field])
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

    def get_answer_for_company(self, company_name: str, question: str, schema: str) -> dict:
        # ????????????????????????????????????
        if self.enable_rule_shortcuts and schema == "name":
            shortcut = self._try_basic_info_shortcut(company_name, question)
            if shortcut is not None:
                shortcut["route"] = "rule"
                shortcut["status"] = "answered" if shortcut.get("final_answer") != "N/A" else "insufficient_evidence"
                shortcut["resolved_company_names"] = [company_name]
                return shortcut

        if self.enable_rule_shortcuts and schema == "number":
            shortcut = self._try_numeric_shortcut(company_name, question)
            if shortcut is not None:
                shortcut["route"] = "rule"
                shortcut["status"] = "answered" if shortcut.get("final_answer") != "N/A" else "insufficient_evidence"
                shortcut["resolved_company_names"] = [company_name]
                return shortcut

        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                bm25_db_dir=self.bm25_db_dir,
                provider=self.api_provider,
                rerank_strategy=self.reranking_strategy,
                rerank_model=self.answering_model,
                embedding_model=self.embedding_model,
                cross_encoder_model=self.cross_encoder_model,
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                embedding_model=self.embedding_model,
                provider=self.api_provider,
            )

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
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
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

    def infer_schema_for_question(self, question: str) -> str:
        # ??????????????????????????? name?
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

    def answer_single_question(self, question_text: str, schema: Optional[str] = None) -> dict:
        # ?????????????????????????
        resolved_companies = self._extract_companies_from_subset(question_text) if self.new_challenge_pipeline else []
        resolved_kind = schema or 'general'

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
        retriever = self._build_retriever()
        retrieval_results = self._retrieve_context_for_company_interactive(retriever, company_name, question_text)
        if not retrieval_results:
            return {
                'question_text': question_text,
                'resolved_kind': resolved_kind,
                'resolved_companies': resolved_companies,
                'route': 'rag',
                'status': 'insufficient_evidence',
                'value': 'N/A',
                'references': [],
                'detail': {
                    'reasoning_summary': '????????????????',
                    'relevant_pages': [],
                },
            }

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_interactive_answer_from_rag_context(
            question=question_text,
            rag_context=rag_context,
            companies=resolved_companies,
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        validated_pages = self._validate_page_references(answer_dict.get('relevant_pages', []), retrieval_results)
        references = self._extract_references(validated_pages, company_name)
        final_answer = answer_dict.get('final_answer')
        status = 'answered' if final_answer != 'N/A' else 'insufficient_evidence'

        return {
            'question_text': question_text,
            'resolved_kind': answer_dict.get('answer_type', resolved_kind),
            'resolved_companies': resolved_companies,
            'route': 'rag',
            'status': status,
            'value': final_answer,
            'references': references,
            'detail': {
                'step_by_step_analysis': answer_dict.get('step_by_step_analysis'),
                'reasoning_summary': answer_dict.get('reasoning_summary'),
                'relevant_pages': validated_pages,
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
            answer_dict = self.process_question(question_text, schema)
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)

            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "route": answer_dict.get("route"),
                    "status": answer_dict.get("status"),
                    "resolved_company_names": answer_dict.get("resolved_company_names", []),
                    "answer_details": {"$ref": detail_ref},
                }

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

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies,
        )
        individual_answers = {}
        aggregated_references = []

        def process_company_question(company: str):
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            answer_dict = self.get_answer_for_company(company_name=company, question=sub_question, schema="number")
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
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

        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data
        comparative_answer["references"] = list(unique_refs.values())
        comparative_answer["route"] = "comparative"
        comparative_answer["status"] = "answered" if comparative_answer.get("final_answer") != "N/A" else "insufficient_evidence"
        comparative_answer["resolved_company_names"] = companies
        return comparative_answer
