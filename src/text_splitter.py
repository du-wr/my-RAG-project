import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self):
        self.chunk_size = int(os.getenv("TEXT_CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("TEXT_CHUNK_OVERLAP", "80"))

    def count_tokens(self, string: str, encoding_name: str = "o200k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))

    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        tables_by_page: Dict[int, List[Dict]] = {}
        for table in tables:
            if "serialized" not in table:
                continue

            page = table["page"]
            tables_by_page.setdefault(page, [])
            table_text = "\n".join(
                block["information_block"] for block in table["serialized"]["information_blocks"]
            )
            tables_by_page[page].append(
                {
                    "page": page,
                    "text": table_text,
                    "table_id": table["table_id"],
                    "length_tokens": self.count_tokens(table_text),
                    "type": "serialized_table",
                }
            )
        return tables_by_page

    def _get_inline_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        tables_by_page: Dict[int, List[Dict]] = {}
        for table in tables or []:
            table_text = str(table.get("text", "")).strip()
            if not table_text:
                continue

            page = int(table["page"])
            tables_by_page.setdefault(page, [])
            tables_by_page[page].append(
                {
                    "page": page,
                    "text": table_text,
                    "table_id": table.get("table_id"),
                    "length_tokens": self.count_tokens(table_text),
                    "type": table.get("type", "mineru_table"),
                }
            )
        return tables_by_page

    def _split_page(self, page: Dict[str, str]) -> List[Dict[str, str]]:
        # 中文切块优先保留自然断句，避免把表述和指标强行切碎。
        separators = ["\n\n", "\n", "。", "；", "！", "？", ". ", " ", ""]
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
        )
        chunks = splitter.split_text(page["text"])
        return [
            {
                "page": page["page"],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk,
            }
            for chunk in chunks
            if chunk.strip()
        ]

    def _split_report(
        self,
        file_content: Dict[str, any],
        serialized_tables_report_path: Optional[Path] = None,
    ) -> Dict[str, any]:
        chunks = []
        chunk_id = 0

        tables_by_page: Dict[int, List[Dict]] = self._get_inline_tables_by_page(
            file_content.get("content", {}).get("tables", [])
        )

        if serialized_tables_report_path is not None and serialized_tables_report_path.exists():
            with open(serialized_tables_report_path, "r", encoding="utf-8") as f:
                parsed_report = json.load(f)
            external_tables = self._get_serialized_tables_by_page(parsed_report.get("tables", []))
            for page, page_tables in external_tables.items():
                tables_by_page.setdefault(page, []).extend(page_tables)

        for page in file_content["content"]["pages"]:
            for chunk in self._split_page(page):
                chunk["id"] = chunk_id
                chunk["type"] = "content"
                chunk_id += 1
                chunks.append(chunk)

            if tables_by_page and page["page"] in tables_by_page:
                for table in tables_by_page[page["page"]]:
                    table_chunk = table.copy()
                    table_chunk["id"] = chunk_id
                    chunk_id += 1
                    chunks.append(table_chunk)

        file_content["content"]["chunks"] = chunks
        return file_content

    def split_all_reports(
        self,
        all_report_dir: Path,
        output_dir: Path,
        serialized_tables_dir: Optional[Path] = None,
    ):
        all_report_paths = list(all_report_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")
                    serialized_tables_path = None

            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)

            updated_report = self._split_report(report_data, serialized_tables_path)

            with open(output_dir / report_path.name, "w", encoding="utf-8") as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)

        print(f"Split {len(all_report_paths)} files")
