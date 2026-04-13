import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class MinerUImporter:
    def __init__(self, subset_path: Path):
        self.subset_path = subset_path
        self.subset_df = pd.read_csv(subset_path, encoding="utf-8-sig")

    def _find_subset_row(self, folder_name: str) -> Optional[dict]:
        for _, row in self.subset_df.iterrows():
            source_filename = str(row.get("source_filename", ""))
            source_stem = Path(source_filename).stem
            if folder_name == source_stem:
                return row.to_dict()
        return None

    def _load_content_list(self, auto_dir: Path) -> List[dict]:
        content_candidates = list(auto_dir.glob("*_content_list.json"))
        if not content_candidates:
            raise FileNotFoundError(f"No *_content_list.json found in {auto_dir}")
        with open(content_candidates[0], "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        text = html.unescape(text or "")
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>|</div>|</tr>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    def _serialize_table_block(self, block: dict) -> str:
        """把 MinerU 的表格 HTML 转成可检索文本，避免表格数值在导入阶段丢失。"""
        table_parts: List[str] = []

        caption = self._strip_html_tags(str(block.get("table_caption", "")))
        if caption:
            table_parts.append(caption)

        table_body = str(block.get("table_body", "") or block.get("html", "") or "")
        if table_body:
            normalized_html = html.unescape(table_body)
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", normalized_html, flags=re.IGNORECASE | re.DOTALL)
            serialized_rows: List[str] = []
            for row_html in rows:
                cells = re.findall(r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", row_html, flags=re.IGNORECASE | re.DOTALL)
                cleaned_cells = []
                for cell in cells:
                    cleaned = self._strip_html_tags(cell)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    if cleaned:
                        cleaned_cells.append(cleaned)
                if cleaned_cells:
                    serialized_rows.append(" | ".join(cleaned_cells))

            if serialized_rows:
                table_parts.append("\n".join(serialized_rows))
            else:
                plain_table = self._strip_html_tags(table_body)
                if plain_table:
                    table_parts.append(plain_table)

        footnote = self._strip_html_tags(str(block.get("table_footnote", "")))
        if footnote:
            table_parts.append(footnote)

        return "\n".join(part for part in table_parts if part).strip()

    def _extract_block_text(self, block: dict) -> str:
        block_type = block.get("type")
        if block_type == "table":
            return self._serialize_table_block(block)
        return str(block.get("text", "")).strip()

    def _build_pages(self, content_list: List[dict]) -> List[dict]:
        pages = defaultdict(list)
        for block in content_list:
            block_type = block.get("type")
            if block_type not in {"text", "table", "equation"}:
                continue

            text = self._extract_block_text(block)
            if not text:
                continue

            page_idx = int(block.get("page_idx", 0)) + 1
            pages[page_idx].append(text)

        ordered_pages = []
        for page_no in sorted(pages):
            # 先按页重建完整文本，后续再统一切块，便于保留页码引用和父文档召回。
            ordered_pages.append(
                {
                    "page": page_no,
                    "text": "\n".join(pages[page_no]).strip(),
                }
            )
        return ordered_pages

    def _build_tables(self, content_list: List[dict]) -> List[dict]:
        """单独保留表格块，供后续切成独立检索单元。"""
        tables: List[dict] = []
        table_id = 0
        for block in content_list:
            if block.get("type") != "table":
                continue

            table_text = self._serialize_table_block(block)
            if not table_text:
                continue

            tables.append(
                {
                    "table_id": table_id,
                    "page": int(block.get("page_idx", 0)) + 1,
                    "type": "mineru_table",
                    "text": table_text,
                    "caption": self._strip_html_tags(str(block.get("table_caption", ""))),
                    "footnote": self._strip_html_tags(str(block.get("table_footnote", ""))),
                }
            )
            table_id += 1
        return tables

    def import_reports(self, source_dir: Path, output_dir: Path) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        imported = 0

        for report_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            auto_dir = report_dir / "auto"
            if not auto_dir.exists():
                continue

            subset_row = self._find_subset_row(report_dir.name)
            if subset_row is None:
                continue

            content_list = self._load_content_list(auto_dir)
            pages = self._build_pages(content_list)
            if not pages:
                continue
            tables = self._build_tables(content_list)

            report = {
                "metainfo": {
                    "sha1_name": subset_row["sha1"],
                    "company_name": subset_row["company_name"],
                    "source_filename": subset_row.get("source_filename", f"{report_dir.name}.pdf"),
                    "parser": "mineru",
                },
                "content": {
                    "pages": pages,
                    "tables": tables,
                },
            }

            output_path = output_dir / f"{subset_row['sha1']}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            imported += 1

        return imported
