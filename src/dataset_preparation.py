import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


def _derive_company_name(pdf_path: Path) -> str:
    stem = pdf_path.stem
    # 兼容“公司名：2025年年度报告”“公司名:2025年年度报告”等常见命名。
    candidate = stem.split("：", 1)[0].split(":", 1)[0].strip()
    candidate = re.sub(r"(20\d{2}年年度报告.*)$", "", candidate).strip()
    candidate = re.sub(r"(公司年度报告全文.*)$", "", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate).strip(" -_")
    return candidate or stem


def _stable_sha1_for_file(pdf_path: Path) -> str:
    return hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()


def prepare_dataset_from_source(root_path: Path, source_dir: Path) -> dict:
    pdf_reports_dir = root_path / "pdf_reports"
    pdf_reports_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(source_dir.rglob("*.pdf"))
    rows = []

    for pdf_file in pdf_files:
        sha1_name = _stable_sha1_for_file(pdf_file)
        target_path = pdf_reports_dir / f"{sha1_name}.pdf"
        shutil.copy2(pdf_file, target_path)
        rows.append(
            {
                "sha1": sha1_name,
                "company_name": _derive_company_name(pdf_file),
                "source_filename": pdf_file.name,
            }
        )

    subset_csv = root_path / "subset.csv"
    with open(subset_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sha1", "company_name", "source_filename"])
        writer.writeheader()
        writer.writerows(rows)

    questions_json = root_path / "questions.json"
    if not questions_json.exists():
        with open(questions_json, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    return {
        "pdf_count": len(pdf_files),
        "pdf_reports_dir": pdf_reports_dir,
        "subset_csv": subset_csv,
        "questions_json": questions_json,
    }
