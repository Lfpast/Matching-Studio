from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import pandas as pd


DEFAULT_STARTUP_FILENAME_REGEX = r"^startup_(?P<year>\d{4})\.(?:xlsx|xls)$"
_NUMBERED_ITEM_RE = re.compile(r"(?<!\S)(\d+)\.\s*")


def _normalize_header(header: str) -> str:
    return " ".join(str(header).strip().lower().split())


@dataclass
class StartupRecord:
    startup_id: str
    source_year: Optional[int]
    source_file: str
    company_name: str
    people: List[str]
    ref_code: str
    funding: str
    background_year: str
    categories: List[str]
    description: str
    tels: List[str]
    emails: List[str]
    website: str
    raw_row_index: int


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def split_numbered_items(text: str) -> List[str]:
    normalized = normalize_text(text).replace("\r", "\n")
    if not normalized:
        return []

    matches = list(_NUMBERED_ITEM_RE.finditer(normalized))
    parsed_items: List[str] = []

    if matches:
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
            value = normalized[start:end].strip(" \t\n;,")
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                parsed_items.append(value)
    else:
        parsed_items = [re.sub(r"\s+", " ", normalized).strip()]

    # De-duplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for item in parsed_items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def split_categories(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [part.strip() for part in normalized.split(",") if part.strip()]


def parse_source_year(file_path: str) -> Optional[int]:
    name = Path(file_path).name
    match = re.match(DEFAULT_STARTUP_FILENAME_REGEX, name, flags=re.IGNORECASE)
    if not match:
        return None
    year_text = str(match.groupdict().get("year", "")).strip()
    return int(year_text) if year_text.isdigit() else None


def build_startup_id(company_name: str, source_year: Optional[int], row_index: int) -> str:
    slug_base = normalize_text(company_name).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug_base).strip("-")
    if not slug:
        slug = "startup"

    year_part = str(source_year) if source_year is not None else "na"
    return f"{slug}-{year_part}-{int(row_index)}"


def load_single_startup_xlsx(path: str, columns: Dict[str, str]) -> List[StartupRecord]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return []

    df = pd.read_excel(str(file_path), engine="openpyxl")
    if df.empty:
        return []

    source_year = parse_source_year(str(file_path))

    actual_columns_by_norm = {_normalize_header(col): col for col in df.columns}
    resolved_columns: Dict[str, str] = {}
    for key, configured_name in columns.items():
        normalized = _normalize_header(configured_name)
        resolved_columns[key] = actual_columns_by_norm.get(normalized, str(configured_name))

    # Keep all configured fields available even when source columns are missing.
    for source_col in resolved_columns.values():
        if source_col and source_col not in df.columns:
            df[source_col] = ""

    def get_value(row: pd.Series, key: str) -> str:
        configured_column = resolved_columns.get(key, "")
        column_name = str(configured_column) if configured_column is not None else ""
        if not column_name:
            return ""

        if column_name in df.columns:
            return normalize_text(row.get(column_name, ""))

        # Fallback: recover columns that differ only by whitespace/newline formatting.
        normalized = _normalize_header(column_name)
        mapped_column = actual_columns_by_norm.get(normalized)
        if not mapped_column or mapped_column not in df.columns:
            return ""
        return normalize_text(row.get(mapped_column, ""))

    records: List[StartupRecord] = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        company_name = get_value(row, "company_name")
        people_text = get_value(row, "people")
        category_text = get_value(row, "category")
        tel_text = get_value(row, "tel")
        email_text = get_value(row, "email")

        record = StartupRecord(
            startup_id=build_startup_id(company_name=company_name, source_year=source_year, row_index=row_index),
            source_year=source_year,
            source_file=str(file_path.resolve()),
            company_name=company_name,
            people=split_numbered_items(people_text),
            ref_code=get_value(row, "ref_code"),
            funding=get_value(row, "funding"),
            background_year=get_value(row, "background_year"),
            categories=split_categories(category_text),
            description=get_value(row, "description"),
            tels=split_numbered_items(tel_text),
            emails=split_numbered_items(email_text),
            website=get_value(row, "website"),
            raw_row_index=row_index,
        )
        records.append(record)

    return records


def discover_startup_sources(directory: str, filename_regex: str, recursive: bool) -> List[Dict[str, object]]:
    base_dir = Path(directory)
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    try:
        pattern = re.compile(filename_regex, flags=re.IGNORECASE)
    except re.error:
        pattern = re.compile(DEFAULT_STARTUP_FILENAME_REGEX, flags=re.IGNORECASE)

    iterator = base_dir.rglob("*") if recursive else base_dir.iterdir()
    discovered: List[Dict[str, object]] = []

    for candidate in iterator:
        if not candidate.is_file():
            continue

        if not pattern.match(candidate.name):
            continue

        source_year = parse_source_year(candidate.name)
        discovered.append(
            {
                "path": str(candidate.resolve()),
                "source_year": source_year,
            }
        )

    discovered.sort(
        key=lambda item: (
            item["source_year"] if item["source_year"] is not None else -1,
            Path(str(item["path"])).name.lower(),
        )
    )
    return discovered


def load_all_startup_sources(config: Dict[str, object]) -> List[StartupRecord]:
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    startup_cfg = config.get("startup", {}) if isinstance(config, dict) else {}

    auto_cfg = data_cfg.get("startup_auto_discovery", {}) if isinstance(data_cfg, dict) else {}
    enabled = bool(auto_cfg.get("enabled", True))
    if not enabled:
        return []

    directory = str(auto_cfg.get("directory", "data/raw/"))
    filename_regex = str(auto_cfg.get("filename_regex", DEFAULT_STARTUP_FILENAME_REGEX))
    recursive = bool(auto_cfg.get("recursive", False))
    columns = startup_cfg.get("columns", {}) if isinstance(startup_cfg, dict) else {}

    sources = discover_startup_sources(directory=directory, filename_regex=filename_regex, recursive=recursive)

    all_records: List[StartupRecord] = []
    for source in sources:
        source_path = str(source.get("path", "")).strip()
        if not source_path:
            continue
        all_records.extend(load_single_startup_xlsx(path=source_path, columns=columns))
    return all_records


def build_startup_text(record: StartupRecord, weights: Dict[str, float]) -> str:
    weights = weights or {}

    company_weight = float(weights.get("company_name", 1.0))
    category_weight = float(weights.get("category", 1.0))
    description_weight = float(weights.get("description", 1.0))

    def repeat_text(label: str, text: str, weight: float) -> List[str]:
        if not text or weight <= 0:
            return []
        repeats = max(1, int(round(weight * 3)))
        return [f"{label}: {text}"] * repeats

    categories_text = ", ".join(record.categories)

    parts: List[str] = []
    parts.extend(repeat_text("Company", record.company_name, company_weight))
    parts.extend(repeat_text("Category", categories_text, category_weight))
    parts.extend(repeat_text("Description", record.description, description_weight))
    return ". ".join(parts)
