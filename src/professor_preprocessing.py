from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
import os
import re
from pathlib import Path

import pandas as pd


DEFAULT_DEEPTECH_FILENAME_REGEX = r"^(?P<source>[A-Za-z]+)_(?P<year>\d{4})\.(?:xlsx|xls)$"


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def _normalize_header(header: str) -> str:
    return " ".join(str(header).strip().lower().split())


@dataclass
class DeepTechProject:
    cluster: str
    technology_title: str
    trl: str
    ip_status: str
    overview: str
    tech_edges: str
    applications: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    source: str = "EAS"


@dataclass
class ProfessorRecord:
    name: str
    department: str
    research_interests: str
    title: str
    url: str
    is_engineering: bool
    years_since_phd: Optional[int] = None
    priority_score: float = 0.0
    attributes: Dict[str, str] = field(default_factory=dict)
    deeptech_projects: List[DeepTechProject] = field(default_factory=list)


def load_raw_dataframe(paths: Iterable[str]) -> pd.DataFrame:
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"No input data found in paths: {list(paths)}")


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)

def load_and_merge_data(info_path: str, projects_path: str, publications_path: str) -> pd.DataFrame:
    df_info = _read_csv_with_fallback(info_path) if os.path.exists(info_path) else pd.DataFrame()
    df_projects = _read_csv_with_fallback(projects_path) if os.path.exists(projects_path) else pd.DataFrame()
    df_pubs = _read_csv_with_fallback(publications_path) if os.path.exists(publications_path) else pd.DataFrame()

    if df_info.empty:
        raise FileNotFoundError(f"Main information file not found: {info_path}")

    # Clean column names for merging
    df_info.columns = [col.strip().lower() for col in df_info.columns]
    if not df_projects.empty:
        df_projects.columns = [col.strip().lower() for col in df_projects.columns]
    if not df_pubs.empty:
        df_pubs.columns = [col.strip().lower() for col in df_pubs.columns]

    # Rename names to name in info if needed
    rename_map = {
        "names": "name",
        "research_interests": "research_interests",
        "research_interest": "research_interests",
        "research interests": "research_interests",
    }
    df_info = df_info.rename(columns=rename_map)
    
    if not df_projects.empty and "name" in df_projects.columns:
        # Group projects by name
        projects_grouped = df_projects.groupby("name").agg({
            "leading_project": lambda x: " | ".join(x.dropna().astype(str)),
            "funding": lambda x: " | ".join(x.dropna().astype(str))
        }).reset_index()
        df_info = pd.merge(df_info, projects_grouped, on="name", how="left")
        
    if not df_pubs.empty and "name" in df_pubs.columns:
        # Group publications by name
        pubs_grouped = df_pubs.groupby("name").agg({
            "paper": lambda x: " | ".join(x.dropna().astype(str))
        }).reset_index()
        df_info = pd.merge(df_info, pubs_grouped, on="name", how="left")

    return df_info


def load_deeptech_data(
    xlsx_path: str,
    column_config: Dict[str, str],
    source_id: str = "EAS",
) -> Dict[str, List[DeepTechProject]]:
    if not xlsx_path or not os.path.exists(xlsx_path):
        return {}

    df = pd.read_excel(xlsx_path, engine="openpyxl")
    if df.empty:
        return {}

    actual_columns_by_norm = {_normalize_header(col): col for col in df.columns}

    resolved_columns: Dict[str, str] = {}
    for key, configured_name in column_config.items():
        normalized = _normalize_header(configured_name)
        resolved_columns[key] = actual_columns_by_norm.get(normalized, configured_name)

    column_keys = {
        "cluster",
        "pi",
        "title",
        "trl",
        "ip_status",
        "overview",
        "tech_edges",
        "app_1",
        "app_2",
        "app_3",
        "industry_1",
        "industry_2",
    }
    for key in column_keys:
        source_col = resolved_columns.get(key, "")
        if source_col and source_col not in df.columns:
            df[source_col] = ""

    deeptech_map: Dict[str, List[DeepTechProject]] = {}

    for _, row in df.iterrows():
        pi_name = _clean_text(row.get(resolved_columns.get("pi", ""), ""))
        if not pi_name:
            continue

        applications = [
            _clean_text(row.get(resolved_columns.get("app_1", ""), "")),
            _clean_text(row.get(resolved_columns.get("app_2", ""), "")),
            _clean_text(row.get(resolved_columns.get("app_3", ""), "")),
        ]
        industries = [
            _clean_text(row.get(resolved_columns.get("industry_1", ""), "")),
            _clean_text(row.get(resolved_columns.get("industry_2", ""), "")),
        ]

        project = DeepTechProject(
            cluster=_clean_text(row.get(resolved_columns.get("cluster", ""), "")),
            technology_title=_clean_text(row.get(resolved_columns.get("title", ""), "")),
            trl=_clean_text(row.get(resolved_columns.get("trl", ""), "")),
            ip_status=_clean_text(row.get(resolved_columns.get("ip_status", ""), "")),
            overview=_clean_text(row.get(resolved_columns.get("overview", ""), "")),
            tech_edges=_clean_text(row.get(resolved_columns.get("tech_edges", ""), "")),
            applications=[app for app in applications if app],
            industries=[industry for industry in industries if industry],
            source=source_id,
        )
        deeptech_map.setdefault(_normalize_name(pi_name), []).append(project)

    return deeptech_map


def load_all_deeptech_sources(
    sources_config: List[Dict[str, str]],
    column_config: Dict[str, str],
) -> Dict[str, List[DeepTechProject]]:
    merged: Dict[str, List[DeepTechProject]] = {}
    for source_cfg in sources_config:
        xlsx_path = str(source_cfg.get("path", "")).strip()
        source_id = str(source_cfg.get("source_id", "EAS")).strip() or "EAS"
        source_map = load_deeptech_data(xlsx_path=xlsx_path, column_config=column_config, source_id=source_id)
        for professor_name, projects in source_map.items():
            merged.setdefault(professor_name, []).extend(projects)
    return merged


def discover_deeptech_sources(
    deeptech_dir: str,
    filename_regex: str = DEFAULT_DEEPTECH_FILENAME_REGEX,
    recursive: bool = False,
) -> List[Dict[str, str]]:
    base_dir = Path(deeptech_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    try:
        name_pattern = re.compile(filename_regex, re.IGNORECASE)
    except re.error:
        name_pattern = re.compile(DEFAULT_DEEPTECH_FILENAME_REGEX, re.IGNORECASE)

    iterator = base_dir.rglob("*") if recursive else base_dir.iterdir()
    discovered: List[tuple[str, int, str]] = []

    for candidate in iterator:
        if not candidate.is_file():
            continue

        match = name_pattern.match(candidate.name)
        if not match:
            continue

        source_raw = str(match.groupdict().get("source", "")).strip()
        year_raw = str(match.groupdict().get("year", "")).strip()

        source_id = source_raw.upper() if source_raw else "EAS"
        year = int(year_raw) if year_raw.isdigit() else 0
        discovered.append((source_id, year, str(candidate.resolve())))

    discovered.sort(key=lambda item: (item[0], item[1], item[2]))

    return [
        {
            "path": path,
            "source_id": source_id,
        }
        for source_id, _year, path in discovered
    ]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    rename_map = {
        "names": "name",
        "research_interests": "research_interests",
        "research_interest": "research_interests",
        "research interests": "research_interests",
    }
    df = df.rename(columns=rename_map)

    for col in ["name", "department", "research_interests", "title", "url", "leading_project", "funding", "paper"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    if "is_engineering" not in df.columns:
        df["is_engineering"] = False
    df["is_engineering"] = df["is_engineering"].fillna(False).astype(bool)

    if "years_since_phd" in df.columns:
        df["years_since_phd"] = pd.to_numeric(df["years_since_phd"], errors="coerce")
    else:
        df["years_since_phd"] = None

    if "name" in df.columns:
        df = df.drop_duplicates(subset=["name", "department", "title"], keep="first")
    return df


def build_records(
    df: pd.DataFrame,
    deeptech_map: Optional[Dict[str, List[DeepTechProject]]] = None,
) -> List[ProfessorRecord]:
    records: List[ProfessorRecord] = []
    deeptech_map = deeptech_map or {}
    extra_columns = [
        col
        for col in df.columns
        if col not in {"name", "department", "research_interests", "title", "url", "is_engineering", "years_since_phd"}
    ]
    for _, row in df.iterrows():
        name = row.get("name", "").strip()
        if not name:
            name = f"Professor {len(records) + 1}"
        attributes = {col: str(row.get(col, "")).strip() for col in extra_columns}
        records.append(
            ProfessorRecord(
                name=name,
                department=row.get("department", ""),
                research_interests=row.get("research_interests", ""),
                title=row.get("title", ""),
                url=row.get("url", ""),
                is_engineering=bool(row.get("is_engineering", False)),
                years_since_phd=row.get("years_since_phd", None),
                attributes=attributes,
                deeptech_projects=deeptech_map.get(_normalize_name(name), []),
            )
        )
    return records

def build_professor_text(record: ProfessorRecord, weights: Optional[Dict[str, float]] = None) -> str:
    weights = weights or {}
    research_weight = float(weights.get("research_interests", 1.0))
    department_weight = float(weights.get("department", 0.6))
    title_weight = float(weights.get("title", 0.3))
    project_weight = float(weights.get("leading_project", 0.8))
    paper_weight = float(weights.get("paper", 0.8))
    deeptech_weight = float(weights.get("deeptech_projects", 0.0))
    other_weight = float(weights.get("other", 0.2))

    def repeat_text(label: str, text: str, weight: float) -> List[str]:
        if not text or weight <= 0:
            return []
        repeats = max(1, int(round(weight * 3)))
        return [f"{label}: {text}"] * repeats

    parts: List[str] = []
    parts.extend(repeat_text("Department", record.department, department_weight))
    parts.extend(repeat_text("Title", record.title, title_weight))
    parts.extend(repeat_text("Research", record.research_interests, research_weight))

    for key, value in record.attributes.items():
        if value:
            if key == "leading_project":
                parts.extend(repeat_text("Projects", value, project_weight))
            elif key == "paper":
                parts.extend(repeat_text("Publications", value, paper_weight))
            else:
                parts.extend(repeat_text(key.replace("_", " ").title(), value, other_weight))

    if record.deeptech_projects:
        deeptech_texts: List[str] = []
        for project in record.deeptech_projects:
            project_text_parts = [
                project.overview,
                project.tech_edges,
                " ".join(project.applications),
                " ".join(project.industries),
            ]
            joined = " ".join(part for part in project_text_parts if part).strip()
            if joined:
                deeptech_texts.append(joined)
        if deeptech_texts:
            parts.extend(repeat_text("DeepTech", " ".join(deeptech_texts), deeptech_weight))

    return ". ".join(parts)
