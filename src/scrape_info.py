#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

FACULTY_LISTING_URL = "https://facultyprofiles.hkust.edu.hk/facultylisting.php"
BASE_URL = "https://facultyprofiles.hkust.edu.hk/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFO_CSV = PROJECT_ROOT / "data" / "raw" / "professor_information.csv"
LOG_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class ScrapeConfig:
    headless: bool = True
    nav_timeout_ms: int = 60_000
    selector_timeout_ms: int = 20_000


_UMLAUT_MAP = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "AE", "Ö": "OE", "Ü": "UE",
}
_UMLAUT_RE = re.compile("|".join(re.escape(k) for k in _UMLAUT_MAP))


def _strip_diacritics(text: str) -> str:
    """Convert accented characters to ASCII equivalents.

    German umlauts are expanded first (Ä→AE, ü→ue, ß→ss),
    then remaining diacritics are stripped via NFKD (é→e, ñ→n).
    """
    text = _UMLAUT_RE.sub(lambda m: _UMLAUT_MAP[m.group()], text)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def normalize_name(name: str) -> str:
    text = str(name)
    # Decode HTML entities (e.g. &#196; → Ä, &amp; → &)
    text = html.unescape(text)
    # Strip invisible Unicode characters (LRM, RLM, ZWJ, ZWNJ, BOM, etc.)
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060\ufeff\u00ad]", "", text)
    # Normalize diacritics to ASCII (Ä→A, é→e, etc.)
    text = _strip_diacritics(text)
    return " ".join(text.strip().lower().split())


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def ensure_logs_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_information_df() -> pd.DataFrame:
    df = read_csv_flexible(INFO_CSV)
    if df.empty:
        return pd.DataFrame(columns=["name", "department", "title", "research interests", "url"])
    for col in ["name", "department", "title", "research interests", "url"]:
        if col not in df.columns:
            df[col] = ""
    return df


def append_information_row(row: list[str]) -> None:
    INFO_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not INFO_CSV.exists() or INFO_CSV.stat().st_size == 0
    encoding = "utf-8-sig" if write_header else "utf-8"
    with INFO_CSV.open("a", newline="", encoding=encoding) as f:
        writer = csv.writer(f, lineterminator="\n")
        if write_header:
            writer.writerow(["name", "department", "title", "research interests", "url"])
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def write_failed_markdown(failed_names: list[str], ts: str) -> Path:
    path = LOG_DIR / f"failed_professors_{ts}.md"
    lines = [
        "# Failed Professor Lookups",
        "",
        "The following professors did not find a valid profile in the HKUST faculty listing.",
        "Common causes: Adjunct/Emeritus/Visiting/Fellow status, inconsistent student identity, or page name.",
        "",
    ]
    if not failed_names:
        lines.append("- None")
    else:
        for name in failed_names:
            lines.append(f"- {name}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_changes_csv(newly_added: list[str], to_delete: list[str], ts: str) -> Path:
    path = LOG_DIR / f"professor_changes_{ts}.csv"
    length = max(len(newly_added), len(to_delete), 1)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["newly added professor", "deleted professor"])
        for i in range(length):
            writer.writerow([
                newly_added[i] if i < len(newly_added) else "",
                to_delete[i] if i < len(to_delete) else "",
            ])
        f.flush()
        os.fsync(f.fileno())
    return path


def remove_departed_professors(info_csv: Path, to_delete: list[str]) -> int:
    if not to_delete or not info_csv.exists():
        return 0

    df = read_csv_flexible(info_csv)
    if df.empty or "name" not in df.columns:
        return 0

    delete_norm = {normalize_name(name) for name in to_delete if normalize_name(name)}
    if not delete_norm:
        return 0

    before_count = len(df)
    keep_mask = ~df["name"].fillna("").astype(str).apply(lambda x: normalize_name(x) in delete_norm)
    cleaned_df = df[keep_mask].copy()
    removed_count = before_count - len(cleaned_df)

    if removed_count > 0:
        cleaned_df.to_csv(info_csv, index=False, encoding="utf-8-sig")

    return removed_count


async def build_name_url_map(page) -> dict[str, str]:
    """Extract {normalized_name: profile_url} from the listing page using data-name attributes."""
    pairs = await page.evaluate(
        r"""
        () => {
            const results = [];
            const cards = document.querySelectorAll('div.results-div[data-name]');
            for (const card of cards) {
                const name = (card.getAttribute('data-name') || '').trim();
                const link = card.querySelector('a[href*="profiles.php"]');
                const href = link ? (link.getAttribute('href') || '') : '';
                if (name && href) results.push({name, href});
            }
            return results;
        }
        """
    )
    mapping: dict[str, str] = {}
    for item in pairs or []:
        norm = normalize_name(item["name"])
        href = item["href"]
        if norm and href:
            if href.startswith("http"):
                mapping[norm] = href
            else:
                mapping[norm] = BASE_URL + href
    return mapping


async def navigate_to_profile(page, professor_name: str, name_url_map: dict[str, str], cfg: ScrapeConfig) -> bool:
    """Navigate directly to a professor's profile page using the pre-built URL map."""
    norm = normalize_name(professor_name)
    url = name_url_map.get(norm)
    if not url:
        query_words = set(norm.split())
        best_url = None
        best_score = 0.0
        for map_norm, map_url in name_url_map.items():
            map_words = set(map_norm.split())
            if not query_words or not map_words:
                continue
            overlap = len(query_words & map_words)
            score = overlap / max(len(query_words), len(map_words))
            if score > best_score:
                best_score = score
                best_url = map_url
        if best_score >= 0.5 and best_url:
            url = best_url
            print(f"  Fuzzy match for '{professor_name}' (score={best_score:.2f}): {url}", flush=True)
        else:
            print(f"  No profile URL found for '{professor_name}'", flush=True)
            return False

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
        await page.wait_for_timeout(3000)
        return True
    except Exception as exc:
        print(f"  Failed to navigate to {url}: {exc}", flush=True)
        return False


def _parse_profile_lines(lines: list[str]) -> dict[str, str]:
    name = ""
    department = ""
    title = ""

    dept_re = re.compile(r"^(Department|Division|School)\s+of\b", flags=re.IGNORECASE)
    rank_re = re.compile(r"\b(Professor|Lecturer|Instructor|Director|Dean)\b", flags=re.IGNORECASE)
    skip_re = re.compile(
        r"^(A|Global Search|Research Interest|Publications|Projects|"
        r"Teaching|RPG Supervision|Privacy Policy|Copyright|Follow HKUST|"
        r"Office of Institutional|Google Scholar|ORCID|Scopus|"
        r"\(\d{3}\)\s*\d|Room\s+\d|.*@.*\..*|Personal Web|View Profile)$",
        flags=re.IGNORECASE,
    )
    cjk_re = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
    edu_re = re.compile(r"\b(PhD|Ph\.D|M\.S|M\.Phil|B\.S|B\.Eng|University|Institute)\b", flags=re.IGNORECASE)

    for line in lines:
        cleaned = clean_whitespace(line)
        if not cleaned or len(cleaned) <= 1:
            continue
        if skip_re.search(cleaned):
            continue
        if cjk_re.search(cleaned):
            continue

        if dept_re.search(cleaned) and not department:
            department = cleaned
            continue
        if rank_re.search(cleaned) and not title and not dept_re.search(cleaned):
            title = cleaned
            continue
        if not name and not dept_re.search(cleaned) and not edu_re.search(cleaned):
            name = cleaned
            continue

    return {"name": name, "department": department, "title": title}


async def _extract_research_interests(page) -> str:
    body_text = await page.locator("body").inner_text()
    lines = [clean_whitespace(ln) for ln in body_text.splitlines() if clean_whitespace(ln)]

    ri_indexes = [i for i, line in enumerate(lines) if line.lower() == "research interest"]
    if not ri_indexes:
        ri_indexes = [i for i, line in enumerate(lines) if "research interest" in line.lower() and len(line) < 30]
    if not ri_indexes:
        return ""

    start = ri_indexes[-1] + 1
    stop_re = re.compile(
        r"^(Publications|Projects|Teaching|RPG Supervision|Privacy Policy|"
        r"Copyright|Follow HKUST|Office of Institutional|Awards|Qualifications|"
        r"Work Experience|Professional Activity|Courses Taught|Education)",
        flags=re.IGNORECASE,
    )

    interests: list[str] = []
    seen: set[str] = set()
    for line in lines[start:]:
        if stop_re.search(line):
            break
        cleaned = clean_whitespace(line)
        if not cleaned or len(cleaned) < 3:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        interests.append(cleaned)

    return "; ".join(interests)


async def _extract_personal_web(page) -> str | None:
    """Extract Personal Web URL from the contact card (div.contact ul.fa-ul).

    Looks for the <li> containing an <i class="fas fa-id-card-alt"> icon,
    then returns the href of the sibling <a> within that same <li>.
    Falls back to matching any <a> whose visible text is "Personal Web".
    Returns None if the professor has no Personal Web link.
    """
    return await page.evaluate(
        r"""
        () => {
            const contactUl = document.querySelector('div.contact ul.fa-ul');
            if (!contactUl) return null;
            const items = contactUl.querySelectorAll('li');
            for (const li of items) {
                // Strategy 1: <li> with fa-id-card-alt icon
                const icon = li.querySelector('i.fa-id-card-alt');
                if (icon) {
                    const a = li.querySelector('a[href]');
                    if (a) return a.getAttribute('href') || null;
                }
                // Strategy 2: <a> with "Personal Web" text
                const a = li.querySelector('a[href]');
                if (a && a.textContent.trim().toLowerCase() === 'personal web') {
                    return a.getAttribute('href') || null;
                }
            }
            return null;
        }
        """
    )


async def scrape_profile_info(page, query_name: str, name_url_map: dict[str, str], cfg: ScrapeConfig) -> dict[str, str] | None:
    opened = await navigate_to_profile(page, query_name, name_url_map, cfg)
    if not opened:
        return None

    body_text = await page.locator("body").inner_text()
    lines = [clean_whitespace(ln) for ln in body_text.splitlines() if clean_whitespace(ln)]
    parsed = _parse_profile_lines(lines)

    personal_web = await _extract_personal_web(page)
    url = personal_web if personal_web else page.url
    research_interests = await _extract_research_interests(page)
    name = parsed["name"] or query_name

    return {
        "name": name,
        "department": parsed["department"],
        "title": parsed["title"],
        "research interests": research_interests,
        "url": url,
    }


async def extract_website_professor_names(page, cfg: ScrapeConfig) -> list[str]:
    await page.goto(FACULTY_LISTING_URL, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
    await page.wait_for_timeout(5000)

    names = await page.evaluate(
        r"""
        () => {
            const out = [];
            const cards = document.querySelectorAll('div.results-div[data-name]');
            for (const card of cards) {
                const name = (card.getAttribute('data-name') || '').trim();
                if (name) out.push(name);
            }
            return out;
        }
        """
    )
    cleaned = [clean_whitespace(str(x)) for x in (names or []) if clean_whitespace(str(x))]
    dedup: dict[str, str] = {}
    for item in cleaned:
        norm = normalize_name(item)
        if norm and norm not in dedup:
            dedup[norm] = item
    return list(dedup.values())


def find_new_names(input_csv: Path, info_df: pd.DataFrame) -> list[str]:
    input_df = read_csv_flexible(input_csv)
    if "name" not in input_df.columns:
        raise ValueError(f"Input CSV missing 'name' column: {input_csv}")

    input_names = [str(x).strip() for x in input_df["name"].fillna("").astype(str) if str(x).strip()]
    existing_names = [str(x).strip() for x in info_df["name"].fillna("").astype(str) if str(x).strip()]

    existing_norm = {normalize_name(x) for x in existing_names if normalize_name(x)}
    out: list[str] = []
    seen: set[str] = set()
    for name in input_names:
        norm = normalize_name(name)
        if not norm or norm in existing_norm or norm in seen:
            continue
        seen.add(norm)
        out.append(name)
    return out


def compute_departed_to_delete(existing_df: pd.DataFrame, website_names: list[str]) -> list[str]:
    existing_records = []
    for _, row in existing_df.iterrows():
        name = str(row.get("name", "") or "").strip()
        title = str(row.get("title", "") or "").strip()
        if name:
            existing_records.append((name, title))

    website_norm = {normalize_name(x) for x in website_names if normalize_name(x)}

    def _is_name_on_website(existing_name: str) -> bool:
        """Check if existing_name matches any website name.

        Uses three strategies:
        1. Exact normalized match
        2. All words in existing_name appear in some website name (subset match)
           e.g. 'allen huang' matches 'allen hao huang'
        3. All words in some website name appear in existing_name (superset match)
        """
        norm = normalize_name(existing_name)
        if norm in website_norm:
            return True
        existing_words = set(norm.split())
        if not existing_words:
            return False
        for web_norm in website_norm:
            web_words = set(web_norm.split())
            # existing is a subset of website name (e.g. "allen huang" ⊂ "allen hao huang")
            if existing_words <= web_words:
                return True
            # website name is a subset of existing name
            if web_words <= existing_words:
                return True
        return False

    departed_candidates = [(name, title) for name, title in existing_records if not _is_name_on_website(name)]

    exempt_keywords = ("adjunct", "emeritus", "visiting", "fellow")
    to_delete = [
        name
        for name, title in departed_candidates
        if not any(keyword in title.lower() for keyword in exempt_keywords)
    ]

    dedup: dict[str, str] = {}
    for name in to_delete:
        norm = normalize_name(name)
        if norm and norm not in dedup:
            dedup[norm] = name
    return list(dedup.values())


async def choose_test_professor_name(existing_norm: set[str], cfg: ScrapeConfig) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=cfg.headless)
        context = await browser.new_context(locale="en-US", timezone_id="Asia/Hong_Kong")
        page = await context.new_page()
        page.set_default_navigation_timeout(cfg.nav_timeout_ms)
        page.set_default_timeout(cfg.selector_timeout_ms)
        try:
            names = await extract_website_professor_names(page, cfg)
            for name in names:
                if normalize_name(name) not in existing_norm:
                    return name
        finally:
            await context.close()
            await browser.close()
    raise RuntimeError("Cannot find a new professor on website for --test")


def create_temp_test_input(info_df: pd.DataFrame, cfg: ScrapeConfig) -> Path:
    existing_norm = {
        normalize_name(str(x).strip())
        for x in info_df.get("name", pd.Series(dtype=str)).fillna("").astype(str)
        if str(x).strip()
    }
    candidate = asyncio.run(choose_test_professor_name(existing_norm, cfg))
    temp_path = PROJECT_ROOT / "temp.csv"
    with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["name"])
        writer.writerow([candidate])
        f.flush()
        os.fsync(f.fileno())
    return temp_path


async def run_incremental(input_csv: Path, cfg: ScrapeConfig) -> dict[str, Any]:
    ensure_logs_dir()
    info_df_before = load_information_df()
    new_names = find_new_names(input_csv, info_df_before)

    newly_added: list[str] = []
    failed_names: list[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=cfg.headless)
        context = await browser.new_context(locale="en-US", timezone_id="Asia/Hong_Kong")
        page = await context.new_page()
        page.set_default_navigation_timeout(cfg.nav_timeout_ms)
        page.set_default_timeout(cfg.selector_timeout_ms)

        try:
            # Load listing page once to build name→URL map
            print("Loading faculty listing to build name-URL map...", flush=True)
            await page.goto(FACULTY_LISTING_URL, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
            await page.wait_for_timeout(5000)
            name_url_map = await build_name_url_map(page)
            print(f"  Found {len(name_url_map)} professor profile URLs", flush=True)

            for idx, name in enumerate(new_names, start=1):
                print(f"[{idx}/{len(new_names)}] Scraping: {name}", flush=True)
                try:
                    info = await scrape_profile_info(page, name, name_url_map, cfg)
                    if info is None:
                        failed_names.append(name)
                        continue

                    append_information_row([
                        info["name"],
                        info["department"],
                        info["title"],
                        info["research interests"],
                        info["url"],
                    ])
                    newly_added.append(info["name"])
                except Exception as exc:
                    print(f"Failed for {name}: {exc}", flush=True)
                    failed_names.append(name)

                await page.wait_for_timeout(800)

            # Reload listing to get current website names for departure detection
            website_names = await extract_website_professor_names(page, cfg)
        finally:
            await context.close()
            await browser.close()

    to_delete = compute_departed_to_delete(info_df_before, website_names)
    removed_count = remove_departed_professors(INFO_CSV, to_delete)
    ts = timestamp()
    failed_md = write_failed_markdown(failed_names, ts)
    changes_csv = write_changes_csv(newly_added, to_delete, ts)

    return {
        "new_names_count": len(new_names),
        "newly_added": newly_added,
        "failed_names": failed_names,
        "to_delete": to_delete,
        "removed_count": removed_count,
        "failed_md": failed_md,
        "changes_csv": changes_csv,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental professor information update.")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path with a 'name' column")
    parser.add_argument("--headful", action="store_true", help="Run browser in headed mode")
    parser.add_argument("--test", action="store_true", help="Create temp.csv with one new real HKUST professor")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = ScrapeConfig(headless=not args.headful)

    info_df = load_information_df()

    input_path: Path | None = Path(args.input).resolve() if args.input else None
    if args.test:
        try:
            temp_path = create_temp_test_input(info_df, cfg)
            print(f"Created test input: {temp_path}", flush=True)
            if input_path is None:
                input_path = temp_path
        except Exception as exc:
            print(f"Failed to create test input: {exc}", file=sys.stderr)
            return 2

    if input_path is None:
        print("Missing required --input <csv>.", file=sys.stderr)
        return 2
    if not input_path.exists():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 2

    try:
        result = asyncio.run(run_incremental(input_path, cfg))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    print("=" * 56, flush=True)
    print(f"Input new-name candidates: {result['new_names_count']}", flush=True)
    print(f"Newly added professors: {len(result['newly_added'])}", flush=True)
    print(f"Failed lookups: {len(result['failed_names'])}", flush=True)
    print(f"Departed professors to delete: {len(result['to_delete'])}", flush=True)
    print(f"Rows removed from professor_information.csv: {result['removed_count']}", flush=True)
    print(f"Failed report: {result['failed_md']}", flush=True)
    print(f"Changes CSV: {result['changes_csv']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
