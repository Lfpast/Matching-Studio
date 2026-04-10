#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import io
import os
import random
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
INFO_CSV = PROJECT_ROOT / "data" / "raw" / "professor_information.csv"
PROJECTS_CSV = PROJECT_ROOT / "data" / "raw" / "professor_projects.csv"

LISTING_URL = "https://facultyprofiles.hkust.edu.hk/facultylisting.php"
BASE_URL = "https://facultyprofiles.hkust.edu.hk/"


@dataclass(frozen=True)
class ScrapeConfig:
    headless: bool = False
    concurrency: int = 1
    human_mode: bool = True
    nav_timeout_ms: int = 60_000
    selector_timeout_ms: int = 20_000
    min_action_delay_ms: int = 250
    max_action_delay_ms: int = 900


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
    text = html.unescape(text)
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060\ufeff\u00ad]", "", text)
    text = _strip_diacritics(text)
    return " ".join(text.strip().lower().split())


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def ascii_only(text: str) -> str:
    return "".join(ch for ch in (text or "") if ord(ch) < 128)


def clean_project_name(raw: str) -> str:
    text = clean_whitespace(ascii_only(raw))
    text = re.sub(r'^["“”\']+', "", text)
    text = re.sub(r'["“”]+$', "", text)
    return text


def clean_funding(raw: str) -> str:
    text = clean_whitespace(ascii_only(raw))
    text = re.sub(r'^["“”\']+', "", text)
    text = re.sub(r'["“”]+$', "", text)
    return text


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    tried_errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            tried_errors.append(f"{encoding}: {exc}")

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(text))
    except Exception as exc:
        tried_errors.append(f"utf-8-replace: {exc}")

    raise RuntimeError(f"Failed to read CSV {path}. Tried encodings -> {' | '.join(tried_errors)}")


def latest_changes_csv() -> Path:
    files = sorted(LOG_DIR.glob("professor_changes_*.csv"))
    if not files:
        raise FileNotFoundError("No professor_changes_*.csv found in logs/")
    return files[-1]


async def fetch_website_professor_names(headless: bool = True) -> list[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(locale="en-US", timezone_id="Asia/Hong_Kong")
        page = await context.new_page()
        try:
            await page.goto(LISTING_URL, wait_until="domcontentloaded")
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
            dedup: dict[str, str] = {}
            for name in names or []:
                clean_name = clean_whitespace(str(name))
                norm = normalize_name(clean_name)
                if norm and norm not in dedup:
                    dedup[norm] = clean_name
            return list(dedup.values())
        finally:
            await context.close()
            await browser.close()


async def build_name_url_map(page) -> dict[str, str]:
    """Extract {normalized_name: profile_url} from the listing page."""
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

    await human_pause(cfg, min_ms=300, max_ms=800)
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
        await page.wait_for_timeout(3000)
        return True
    except Exception as exc:
        print(f"  Failed to navigate to {url}: {exc}", flush=True)
        return False


async def human_pause(cfg: ScrapeConfig, *, min_ms: int | None = None, max_ms: int | None = None) -> None:
    if not cfg.human_mode:
        return
    lo = cfg.min_action_delay_ms if min_ms is None else min_ms
    hi = cfg.max_action_delay_ms if max_ms is None else max_ms
    await asyncio.sleep(random.uniform(lo, hi) / 1000.0)


def load_changes(path: Path) -> tuple[list[str], list[str]]:
    df = read_csv_flexible(path)
    for col in ["newly added professor", "deleted professor"]:
        if col not in df.columns:
            df[col] = ""

    new_profs = [
        str(x).strip()
        for x in df["newly added professor"].fillna("").astype(str)
        if str(x).strip()
    ]
    del_profs = [
        str(x).strip()
        for x in df["deleted professor"].fillna("").astype(str)
        if str(x).strip()
    ]

    seen_new: set[str] = set()
    seen_del: set[str] = set()
    dedup_new: list[str] = []
    dedup_del: list[str] = []
    for name in new_profs:
        norm = normalize_name(name)
        if norm and norm not in seen_new:
            seen_new.add(norm)
            dedup_new.append(name)
    for name in del_profs:
        norm = normalize_name(name)
        if norm and norm not in seen_del:
            seen_del.add(norm)
            dedup_del.append(name)
    return dedup_new, dedup_del





class AppendOnlyCSVWriter:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.queue: asyncio.Queue[tuple[str, str, str] | None] = asyncio.Queue(maxsize=5000)
        self.task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        if write_header:
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["name", "leading_project", "funding"])
                f.flush()
                os.fsync(f.fileno())
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            while True:
                item = await self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                writer.writerow(item)
                f.flush()
                os.fsync(f.fileno())
                self.queue.task_done()

    async def write_row(self, name: str, project: str, funding: str) -> None:
        await self.queue.put((name, project, funding))

    async def close(self) -> None:
        await self.queue.put(None)
        await self.queue.join()
        if self.task is not None:
            await self.task


async def click_projects_tab(page) -> bool:
    """Click the Projects tab on a professor profile page."""
    try:
        tab = page.locator("a[href='#proj']")
        if await tab.count() > 0 and await tab.first.is_visible():
            await tab.first.click(timeout=5_000)
            await page.wait_for_timeout(1500)
            return True
    except Exception:
        pass
    # Fallback: try JS click
    try:
        clicked = await page.evaluate(
            r"""() => {
                const el = document.querySelector("a[href='#proj']");
                if (el) { el.click(); return true; }
                return false;
            }"""
        )
        if clicked:
            await page.wait_for_timeout(1500)
        return bool(clicked)
    except Exception:
        return False


async def extract_leading_projects(page) -> list[tuple[str, str]]:
    """Extract leading projects from the Projects tab using <li data-proj='leading'> items."""
    try:
        await page.wait_for_selector("#proj", timeout=10_000)
    except PlaywrightTimeoutError:
        return []

    await page.wait_for_timeout(1000)

    try:
        data = await page.evaluate(
            r"""
            () => {
                const normalize = (text) => (text || '').replace(/\s+/g, ' ').trim();
                const results = [];
                const items = document.querySelectorAll('li[data-proj="leading"]');
                for (const li of items) {
                    const style = window.getComputedStyle(li);
                    if (style.display === 'none') continue;

                    // All spans are nested under the first col-12
                    const col = li.querySelector('div.col-12');
                    if (!col) continue;
                    const spans = col.querySelectorAll('span');
                    let project = '';
                    let funding = '';

                    // First non-icon span with content is the project name
                    for (const s of spans) {
                        if (s.classList.contains('fa-li')) continue;
                        if (s.classList.contains('badge')) continue;
                        if (s.classList.contains('text-info')) continue;
                        if (s.classList.contains('text-success')) continue;
                        const txt = normalize(s.innerText || s.textContent);
                        if (txt) { project = txt; break; }
                    }

                    // Funding: span.text-info
                    const fundingEl = li.querySelector('span.text-info');
                    if (fundingEl) {
                        funding = normalize(fundingEl.innerText || fundingEl.textContent);
                    }

                    if (project) {
                        results.push({ project, funding });
                    }
                }
                return results;
            }
            """
        )
    except Exception:
        data = []

    projects: list[tuple[str, str]] = []
    for item in data or []:
        project = clean_project_name(str(item.get("project") or ""))
        funding = clean_funding(str(item.get("funding") or ""))
        if project:
            projects.append((project, funding))

    return projects


async def scrape_single_professor(
    page, cfg: ScrapeConfig, writer: AppendOnlyCSVWriter,
    prof_name: str, name_url_map: dict[str, str],
) -> int:
    added = 0
    opened = await navigate_to_profile(page, prof_name, name_url_map, cfg)
    if not opened:
        return 0
    await human_pause(cfg, min_ms=260, max_ms=700)
    opened_proj = await click_projects_tab(page)
    if not opened_proj:
        print(f"  Could not open Projects tab for '{prof_name}'", flush=True)
        return 0
    await human_pause(cfg, min_ms=250, max_ms=650)
    projects = await extract_leading_projects(page)
    for project, funding in projects:
        await writer.write_row(prof_name, project, funding)
        added += 1
    await human_pause(cfg, min_ms=450, max_ms=1200)
    return added


def delete_departed_projects(del_profs: list[str]) -> dict[str, int]:
    if not PROJECTS_CSV.exists():
        return {name: 0 for name in del_profs}

    df = read_csv_flexible(PROJECTS_CSV)
    if df.empty or "name" not in df.columns:
        return {name: 0 for name in del_profs}

    norm_series = df["name"].fillna("").astype(str).map(normalize_name)
    counts_before = norm_series.value_counts().to_dict()

    del_norm = {normalize_name(x) for x in del_profs if normalize_name(x)}
    keep_mask = ~norm_series.isin(del_norm)
    filtered = df[keep_mask].copy()
    filtered.to_csv(PROJECTS_CSV, index=False, encoding="utf-8-sig")

    deleted_counts: dict[str, int] = {}
    for name in del_profs:
        deleted_counts[name] = int(counts_before.get(normalize_name(name), 0))
    return deleted_counts


def write_summary_md(added_counts: dict[str, int], deleted_counts: dict[str, int], source_changes_file: Path) -> Path:
    out = LOG_DIR / f"project_update_summary_{timestamp()}.md"
    lines = [
        "# Project Incremental Update Summary",
        "",
        f"- source changes file: {source_changes_file.name}",
        "",
        "## Newly Added Professors",
    ]
    if not added_counts:
        lines.append("- None")
    else:
        for name, count in added_counts.items():
            lines.append(f"- {name}: {count}")

    lines += ["", "## Deleted Professors", ""]
    if not deleted_counts:
        lines.append("- None")
    else:
        for name, count in deleted_counts.items():
            lines.append(f"- {name}: {count}")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def create_temp_csv_for_test() -> Path:
    temp_path = PROJECT_ROOT / "temp.csv"
    df = read_csv_flexible(INFO_CSV)
    existing = {
        normalize_name(str(x).strip())
        for x in df.get("name", pd.Series(dtype=str)).fillna("").astype(str)
        if str(x).strip()
    }
    website_names = asyncio.run(fetch_website_professor_names(headless=True))
    selected = next((name for name in website_names if normalize_name(name) not in existing), None)
    if not selected:
        raise RuntimeError("Cannot find a new HKUST professor for --test")

    with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["name"])
        writer.writerow([selected])
        f.flush()
        os.fsync(f.fileno())
    return temp_path


async def run_incremental(cfg: ScrapeConfig, changes_file: Path) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    new_profs, del_profs = load_changes(changes_file)

    writer = AppendOnlyCSVWriter(PROJECTS_CSV)
    await writer.start()

    added_counts: dict[str, int] = {name: 0 for name in new_profs}
    total_new = len(new_profs)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=cfg.headless, slow_mo=35 if cfg.human_mode else 0)

        # Load listing page once to build name→URL map
        context = await browser.new_context(locale="en-US", timezone_id="Asia/Hong_Kong")
        page = await context.new_page()
        page.set_default_navigation_timeout(cfg.nav_timeout_ms)
        page.set_default_timeout(cfg.selector_timeout_ms)

        print("Loading faculty listing page...", flush=True)
        await page.goto(LISTING_URL, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
        await page.wait_for_timeout(5000)
        name_url_map = await build_name_url_map(page)
        print(f"Built name→URL map with {len(name_url_map)} entries", flush=True)

        for idx, prof_name in enumerate(new_profs, start=1):
            try:
                count = await scrape_single_professor(page, cfg, writer, prof_name, name_url_map)
                added_counts[prof_name] = count
                print(f"[{idx}/{max(total_new, 1)}] {prof_name}: {count} leading projects", flush=True)
            except Exception as exc:
                print(f"  Project scrape failed for {prof_name}: {exc}", flush=True)
                print(f"[{idx}/{max(total_new, 1)}] {prof_name}: 0 leading projects", flush=True)

        await context.close()
        await browser.close()

    await writer.close()

    deleted_counts = delete_departed_projects(del_profs)
    summary = write_summary_md(added_counts, deleted_counts, changes_file)

    return {
        "changes_file": changes_file,
        "new_count": len(new_profs),
        "del_count": len(del_profs),
        "added_counts": added_counts,
        "deleted_counts": deleted_counts,
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental project update from professor_changes CSV")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--fast", action="store_true", help="Disable human-like pacing and use higher concurrency")
    parser.add_argument("--test", action="store_true", help="Create temp.csv test input in project root")
    parser.add_argument("--changes-file", type=str, default=None, help="Optional explicit professor_changes CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.test:
        try:
            temp = create_temp_csv_for_test()
            print(f"Created test input: {temp}", flush=True)
        except Exception as exc:
            print(f"Failed to create test input: {exc}", file=sys.stderr)
            return 2

    try:
        changes_file = Path(args.changes_file).resolve() if args.changes_file else latest_changes_csv()
    except Exception as exc:
        print(f"Cannot locate changes file: {exc}", file=sys.stderr)
        return 2

    cfg = ScrapeConfig(
        headless=bool(args.headless),
        human_mode=not args.fast,
        concurrency=6 if args.fast else 1,
    )

    try:
        result = asyncio.run(run_incremental(cfg, changes_file))
    except KeyboardInterrupt:
        return 130
    except PlaywrightTimeoutError as exc:
        print(f"Timeout: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    print("=" * 56, flush=True)
    print(f"Changes file: {result['changes_file']}", flush=True)
    print(f"New professors processed: {result['new_count']}", flush=True)
    print(f"Deleted professors processed: {result['del_count']}", flush=True)
    print(f"Summary: {result['summary']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
