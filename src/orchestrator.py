import asyncio
import sys
import re
import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import pandas as pd
from datetime import datetime

class UpdateStage(Enum):
    """更新流程阶段"""
    SCRAPE_INFO = "scrape_info"
    SCRAPE_PUBLICATION = "scrape_publication"
    SCRAPE_PROJECT = "scrape_project"
    MERGING_REPORTS = "merging_reports"
    COMPLETED = "completed"

@dataclass
class ProfessorStats:
    """单个教授的统计"""
    name: str
    stage_completed: str  # 最后完成的阶段
    publications_added: int = 0
    publications_deleted: int = 0
    projects_added: int = 0
    projects_deleted: int = 0
    error_message: Optional[str] = None

@dataclass
class UpdateResult:
    """更新流程的最终结果"""
    success: bool
    total_professors: int
    completed_professors: int
    failed_professors: List[ProfessorStats] = field(default_factory=list)
    markdown_content: str = ""
    summary_stats: dict = field(default_factory=dict)
    error_summary: str = ""

class DatabaseUpdateOrchestrator:
    def __init__(self, config: dict, websocket_manager, user_id: str):
        self.config = config
        self.websocket_manager = websocket_manager
        self.user_id = user_id
        self.professor_count = 0  # 待处理教授数
        self.stats: List[ProfessorStats] = []
        self.current_progress_pct = 0.0

    async def run_update_pipeline(self, input_csv_path: str, deeptech_xlsx_path: str) -> UpdateResult:
        try:
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Input CSV not found at {input_csv_path}")
                
            self.professor_count = self._parse_professor_count_from_csv(input_csv_path)
            if self.professor_count == 0:
                raise ValueError("No professors found in input CSV")

            # Stage 1: scrape_info
            await self.websocket_manager.send_progress(
                self.user_id,
                0.0,
                UpdateStage.SCRAPE_INFO.value,
                "Starting"
            )
            await self._run_script(
                "src/scrape_info.py",
                ["--input", input_csv_path],
                UpdateStage.SCRAPE_INFO.value,
                0.0,
                33.3
            )
            
            # Find the latest changes csv in logs/
            logs_dir = self.config.get("database_update", {}).get("logs_directory", "./logs/")
            changes_csvs = glob.glob(os.path.join(logs_dir, "professor_changes_*.csv"))
            if not changes_csvs:
                raise Exception("Cannot find professor_changes_*.csv in logs directory")
            latest_changes_csv = max(changes_csvs, key=os.path.getctime)
            
            # Stage 2: scrape_publication
            await self.websocket_manager.send_progress(
                self.user_id,
                33.3,
                UpdateStage.SCRAPE_PUBLICATION.value,
                "Starting"
            )
            await self._run_script(
                "src/scrape_publication.py",
                ["--changes-file", latest_changes_csv],
                UpdateStage.SCRAPE_PUBLICATION.value,
                33.3,
                33.3
            )
            
            # Stage 3: scrape_project
            await self.websocket_manager.send_progress(
                self.user_id,
                66.6,
                UpdateStage.SCRAPE_PROJECT.value,
                "Starting"
            )
            args3 = ["--changes-file", latest_changes_csv]
            if deeptech_xlsx_path and os.path.exists(deeptech_xlsx_path):
                await self.websocket_manager.send_log(
                    self.user_id,
                    "[scrape_project] DeepTech xlsx uploaded; current scraper CLI does not consume this file directly."
                )
            await self._run_script("src/scrape_project.py", args3, UpdateStage.SCRAPE_PROJECT.value, 66.6, 33.4)
            
            # Stage 4: merging reports
            await self.websocket_manager.send_progress(self.user_id, 95.0, UpdateStage.MERGING_REPORTS.value, "Finalizing")
            
            pub_summaries = glob.glob(os.path.join(logs_dir, "publication_update_summary_*.md"))
            proj_summaries = glob.glob(os.path.join(logs_dir, "project_update_summary_*.md"))
            failed_summaries = glob.glob(os.path.join(logs_dir, "failed_professors_*.md"))
            
            latest_pub = max(pub_summaries, key=os.path.getctime) if pub_summaries else ""
            latest_proj = max(proj_summaries, key=os.path.getctime) if proj_summaries else ""
            latest_failed = max(failed_summaries, key=os.path.getctime) if failed_summaries else ""
            
            merged_md = self._merge_markdown_reports(latest_pub, latest_proj, latest_failed)
            
            summary_stats = {
                "total": self.professor_count,
            }
            
            return UpdateResult(
                success=True,
                total_professors=self.professor_count,
                completed_professors=self.professor_count,
                markdown_content=merged_md,
                summary_stats=summary_stats
            )
            
        except Exception as e:
            await self.websocket_manager.send_error(self.user_id, str(e))
            raise

    def _extract_professor_name_from_line(self, line: str) -> Optional[str]:
        # scrape_info pattern: "[1/25] Scraping: John Doe"
        m = re.search(r"\[\s*\d+\s*/\s*\d+\s*\]\s*Scraping:\s*(.+)$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # marker + processing pattern: "[2/15] Processing publication: John Doe"
        m = re.search(r"\[\s*\d+\s*/\s*\d+\s*\]\s*Processing\s+(?:publication|project):\s*(.+)$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # scrape_publication pattern: "John Doe: 12 publications"
        m = re.search(r"^\s*(?:\[\s*\d+\s*/\s*\d+\s*\]\s*)?([^:]{2,120}):\s+\d+\s+publications\b", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # scrape_project pattern: "John Doe: 3 leading projects"
        m = re.search(r"^\s*(?:\[\s*\d+\s*/\s*\d+\s*\]\s*)?([^:]{2,120}):\s+\d+\s+leading projects\b", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Generic fallback pattern: "Processing: John Doe"
        m = re.search(r"Processing\s*:\s*([a-zA-Z][a-zA-Z\s.\-']{1,120})", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        return None

    def _extract_progress_marker(self, line: str) -> Optional[Tuple[int, int]]:
        marker = re.search(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]", line)
        if not marker:
            return None

        current = int(marker.group(1))
        total = int(marker.group(2))
        if total <= 0:
            return None
        return min(current, total), total

    async def _run_script(self, script_path: str, args: List[str], current_stage: str, start_pct: float, stage_pct: float) -> str:
        try:
            cmd = [sys.executable, script_path] + args
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            timeout = self.config.get("database_update", {}).get("script_timeout_seconds", 3600)
            
            stdout_lines = []
            stderr_lines = []
            seen_professors = set()

            await self.websocket_manager.send_log(self.user_id, f"[{current_stage}] started")
            
            async def read_stream(stream, is_stderr: bool):
                while True:
                    line = await stream.readline()
                    if line:
                        decoded_line = line.decode('utf-8', errors='ignore').strip()
                        if not decoded_line:
                            continue

                        if not is_stderr:
                            stdout_lines.append(decoded_line)
                            await self.websocket_manager.send_log(self.user_id, f"[{current_stage}] {decoded_line[:180]}")

                            marker = self._extract_progress_marker(decoded_line)
                            prof_name = self._extract_professor_name_from_line(decoded_line)

                            target_progress = None
                            if marker:
                                current, total = marker
                                target_progress = start_pct + (current / total) * stage_pct
                                if prof_name:
                                    seen_professors.add(prof_name)
                            elif prof_name and prof_name not in seen_professors:
                                seen_professors.add(prof_name)
                                target_progress = start_pct + (len(seen_professors) / max(self.professor_count, 1)) * stage_pct

                            if target_progress is not None:
                                progress = min(max(target_progress, start_pct), start_pct + stage_pct)
                                if progress <= self.current_progress_pct + 0.01:
                                    continue

                                self.current_progress_pct = progress
                                await self.websocket_manager.send_progress(
                                    self.user_id,
                                    progress,
                                    current_stage,
                                    prof_name or f"{current_stage} running"
                                )
                        else:
                            stderr_lines.append(decoded_line)
                    else:
                        break

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(process.stdout, False),
                        read_stream(process.stderr, True)
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise Exception(f"Script {script_path} timed out after {timeout} seconds")
            
            await process.wait()
            
            if process.returncode != 0:
                stderr_tail = " | ".join(stderr_lines[-3:]) if stderr_lines else "No stderr output"
                raise Exception(f"Script failed with return code {process.returncode}: {stderr_tail}")

            # 即使未解析到单教授日志，也保证阶段进度可见
            stage_end_progress = min(start_pct + stage_pct, 99.0)
            if self.current_progress_pct < stage_end_progress:
                self.current_progress_pct = stage_end_progress
                await self.websocket_manager.send_progress(
                    self.user_id,
                    stage_end_progress,
                    current_stage,
                    "Stage completed"
                )
            await self.websocket_manager.send_log(self.user_id, f"[{current_stage}] completed")
            
            return "\n".join(stdout_lines)
            
        except Exception as e:
            await self.websocket_manager.send_error(
                self.user_id,
                str(e),
                current_stage=current_stage
            )
            raise

    def _merge_markdown_reports(self, pub_summary_path: str, proj_summary_path: str, failed_summary_path: str = "") -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pub_content = "No publication summary found."
        proj_content = "No project summary found."
        failed_content = "No failed professor summary found."
        
        if os.path.exists(pub_summary_path):
            with open(pub_summary_path, 'r', encoding='utf-8') as f:
                pub_content = f.read()
                
        if os.path.exists(proj_summary_path):
            with open(proj_summary_path, 'r', encoding='utf-8') as f:
                proj_content = f.read()

        if failed_summary_path and os.path.exists(failed_summary_path):
            with open(failed_summary_path, 'r', encoding='utf-8') as f:
                failed_content = f.read()
                
        merged = f"# Database Update Report - {timestamp}\n\n"
        merged += f"## Summary\n"
        merged += f"- Total Professors Processed (Input Estimate): {self.professor_count}\n\n"
        merged += "## Publication Summary\n"
        merged += pub_content + "\n\n"
        merged += "## Project Summary\n"
        merged += proj_content + "\n\n"
        merged += "## Failed Professor Lookup Summary\n"
        merged += failed_content + "\n"
        
        return merged

    def _parse_professor_count_from_csv(self, csv_path: str) -> int:
        try:
            df = pd.read_csv(csv_path)
            return len(df)
        except Exception:
            return 0
