from __future__ import annotations

import os
import re
from pathlib import Path
import asyncio
import uuid

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, HTTPException, status, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.professor_preprocessing import (
    DEFAULT_DEEPTECH_FILENAME_REGEX,
    build_records,
    clean_dataframe,
    discover_deeptech_sources,
    load_all_deeptech_sources,
    load_and_merge_data,
    load_deeptech_data,
)
from src.embedding_model import TextEmbedder
from src.professor_graph_builder import build_graph
from src.professor_matching_engine import MatchingEngine
from src.professor_priority_strategy import assign_priority_scores
from src.startup_graph_builder import build_startup_graph
from src.startup_matching_engine import StartupMatchingEngine
from src.startup_preprocessing import DEFAULT_STARTUP_FILENAME_REGEX, load_all_startup_sources
from api.schemas import (
    MatchRequest, MatchResponse, KeywordItem,
    LoginRequest, LoginResponse, FileUploadResponse,
    UpdateStartRequest, UpdateStartResponse, UpdateResultResponse
)
from api.auth import load_credentials_from_config, verify_password, create_token, verify_token, extract_token_from_header
from api.websocket_manager import ConnectionManager
from src.orchestrator import DatabaseUpdateOrchestrator
from datetime import datetime


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {}


def build_engine(config_path: str) -> tuple:
    """Build professor engine, startup engine, and config."""
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    file_cfg = config.get("file_upload", {})
    project_root = Path(__file__).resolve().parent.parent

    def _resolve_path(path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path

    professor_df = load_and_merge_data(
        info_path=data_cfg.get("raw_csv", "data/raw/professor_information.csv"),
        projects_path=data_cfg.get("projects_csv", "data/raw/professor_projects.csv"),
        publications_path=data_cfg.get("publications_csv", "data/raw/professor_publications.csv")
    )
    professor_df = clean_dataframe(professor_df)

    auto_cfg = data_cfg.get("deeptech_auto_discovery", {})
    auto_enabled = bool(auto_cfg.get("enabled", True))
    auto_recursive = bool(auto_cfg.get("recursive", False))
    auto_regex_text = str(auto_cfg.get("filename_regex", DEFAULT_DEEPTECH_FILENAME_REGEX))
    auto_dir = str(auto_cfg.get("directory", file_cfg.get("deeptech_destination", "data/raw/")))

    try:
        auto_regex = re.compile(auto_regex_text, re.IGNORECASE)
    except re.error:
        auto_regex = re.compile(DEFAULT_DEEPTECH_FILENAME_REGEX, re.IGNORECASE)

    discovered_sources = discover_deeptech_sources(
        deeptech_dir=str(_resolve_path(auto_dir)),
        filename_regex=auto_regex.pattern,
        recursive=auto_recursive,
    ) if auto_enabled else []

    configured_sources = data_cfg.get("deeptech_sources", [])
    merged_sources = []
    seen_paths = set()

    for source_cfg in [*discovered_sources, *configured_sources]:
        raw_path = str(source_cfg.get("path", "")).strip()
        if not raw_path:
            continue

        resolved_path = _resolve_path(raw_path)
        path_key = str(resolved_path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        raw_source_id = str(source_cfg.get("source_id", "")).strip()
        if raw_source_id:
            source_id = raw_source_id.upper()
        else:
            name_match = auto_regex.match(resolved_path.name)
            source_id = str(name_match.group("source") if name_match else "EAS").upper()

        merged_sources.append({
            "path": path_key,
            "source_id": source_id,
        })

    professor_deeptech_cfg = config.get("deeptech", {})
    professor_deeptech_columns = professor_deeptech_cfg.get("columns", {})
    if merged_sources:
        professor_deeptech_map = load_all_deeptech_sources(
            sources_config=merged_sources,
            column_config=professor_deeptech_columns,
        )
    else:
        professor_deeptech_map = load_deeptech_data(
            xlsx_path=data_cfg.get("deeptech_xlsx", "data/raw/EAS.xlsx"),
            column_config=professor_deeptech_columns,
            source_id="EAS",
        )
    professor_records = build_records(professor_df, deeptech_map=professor_deeptech_map)

    professor_priority_cfg = config.get("priority", {})
    assign_priority_scores(
        professor_records,
        w_years=professor_priority_cfg.get("w_years", 1.0),
        w_title=professor_priority_cfg.get("w_title", 1.0),
        w_engineering=professor_priority_cfg.get("w_engineering", 1.0),
        default_years_since_phd=professor_priority_cfg.get("default_years_since_phd", 10),
        engineering_bonus=professor_priority_cfg.get("engineering_bonus", 1.2),
    )

    professor_embedding_cfg = config.get("embedding", {})
    embedder = TextEmbedder(model_name=professor_embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    professor_attribute_weights = professor_embedding_cfg.get("attribute_weights", {})

    professor_graph_cfg = config.get("graph", {})
    professor_graph = build_graph(
        professor_records,
        similarity_threshold=professor_graph_cfg.get("similarity_threshold", 0.2),
        department_edge_weight=professor_graph_cfg.get("department_edge_weight", 0.1),
    )
    
    professor_query_cfg = config.get("query", {})

    professor_engine = MatchingEngine(
        records=professor_records, 
        embedder=embedder, 
        graph=professor_graph, 
        attribute_weights=professor_attribute_weights,
        query_config=professor_query_cfg,
    )

    startup_cfg = config.get("startup", {})
    startup_auto_cfg = data_cfg.get("startup_auto_discovery", {})
    startup_directory = str(
        startup_auto_cfg.get(
            "directory",
            file_cfg.get("startup_destination", file_cfg.get("deeptech_destination", "data/raw/")),
        )
    )

    startup_loader_cfg = {
        "data": {
            "startup_auto_discovery": {
                "enabled": bool(startup_auto_cfg.get("enabled", True)),
                "directory": str(_resolve_path(startup_directory)),
                "recursive": bool(startup_auto_cfg.get("recursive", False)),
                "filename_regex": str(
                    startup_auto_cfg.get("filename_regex", DEFAULT_STARTUP_FILENAME_REGEX)
                ),
            }
        },
        "startup": startup_cfg,
    }
    startup_records = load_all_startup_sources(startup_loader_cfg)

    startup_graph_cfg = startup_cfg.get("graph", {})
    category_weight = float(startup_graph_cfg.get("category_weight", 0.571))
    description_weight = float(startup_graph_cfg.get("description_weight", 0.429))
    total_weight = category_weight + description_weight
    if total_weight <= 0:
        category_weight, description_weight = 0.571, 0.429
    else:
        category_weight = category_weight / total_weight
        description_weight = description_weight / total_weight

    startup_graph = build_startup_graph(
        startup_records,
        similarity_threshold=float(startup_graph_cfg.get("similarity_threshold", 0.22)),
        category_weight=category_weight,
        description_weight=description_weight,
    )

    startup_engine = StartupMatchingEngine(
        records=startup_records,
        embedder=embedder,
        graph=startup_graph,
        query_processor=professor_engine.query_processor,
        config=startup_cfg,
    )
    
    return professor_engine, startup_engine, config


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path

app = FastAPI(title="Professor Matching API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
CONFIG_PATH = os.environ.get("PROF_MATCH_CONFIG", "config/config.yaml")
professor_engine, startup_engine, config = build_engine(CONFIG_PATH)
professor_query_cfg = config.get("query", {})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
async def match(request: MatchRequest) -> MatchResponse:
    professor_validate_query = request.validate_query and professor_query_cfg.get("enable_validation", True)
    professor_use_keyword_extraction = request.use_keyword_extraction and professor_query_cfg.get("enable_keyword_extraction", True)

    if request.mode == "startup":
        startup_matching_cfg = config.get("startup", {}).get("matching", {})
        provided_fields = set(getattr(request, "model_fields_set", set()))
        startup_validate_query = bool(professor_query_cfg.get("enable_validation", True))
        startup_use_keyword_extraction = bool(professor_query_cfg.get("enable_keyword_extraction", True))

        top_k = request.top_k if "top_k" in provided_fields else int(startup_matching_cfg.get("default_top_k", request.top_k))
        alpha = 1.0
        beta = 0.0
        graph_neighbor_weight = (
            request.graph_neighbor_weight
            if "graph_neighbor_weight" in provided_fields
            else float(startup_matching_cfg.get("default_graph_neighbor_weight", request.graph_neighbor_weight))
        )

        result = startup_engine.match(
            query=request.query,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
            graph_neighbor_weight=graph_neighbor_weight,
            validate_query=startup_validate_query,
            use_keyword_extraction=startup_use_keyword_extraction,
        )
        professor_results = []
        startup_results = result.get("startup_results", [])
    else:
        result = professor_engine.match(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha,
            beta=request.beta,
            graph_neighbor_weight=request.graph_neighbor_weight,
            validate_query=professor_validate_query,
            use_keyword_extraction=professor_use_keyword_extraction,
        )
        professor_results = result.get("results", [])
        startup_results = []
    
    return MatchResponse(
        query=request.query,
        mode=request.mode,
        status=result["status"],
        message=result["message"],
        suggestions=result["suggestions"],
        results=professor_results,
        startup_results=startup_results,
        keywords=[KeywordItem(keyword=kw, weight=w) for kw, w in result["keywords"]],
        enhanced_query=result["enhanced_query"],
    )

# 初始化WebSocket管理器
ws_manager = ConnectionManager()

# 从config读取认证凭证
credentials = load_credentials_from_config(CONFIG_PATH)


def refresh_runtime_engine() -> None:
    global professor_engine, startup_engine, config, professor_query_cfg, credentials

    new_professor_engine, new_startup_engine, new_config = build_engine(CONFIG_PATH)
    professor_engine = new_professor_engine
    startup_engine = new_startup_engine
    config = new_config
    professor_query_cfg = config.get("query", {})
    credentials = load_credentials_from_config(CONFIG_PATH)

# 活跃任务追踪（task_id -> {user_id, status, progress, ...}）
active_tasks = {}

@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    if not verify_password(request.username, request.password, credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    token = create_token(request.username, config)
    ttl_minutes = config.get("auth", {}).get("token_ttl_minutes", 60)
    
    return LoginResponse(
        token=token,
        username=request.username,
        expires_in=ttl_minutes * 60
    )


@app.post("/api/upload/input-csv", response_model=FileUploadResponse, tags=["Database Update"])
async def upload_input_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    file_config = config.get("file_upload", {})
    max_size = file_config.get("input_csv_max_size_mb", 10) * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail=f"File exceeds {max_size/1024/1024}MB limit")
    
    destination = resolve_project_path(file_config.get("input_csv_destination", "./")) / "input.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, "wb") as f:
        f.write(content)
    
    return FileUploadResponse(
        status="success",
        filename="input.csv",
        message="File uploaded successfully"
    )


@app.post("/api/upload/deeptech-xlsx", response_model=FileUploadResponse, tags=["Database Update"])
async def upload_deeptech_xlsx(file: UploadFile = File(...)):
    allowed_exts = config.get("file_upload", {}).get("deeptech_allowed_extensions", [".xlsx", ".xls"])
    if not any(file.filename.lower().endswith(str(ext).lower()) for ext in allowed_exts):
        raise HTTPException(status_code=400, detail="Only .xlsx files allowed")

    auto_cfg = config.get("data", {}).get("deeptech_auto_discovery", {})
    filename_regex_text = str(auto_cfg.get("filename_regex", DEFAULT_DEEPTECH_FILENAME_REGEX))
    try:
        filename_regex = re.compile(filename_regex_text, re.IGNORECASE)
    except re.error:
        filename_regex = re.compile(DEFAULT_DEEPTECH_FILENAME_REGEX, re.IGNORECASE)

    if not filename_regex.match(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Filename must match {DeepTech_Field}_{year}.xlsx, e.g. EAS_2025.xlsx",
        )
    
    max_size = config.get("file_upload", {}).get("deeptech_max_size_mb", 50) * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail=f"File exceeds {max_size/1024/1024}MB limit")
    
    destination_dir = resolve_project_path(config.get("file_upload", {}).get("deeptech_destination", "data/raw/"))
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / file.filename
    
    with open(destination, "wb") as f:
        f.write(content)
    
    return FileUploadResponse(
        status="success",
        filename=file.filename,
        message="File uploaded successfully"
    )


@app.post("/api/upload/startup-xlsx", response_model=FileUploadResponse, tags=["Database Update"])
async def upload_startup_xlsx(file: UploadFile = File(...)):
    file_cfg = config.get("file_upload", {})
    allowed_exts = file_cfg.get("startup_allowed_extensions", [".xlsx", ".xls"])
    if not any(file.filename.lower().endswith(str(ext).lower()) for ext in allowed_exts):
        raise HTTPException(status_code=400, detail="Only .xlsx files allowed")

    startup_auto_cfg = config.get("data", {}).get("startup_auto_discovery", {})
    filename_regex_text = str(startup_auto_cfg.get("filename_regex", DEFAULT_STARTUP_FILENAME_REGEX))
    try:
        filename_regex = re.compile(filename_regex_text, re.IGNORECASE)
    except re.error:
        filename_regex = re.compile(DEFAULT_STARTUP_FILENAME_REGEX, re.IGNORECASE)

    if not filename_regex.match(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Filename must match startup_{year}.xlsx, e.g. startup_2026.xlsx",
        )

    max_size = file_cfg.get("startup_max_size_mb", 50) * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail=f"File exceeds {max_size/1024/1024}MB limit")

    destination_dir = resolve_project_path(
        file_cfg.get("startup_destination", file_cfg.get("deeptech_destination", "data/raw/"))
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / file.filename

    with open(destination, "wb") as f:
        f.write(content)

    # Startup upload is independent from database update; refresh engines immediately.
    refresh_runtime_engine()

    return FileUploadResponse(
        status="success",
        filename=file.filename,
        message="File uploaded successfully"
    )

async def _execute_update_pipeline(task_id: str, username: str, input_csv_path: str, deeptech_xlsx_path: str):
    """后台执行数据库更新流程"""
    try:
        orchestrator = DatabaseUpdateOrchestrator(config, ws_manager, username)
        result = await orchestrator.run_update_pipeline(input_csv_path, deeptech_xlsx_path)

        # Refresh in-memory engine so newly uploaded DeepTech files are recognized immediately.
        refresh_runtime_engine()
        
        active_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "current_stage": "completed",
            "markdown_content": result.markdown_content,
            "summary_stats": result.summary_stats
        })
        
        await ws_manager.send_completion(username, result.markdown_content, result.summary_stats)
    
    except Exception as e:
        error_msg = str(e)
        active_tasks[task_id].update({
            "status": "failed",
            "error_message": error_msg
        })
        await ws_manager.send_error(username, error_msg)


@app.post("/api/start-update", response_model=UpdateStartResponse, tags=["Database Update"])
async def start_database_update(request: UpdateStartRequest, authorization: str = Header(None)):
    token = extract_token_from_header(authorization)
    username = verify_token(token, config)
    
    input_path = resolve_project_path(config.get("file_upload", {}).get("input_csv_destination", "./")) / request.input_csv_filename
    
    if not input_path.exists():
        raise HTTPException(status_code=400, detail="input.csv not found")

    deeptech_path = None
    if request.deeptech_xlsx_filename:
        dt_path = resolve_project_path(config.get("file_upload", {}).get("deeptech_destination", "data/raw/")) / request.deeptech_xlsx_filename
        if dt_path.exists():
            deeptech_path = dt_path
        else:
            raise HTTPException(status_code=400, detail="DeepTech file not found")
    
    task_id = str(uuid.uuid4())
    active_tasks[task_id] = {
        "user_id": username,
        "status": "started",
        "progress": 0,
        "current_stage": None,
        "current_professor": None
    }
    
    asyncio.create_task(
        _execute_update_pipeline(task_id, username, str(input_path), str(deeptech_path) if deeptech_path else "")
    )
    
    return UpdateStartResponse(
        task_id=task_id,
        status="started",
        message="Database update started"
    )

@app.get("/api/update-result/{task_id}", response_model=UpdateResultResponse, tags=["Database Update"])
async def get_update_result(task_id: str, authorization: str = Header(None)):
    token = extract_token_from_header(authorization)
    username = verify_token(token, config)
    
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    
    if task["user_id"] != username:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    return UpdateResultResponse(
        status=task.get("status", "unknown"),
        progress_pct=task.get("progress"),
        current_stage=task.get("current_stage"),
        current_professor=task.get("current_professor"),
        markdown_content=task.get("markdown_content"),
        summary_stats=task.get("summary_stats"),
        error_message=task.get("error_message")
    )


def get_iso_timestamp():
    return datetime.utcnow().isoformat() + "Z"

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    try:
        username = verify_token(token, config)
    except HTTPException:
        await websocket.close(code=1008, reason="Unauthorized")
        return
    
    await websocket.accept()
    await ws_manager.connect(websocket, username)
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "subscribe_progress":
                pass
            elif message_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": get_iso_timestamp()})
    
    except WebSocketDisconnect:
        await ws_manager.disconnect(username)
