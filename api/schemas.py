from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.8
    beta: float = 0.2
    graph_neighbor_weight: float = 0.1
    validate_query: bool = True
    use_keyword_extraction: bool = True
    mode: Literal["professor", "startup"] = "professor"


class ProfessorMatchItem(BaseModel):
    name: str
    department: str
    title: str
    url: str
    research_interests: str
    score: float
    similarity: float
    priority_score: float
    deeptech_projects: List["ProfessorDeepTechItem"] = Field(default_factory=list)


class ProfessorDeepTechItem(BaseModel):
    source: str = "EAS"
    cluster: str
    technology_title: str
    trl: str
    ip_status: str
    overview: str
    tech_edges: str
    applications: List[str]
    industries: List[str]
    relevance_score: float


class KeywordItem(BaseModel):
    keyword: str
    weight: float


class StartupItem(BaseModel):
    startup_id: str
    company_name: str
    website: Optional[str] = None
    people: List[str] = Field(default_factory=list)
    ref_code: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    source_year: Optional[int] = None
    description: str = ""
    tels: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    funding: Optional[str] = None
    background_year: Optional[str] = None
    matched_keywords: List[str] = Field(default_factory=list)
    score: float


class MatchResponse(BaseModel):
    query: str
    mode: str
    status: str  # "valid", "invalid", "weak_relevance", "needs_clarification"
    message: str
    suggestions: List[str]
    results: List[ProfessorMatchItem]
    startup_results: List[StartupItem] = Field(default_factory=list)
    keywords: List[KeywordItem]
    enhanced_query: str

class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """登录响应"""
    token: str
    username: str
    expires_in: int  # token有效期（秒）

class FileUploadResponse(BaseModel):
    """文件上传响应"""
    status: str  # "success" 或 "error"
    filename: str  # 保存后的文件名
    message: Optional[str] = None  # 错误信息（若有）

class UpdateStartRequest(BaseModel):
    """启动数据库更新请求"""
    input_csv_filename: str  # 已上传的input.csv文件名
    deeptech_xlsx_filename: Optional[str] = None  # 已上传的xlsx文件名

class UpdateStartResponse(BaseModel):
    """启动更新响应"""
    task_id: str  # 后台任务ID（UUID）
    status: str  # "started"
    message: str

class UpdateResultResponse(BaseModel):
    """获取更新结果响应"""
    status: str  # "processing" / "completed" / "failed"
    progress_pct: Optional[float] = None  # 当前进度百分比
    current_stage: Optional[str] = None  # 当前阶段
    current_professor: Optional[str] = None  # 当前处理的教授名字
    markdown_content: Optional[str] = None  # 完成后的报告内容
    summary_stats: Optional[dict] = None  # 统计摘要
    error_message: Optional[str] = None  # 错误信息（若失败）
