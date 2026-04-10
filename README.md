# Professor-Industry Matching System

A GraphRAG-powered intelligent matching platform that connects enterprises with academic research expertise at HKUST. The system combines semantic search, knowledge graph integration, priority-based ranking, startup search, and a real-time database update pipeline.

## System Overview

The Professor-Industry Matching System is built to bridge industry innovation needs with academic research strengths at HKUST. It combines:

- Semantic intelligent search for enterprise queries
- Dynamic professor knowledge graphs built from research interests, publications, projects, and department relationships
- Strategic priority ranking that balances semantic relevance with career stage, title, and engineering focus
- Multi-source DeepTech integration from EAS, BMH, and MES Excel sources
- Startup search with an independent startup graph and matching pipeline
- JWT-based authentication and session persistence
- WebSocket-powered progress reporting for long-running database updates
- Incremental data scraping and update orchestration

## Core Capabilities

### 1. Intelligent Query Processing
The system accepts enterprise research queries and automatically:
- Validates relevance to academic research domains
- Classifies query strength as `VALID`, `WEAK_RELEVANCE`, `INVALID`, or `NEEDS_CLARIFICATION`
- Extracts domain keywords with stopword filtering
- Provides suggestions for ambiguous or weak queries

### 2. Semantic Similarity Matching
- Encodes professor profiles using Sentence-Transformers (`all-mpnet-base-v2` by default)
- Computes cosine similarity between query vectors and professor expertise vectors
- Incorporates research interests, publications, leading projects, and DeepTech initiatives
- Supports a TfidfVectorizer fallback when the transformer backend is unavailable

### 3. Knowledge Graph Enhancement
- Builds a NetworkX graph with professors as nodes
- Creates weighted edges from:
  - Jaccard similarity of research interest tokens
  - Publication and project overlap
  - DeepTech cluster associations
  - Department-level connections
- Applies graph neighbor boost during ranking

### 4. Priority-Based Strategic Ranking
- Scores professors using a combination of:
  - Years since PhD
  - Title mapping
  - Engineering background
- Blends semantic relevance and priority scores through configurable weights

### 5. Multi-Source DeepTech Project Integration
- Loads DeepTech sources from year-suffixed Excel files in `data/raw/`
- Supports source tracking for EAS, BMH, and MES projects
- Preserves project metadata such as TRL, IP status, applications, industries, and technology overview

### 6. Secure User Authentication
- Uses JWT tokens for API access
- Loads credentials from `config/config.yaml`
- Supports token TTL configuration and browser session persistence

### 7. Multi-Format File Upload
- Accepts professor information CSV uploads
- Accepts DeepTech Excel uploads
- Uses multipart form-data with token-based authorization
- Routes uploads to the configured destinations automatically

### 8. Real-Time Database Update Pipeline
- Runs a three-stage asynchronous scraping workflow
- Tracks progress with WebSocket updates
- Updates progress per professor, not just per stage
- Generates Markdown summaries and stores logs in `logs/`

### 9. Interactive Multi-Page Web Interface
- Professor search page with ranking controls and result cards
- Startup search page with startup cards and contact actions
- Database update page with file uploads, progress bar, and update results
- Sidebar navigation with modal-based login

### 10. Startup Search (Mode-Based Matching)
- Auto-discovers startup Excel files matching `startup_YYYY.xlsx`
- Uses an independent startup preprocessing, graph, and ranking pipeline
- Reuses `POST /match` with `mode=startup`

## Project Structure

```text
Professor_Matching_System/
├── api/
│   ├── app.py
│   ├── auth.py
│   ├── schemas.py
│   ├── websocket_manager.py
│   └── static/
│       ├── app.js
│       ├── index.html
│       └── styles.css
├── config/
│   └── config.yaml
├── data/
│   └── raw/
│       ├── BMH_2025.xlsx
│       ├── EAS_2025.xlsx
│       ├── MES_2025.xlsx
│       ├── professor_information.csv
│       ├── professor_projects.csv
│       ├── professor_publications.csv
│       └── startup_2025.xlsx
├── dev/
│   ├── DEPLOYMENT_WIN10_GUIDE.md
│   ├── QUICK_REFERENCE.md
│   ├── config.example.yaml
│   ├── requirements_cpu.txt
│   ├── requirements_gpu.txt
│   ├── run_cpu.bat
│   └── run_gpu.bat
├── docs/
│   ├── api_documentation.md
│   ├── Database-Dynamic-Update-Guide.pdf
│   ├── Professor-Industry Matching System.pptx
│   └── System-Startup-Guide.pdf
├── plans/
│   ├── plan-databaseUpdateIntegration.prompt.md
│   ├── plan-dynamicProfessorDataUpdate.prompt.md
│   ├── plan-embedDeepTechProjects.prompt.md
│   ├── plan-integrateBmhMesDeepTech.prompt.md
│   ├── plan-startupSearch.prompt.md
│   └── plan-systemStartUp.prompt.md
├── src/
│   ├── __init__.py
│   ├── embedding_model.py
│   ├── evaluation.py
│   ├── orchestrator.py
│   ├── professor_graph_builder.py
│   ├── professor_matching_engine.py
│   ├── professor_preprocessing.py
│   ├── professor_priority_strategy.py
│   ├── query_processor.py
│   ├── scrape_info.py
│   ├── scrape_project.py
│   ├── scrape_publication.py
│   ├── startup_graph_builder.py
│   ├── startup_matching_engine.py
│   └── startup_preprocessing.py
├── environment.yml
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

## Backend Modules

### Professor Matching Pipeline
- `src/professor_preprocessing.py`: Loads, merges, cleans, and enriches professor records; auto-discovers DeepTech sources.
- `src/embedding_model.py`: Wraps the embedding backend and provides Sentence-Transformers / TF-IDF fallback behavior.
- `src/professor_graph_builder.py`: Builds the professor knowledge graph.
- `src/professor_matching_engine.py`: Performs validation, semantic scoring, graph boosting, and DeepTech project ranking.
- `src/professor_priority_strategy.py`: Calculates priority scores from title, years since PhD, and engineering background.
- `src/query_processor.py`: Handles query validation and keyword extraction.

### Startup Matching Pipeline
- `src/startup_preprocessing.py`: Loads startup records and auto-discovers `startup_YYYY.xlsx` sources.
- `src/startup_graph_builder.py`: Builds the startup graph.
- `src/startup_matching_engine.py`: Executes startup matching and result formatting.

### Data Update Pipeline
- `src/scrape_info.py`: Scrapes professor information and detects additions or departures.
- `src/scrape_publication.py`: Scrapes professor publications.
- `src/scrape_project.py`: Scrapes professor projects.
- `src/orchestrator.py`: Orchestrates the update workflow and progress reporting.
- `src/evaluation.py`: Evaluation and testing helpers.

### API and Frontend
- `api/app.py`: FastAPI application, REST endpoints, and WebSocket endpoint.
- `api/auth.py`: Login, JWT creation, and token validation.
- `api/websocket_manager.py`: WebSocket connection management.
- `api/schemas.py`: Request and response schemas.
- `api/static/index.html`: Multi-page UI shell.
- `api/static/app.js`: Frontend behavior for auth, matching, uploads, and updates.
- `api/static/styles.css`: UI styling and responsive layout.

## Data Models

### ProfessorRecord
```python
name: str
department: str
research_interests: str
title: str
url: str
is_engineering: bool
years_since_phd: Optional[int]
priority_score: float
attributes: Dict[str, str]
deeptech_projects: List[DeepTechProject]
```

### DeepTechProject
```python
source: str
cluster: str
technology_title: str
trl: str
ip_status: str
overview: str
tech_edges: str
applications: List[str]
industries: List[str]
```

### StartupRecord
```python
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
```

### Query Status Classification
```python
VALID
WEAK_RELEVANCE
INVALID
NEEDS_CLARIFICATION
```

## Configuration Highlights

The main configuration file is `config/config.yaml`.

- `data.raw_csv`, `data.projects_csv`, `data.publications_csv`
- `data.deeptech_auto_discovery` and `data.startup_auto_discovery`
- `embedding.model_name`, `embedding.batch_size`, `embedding.attribute_weights`
- `deeptech.columns`
- `startup.columns`, `startup.graph`, `startup.matching`, `startup.embedding_weights`, `startup.semantic_matching`
- `graph.similarity_threshold`, `graph.department_edge_weight`
- `matching.alpha`, `matching.beta`, `matching.graph_neighbor_weight`
- `priority.w_years`, `priority.w_title`, `priority.w_engineering`
- `query.enable_validation`, `query.enable_keyword_extraction`, `query.similarity_threshold`, `query.weak_threshold`
- `auth.credentials`, `auth.jwt_secret`, `auth.token_ttl_minutes`, `auth.session_ttl_hours`
- `file_upload.input_csv_destination`, `file_upload.deeptech_destination`, `file_upload.temp_directory`
- `database_update.logs_directory`, `database_update.script_timeout_seconds`, `database_update.progress_update_interval`
- `websocket.connection_timeout`, `websocket.heartbeat_interval`

## System Workflows

### Startup Sequence
1. Load configuration from `config/config.yaml`.
2. Load and merge professor info, publications, and project CSVs.
3. Auto-discover DeepTech and startup Excel files in `data/raw/`.
4. Clean and merge records.
5. Build professor and startup records.
6. Assign priority scores to professors.
7. Initialize embedding models and build embeddings.
8. Construct the professor graph.
9. Start the FastAPI app and WebSocket manager.

### Query Workflow
1. The user submits a query from the Professor Search or Startup Search page.
2. The frontend sends `POST /match` with `mode=professor` or `mode=startup`.
3. The query is validated and keywords are extracted when enabled.
4. The matching engine calculates semantic similarity, graph boost, and priority scores.
5. Results are returned as professor cards or startup cards.

### Database Update Workflow
1. The user opens the Database Update page.
2. CSV and DeepTech Excel files are uploaded with the dedicated upload endpoints.
3. `POST /api/start-update` starts the update workflow.
4. WebSocket progress is connected as a best-effort channel and does not block task start.
5. Stage 1 scrapes professor information.
6. Stage 2 scrapes publications for new professors.
7. Stage 3 scrapes projects and updates the project CSV.
8. A Markdown report is written to `logs/` and shown in the UI.

### Source Auto-Discovery
1. `data.deeptech_auto_discovery` scans `data/raw/` for files matching `EAS_YYYY.xlsx`, `BMH_YYYY.xlsx`, and `MES_YYYY.xlsx`.
2. `data.startup_auto_discovery` scans `data/raw/` for `startup_YYYY.xlsx` files.
3. Column mappings are resolved from `deeptech.columns` and `startup.columns`.
4. New source files are picked up automatically on startup.

## Getting Started

### Prerequisites
- Python 3.10+
- Conda, venv, or another virtual environment manager
- Enough RAM for the embedding model and scraping workflow

### Quick Start
```bash
# Match professors from the command line
python main.py --query "autonomous vehicles" --top-k 5

# Start the API server
uvicorn api.app:app --reload
```

### Run Tests
```bash
pytest
```

## Customization & Extensibility

### Adjust Matching Parameters
Edit `config/config.yaml`:
- `matching.alpha` and `matching.beta` control the balance between semantic relevance and priority scoring.
- `matching.graph_neighbor_weight` controls how much graph neighbors influence the final score.
- `embedding.attribute_weights` controls the relative impact of DeepTech, publication, research-interest, project, department, and title text.

### Add New Data Sources
- Place Excel files in `data/raw/` using the configured filename patterns.
- Update `deeptech.columns` or `startup.columns` if spreadsheet headers differ.
- Adjust the regex under `data.deeptech_auto_discovery` or `data.startup_auto_discovery` if you introduce a new source family.

### Extend Query Processing
- Modify `src/query_processor.py` to tune classification or keyword extraction.
- Adjust thresholds in `query` inside `config/config.yaml`.

### Extend Priority Logic
- Modify `src/professor_priority_strategy.py` if you want different ranking rules.

## Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| System startup | 30-60s | First run may download embedding weights |
| Single query execution | 1-3s | Embedding + graph + ranking |
| Batch query (10 queries) | 5-10s | Embedding amortization helps |
| Incremental update (10 new professors) | 5-15 min | Depends on source site responsiveness |
| Full data refresh | Variable | Depends on the number of new records and network speed |

## Troubleshooting & Support

For detailed troubleshooting and operational guidance, refer to:
- `docs/System-Startup-Guide.pdf` for installation and initial configuration
- `docs/Database-Dynamic-Update-Guide.pdf` for upload and update operations
- `docs/api_documentation.md` for endpoint and schema reference
- `dev/QUICK_REFERENCE.md` for day-to-day operational notes

## Citation & Attribution

This system implements GraphRAG principles for academic-industry matching with custom extensions for knowledge graph construction, strategic ranking, startup search, and real-time data management. Developed by the OKT (Open Knowledge Transfer) initiative at HKUST.

## License & Support

Internal use only. Contact the OKT team for licensing inquiries or feature requests.
