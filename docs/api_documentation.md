# API Documentation

## Base

- Local development: `http://localhost:8000`

## Health Check

### `GET /health`

Returns service health.

Response:

```json
{
  "status": "ok"
}
```

## Match Endpoint

### `POST /match`

Unified matching endpoint for both professor search and startup search.

Request body:

```json
{
  "query": "medical imaging ai",
  "top_k": 5,
  "alpha": 0.85,
  "beta": 0.1,
  "graph_neighbor_weight": 0.15,
  "validate_query": true,
  "use_keyword_extraction": true,
  "mode": "startup"
}
```

Request fields:

- `query` (string): Natural-language query.
- `top_k` (int): Number of returned items.
- `alpha` (float): Semantic score weight.
- `beta` (float): Reserved for startup priority compatibility.
- `graph_neighbor_weight` (float): Graph-neighbor boost weight.
- `validate_query` (bool): Enable query validation.
- `use_keyword_extraction` (bool): Enable keyword extraction.
- `mode` (string): `"professor"` or `"startup"`.

### Response (Professor Mode)

When `mode="professor"`, `results` is populated and `startup_results` is empty.

```json
{
  "query": "robotics control",
  "mode": "professor",
  "status": "valid",
  "message": "Query is valid and relevant.",
  "suggestions": [],
  "results": [
    {
      "name": "Prof. Example",
      "department": "Mechanical Engineering",
      "title": "Associate Professor",
      "url": "https://example.edu",
      "research_interests": "robotics and control",
      "score": 0.901,
      "similarity": 0.812,
      "priority_score": 0.534,
      "deeptech_projects": []
    }
  ],
  "startup_results": [],
  "keywords": [
    {
      "keyword": "robotics",
      "weight": 0.95
    }
  ],
  "enhanced_query": "robotics control"
}
```

### Response (Startup Mode)

When `mode="startup"`, `startup_results` is populated and `results` is empty.

```json
{
  "query": "medical imaging ai",
  "mode": "startup",
  "status": "valid",
  "message": "Query is valid and relevant.",
  "suggestions": [],
  "results": [],
  "startup_results": [
    {
      "startup_id": "med-ai-lab-2025-1",
      "company_name": "Med AI Lab",
      "website": "https://med.ai",
      "people": ["Alice", "Bob"],
      "ref_code": "M-01",
      "categories": ["AI", "Healthcare"],
      "source_year": 2025,
      "description": "Medical imaging AI platform",
      "tels": ["+85212345678"],
      "emails": ["contact@med.ai"],
      "funding": "Series A",
      "background_year": "2025",
      "matched_keywords": ["medical", "imaging", "ai"],
      "score": 0.932
    }
  ],
  "keywords": [
    {
      "keyword": "medical",
      "weight": 0.92
    },
    {
      "keyword": "imaging",
      "weight": 0.88
    }
  ],
  "enhanced_query": "medical imaging ai"
}
```

### Query Validation Status

Common status values for both modes:

- `valid`
- `weak_relevance`
- `needs_clarification`
- `invalid`

If status is `invalid`, result arrays are empty and `suggestions` provides rewrite hints.

## Authentication and Database Update APIs

Database update and file upload APIs are unchanged and still require login token for protected endpoints.
