# rsd-convert

**Training Package → Element-level Skill Statements (BART)**

A Streamlit web app that ingests Australian VET training package CSVs or XLSXs, auto-detects the format, normalises to a Unit → Element → Performance Criteria hierarchy, and generates one BART-framework skill statement per element using OpenAI or Anthropic Claude — with concurrent processing, QA auto-rewrite, and full Postgres persistence.

---

## What's new (v2)

| Area | Change |
|---|---|
| **Multi-provider** | OpenAI, Anthropic, or automatic fallback between them |
| **Temperature fix** | Temperature slider now correctly passed to the generator |
| **Concurrency** | ThreadPoolExecutor — up to 16 parallel API calls per batch |
| **Error handling** | Per-element try/catch + retry; one failure never kills a batch |
| **XLSX support** | Upload `.xlsx` as well as `.csv` |
| **Fingerprinting** | MD5 content hash — detects re-uploads with same shape but changed data |
| **Session isolation** | Each browser session gets a UUID; users can only resume their own runs |
| **Always-downloadable** | Results accumulate in session state even without DB |
| **Inline editing** | `st.data_editor` lets you correct statements before export |
| **Keywords module** | `core/keyword_generator.py` committed — no longer an optional stub |
| **Tests** | pytest suite covering extractors, fingerprinting, and exporters |
| **Google Cloud** | Dockerfile, Cloud Build CI/CD, Terraform for full GCP stack |

---

## Local development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest

cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY or ANTHROPIC_API_KEY

streamlit run app.py
```

Run tests:

```bash
pytest tests/ -v
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | If using OpenAI provider | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic provider | Anthropic API key |
| `DATABASE_URL` | Optional | Postgres connection string — enables persistence & resume |
| `OPENAI_MODEL` | Optional | Default model override (e.g. `gpt-4.1`) |

---

## Deploy to Google Cloud Run

### Prerequisites

- GCP project with billing enabled
- `gcloud` CLI authenticated (`gcloud auth login`)
- Terraform ≥ 1.6 (`brew install terraform` or [terraform.io](https://terraform.io))
- GitHub repo connected to Cloud Build

### Step 1 — Provision infrastructure

```bash
cd infra
terraform init
terraform apply \
  -var="project_id=YOUR_GCP_PROJECT_ID" \
  -var="github_owner=YOUR_GITHUB_USERNAME"
```

This creates:
- Artifact Registry Docker repo
- Cloud SQL Postgres 15 instance (private IP via VPC connector)
- Secret Manager secrets
- Cloud Run service (0–10 instances, 2 CPU / 2 GB RAM, 60-min timeout)
- Cloud Build trigger on `main` branch pushes

### Step 2 — Add API keys to Secret Manager

```bash
# OpenAI
echo -n "sk-..." | gcloud secrets versions add openai-api-key --data-file=-

# Anthropic
echo -n "sk-ant-..." | gcloud secrets versions add anthropic-api-key --data-file=-
```

The DATABASE_URL secret is automatically populated by Terraform from the Cloud SQL instance.

### Step 3 — First deployment

Push to `main` — Cloud Build will:
1. Run `pytest tests/`
2. Build and push the Docker image to Artifact Registry
3. Deploy to Cloud Run

Monitor at: **https://console.cloud.google.com/cloud-build/builds**

### Step 4 — Get your URL

```bash
terraform output cloud_run_url
```

Or check the Cloud Run console.

---

## Manual Docker build (optional)

```bash
# Build
docker build -t rsd-convert .

# Run locally with env file
docker run -p 8080:8080 --env-file .env rsd-convert
```

---

## Supported CSV/XLSX formats

| Extractor | Description |
|---|---|
| `training_gov_blob` | training.gov.au export — one column containing both element headings and numbered PCs |
| `row_per_pc` | Explicit Element + Performance Criteria columns, one row per PC |

Add new extractors in `core/extractors/`, implement `.score(df)` and `.extract(df)`, then register in `core/extractors/__init__.py`.

---

## Architecture

```
Upload (CSV/XLSX)
  └─► Extractor Registry (auto-score or forced)
        └─► Normalised DataFrame [unit_code, unit_title, element_title, pcs_text]
              └─► Batch controller (start_index, end_index)
                    └─► ThreadPoolExecutor (N workers)
                          ├─► OpenAI provider  ─┐
                          └─► Anthropic provider ┤─► BART generator → QA → rewrite
                                                 └─► Keyword generator (optional)
                    └─► DB upsert (Postgres / Cloud SQL)   ← session-scoped run_id
                    └─► session_state fallback (always)
                          └─► rsd_output.csv + traceability.csv
```

---

## Cost estimates (GCP, australia-southeast1)

| Resource | Est. monthly cost |
|---|---|
| Cloud Run (0–2 instances average) | ~$5–15 |
| Cloud SQL db-g1-small | ~$25 |
| Artifact Registry (a few GB) | ~$1 |
| Secret Manager | < $1 |
| Cloud Build (free tier 120 min/day) | $0 |
| **Total** | **~$30–40/month** |

Scale down Cloud SQL to `db-f1-micro` (not HA) for dev/test to save ~$10/month.

---

## Security notes

- API keys stored in Secret Manager — never in environment variables in code or Docker images
- Cloud Run service account has least-privilege IAM roles (`secretmanager.secretAccessor`, `cloudsql.client`)
- No public IP on Cloud SQL — accessible only via VPC connector
- Session-scoped run IDs prevent users from reading or overwriting each other's runs
- `allUsers` invoker role set on Cloud Run for public access — restrict to specific identities or add Cloud IAP if you need auth

---

## Adding a new extractor

1. Create `core/extractors/my_extractor.py` with a class that implements:
   - `name: str` — unique identifier
   - `version: str`
   - `score(df) -> float` — returns 0.0–1.0 confidence
   - `extract(df) -> pd.DataFrame` — returns normalised DataFrame
2. Import and add to `core/extractors/__init__.py`
3. The registry picks it up automatically on next restart
