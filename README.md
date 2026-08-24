<div align="center">

# ✦ AI Service

### AI assistance for Bin E-Commerce seller workflows.

<p>
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-Vision-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img alt="Redis" src="https://img.shields.io/badge/Redis-Cache-DC382D?style=for-the-badge&logo=redis&logoColor=white" />
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#what-it-does">What it does</a> ·
  <a href="#technology">Technology</a> ·
  <a href="#integration">Integration</a>
</p>

</div>

---

## Overview

AI Service provides a dedicated backend for seller and customer AI features. The first feature will help sellers generate product name suggestions from product images, category information, brand data, and seller-provided content.

## What It Does

- Generates three Vietnamese product name suggestions from up to three CDN images.
- Uses product category, brand, description, and attributes as additional context.
- Returns one recommended suggestion with an explanation.
- Detects and removes sensitive data such as UUIDs, URLs, API keys, emails, phone numbers, and internal identifiers.
- Applies request validation, seller rate limits, response caching, and provider timeouts.
- Keeps OpenAI credentials on the server and never exposes them to the web application.

## Technology

| Area | Technology |
| --- | --- |
| Runtime | Python 3.12 |
| HTTP API | FastAPI |
| Validation | Pydantic v2 |
| AI provider | OpenAI Vision model (`gpt-4.1-mini`) |
| Cache and rate limit | Typed in-memory adapters (Redis-ready) |
| Testing | Pytest and pytest-asyncio |
| Code quality | Ruff and Mypy |
| Deployment | Local process for now; container deployment planned later |

## Integration

```text
Seller Center (Next.js)
        ↓
API Gateway (NestJS)
        ↓
AI Service (FastAPI)
        ↓
OpenAI Vision Model
```

The service is accessed through API Gateway. Gateway authentication and the `seller.ai.product_content.generate` permission protect seller requests before they reach the AI provider.

## Structure

```text
app/
├── core/
│   ├── config/                    # Runtime settings and environment loading
│   ├── dependencies/              # FastAPI dependency wiring
│   ├── errors/                    # Safe application and infrastructure errors
│   ├── logging/                   # Logging setup and observability primitives
│   └── security/                  # Gateway user context and permission checks
└── modules/
    └── product_content/
        ├── domain/               # Framework-independent business rules
        ├── application/          # Use cases and workflow orchestration
        ├── infrastructure/      # OpenAI, Redis, cache and rate-limit adapters
        └── presentation/         # FastAPI routes and Pydantic schemas
tests/
├── unit/                         # Isolated domain and application tests
└── integration/                  # HTTP and infrastructure boundary tests
```

New capabilities such as buyer assistance, search, and recommendations should be added as sibling modules under `app/modules/` without coupling them to `product_content`.

## API

Internal endpoint:

```text
POST /api/v1/seller/product-content/name-suggestions
```

The API accepts category data, optional brand and seller input, and up to three HTTPS image URLs from the configured media CDN.

## LLM Provider Architecture

The application depends on the `LLMNameSuggestionProvider` port, not on an OpenAI SDK directly. The current `OpenAINameSuggestionProvider` is the first adapter. Future providers such as Anthropic, Gemini, or a self-hosted model can implement the same port and be selected through `LLM_PROVIDER` without changing the HTTP contract or use case.

```text
Presentation → Application Use Case → LLM Provider Port
                                      ├── OpenAI adapter
                                      ├── Anthropic adapter (future)
                                      └── Local model adapter (future)
```

## Local Commands

Run these commands from `services/ai-service`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python -m uvicorn app.main:app --reload --port 3009
```

After the virtual environment and dependencies are ready, the service can also be started with the same command style as the Node.js services:

```cmd
npm run dev
```

The `npm` script only orchestrates the Python process; FastAPI still runs through Uvicorn. For a production-style process, use:

```cmd
npm run start
```

Quality checks:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy app
.\.venv\Scripts\python -m compileall -q app
```

## API Key Setup

1. Create an API key in the OpenAI Platform dashboard.
2. Put it only in the local `services/ai-service/.env` file as `OPENAI_API_KEY=...`.
3. Set `MEDIA_PUBLIC_CDN_URL` to the HTTPS origin that owns product media.
4. Never put the key in Next.js, API Gateway, Git, or committed `.env` files.

The OpenAI Python SDK reads the key from the environment; the service never returns or logs it.

## Current Status

The first seller product-name suggestion use case is implemented. The service remains intentionally stateless and does not persist prompts, images, or product content.
