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
| Cache and rate limit | Redis (`redis.asyncio`) |
| Testing | Pytest and pytest-asyncio |
| Code quality | Ruff and Mypy |
| Deployment | Docker |

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

## API

Planned internal endpoint:

```text
POST /api/v1/seller/product-content/name-suggestions
```

The API accepts category data, optional brand and seller input, and up to three HTTPS image URLs from the configured media CDN.

## Current Status

The service currently contains documentation only. Runtime code will be added after the architecture and API contract are reviewed.
