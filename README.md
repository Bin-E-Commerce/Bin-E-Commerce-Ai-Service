<div align="center">

<img src="https://raw.githubusercontent.com/Bin-E-Commerce/Bin-E-Commerce-UI-Web/main/public/images/logo/logo_background_white.png" alt="Bin E-Commerce" width="190" />

# AI Service

**Turn product images and customer intent into safer seller tools and smarter recommendations.**

<p>
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-Provider-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-Async-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img alt="Apache Kafka" src="https://img.shields.io/badge/Kafka-Workers-231F20?style=for-the-badge&logo=apachekafka&logoColor=white" />
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-Ranking-2E8B57?style=for-the-badge" />
</p>

</div>

---

## Contents

1. [Problem](#1-problem)
2. [Service at a glance](#2-service-at-a-glance)
3. [Core capabilities](#3-core-capabilities)
4. [Trust surface](#4-trust-surface)
5. [See It Work](#5-see-it-work)
6. [Install](#6-install)
7. [Getting Started](#7-getting-started)
8. [How It Works](#8-how-it-works)
9. [Service Boundary](#9-service-boundary)
10. [Service Communication](#10-service-communication)
11. [Data Ownership](#11-data-ownership)
12. [API Reference](#12-api-reference)
13. [Project Structure](#13-project-structure)
14. [Configuration Reference](#14-configuration-reference)
15. [Development](#15-development)
16. [Engineering Decisions](#16-engineering-decisions)
17. [Operational Notes](#17-operational-notes)
18. [Documentation Findings](#18-documentation-findings)
19. [FAQ](#19-faq)
20. [Ownership](#20-ownership)

---

## 1. Problem

AI calls must not live in the browser or be mixed into commerce services. Provider credentials, prompt limits, image validation, model failures, rate limits, and model versions need one server-side boundary that can fail safely.

AI Service provides that boundary for Bin E-Commerce. It combines synchronous seller APIs with independent Kafka workers, while keeping external provider details behind application ports and returning stable contracts to callers.

The design follows familiar Hexagonal/Clean Architecture ideas: the domain and use cases do not import FastAPI, OpenAI, Kafka, SQLAlchemy, or Redis. The concrete adapters are composed at the application entrypoint.

## 2. Service at a glance

| | |
| --- | --- |
| **Domain** | Seller AI assistance and recommendation infrastructure |
| **HTTP runtime** | FastAPI + Uvicorn |
| **Language** | Python 3.12 |
| **Persistence** | PostgreSQL through SQLAlchemy async adapters |
| **Cache and limits** | Redis in service mode; in-memory adapters only in explicit memory mode |
| **Messaging** | Kafka through `aiokafka` |
| **AI providers** | Configured OpenAI text/image/embedding adapters; provider ports remain replaceable |
| **Ranking** | LightGBM artifact with deterministic `ranking-fallback-v1` scorer |
| **Default port** | `3009` |
| **Entry point** | `app.entrypoints.api:app` |

## 3. Core capabilities

| Capability | What it does | Why it matters |
| --- | --- | --- |
| Product name suggestions | Produces three Vietnamese suggestions from category, seller context, and approved CDN images | Reduces repetitive seller input while preserving a review step |
| Product descriptions | Generates one preview description with public warnings | Keeps generated content inside a bounded, reviewable contract |
| Image optimization | Creates white-background or lifestyle outputs as reviewable jobs | Keeps originals untouched until a seller explicitly applies results |
| Embeddings | Consumes recommendation embedding jobs and publishes versioned vectors | Moves provider latency and retry work off the request path |
| ML ranking | Scores a bounded batch of normalized feature vectors | Lets Recommendation Service trial a model without exposing model internals |
| Safe fallback | Uses deterministic output when the model artifact is missing or fails | Preserves the downstream ranking contract during rollout and incidents |

## 4. Trust surface

> **What this service touches:** configured OpenAI endpoints, Media Service, Product Service, PostgreSQL, Redis, and Kafka. It also reads local model artifacts when `RANKING_MODEL_PATH` is set.
>
> **What leaves the process:** bounded seller context and approved CDN image URLs may be sent to the configured AI provider. Raw secrets, access tokens, internal headers, and database credentials must never enter prompts or event payloads.
>
> **Permission surface:** seller HTTP routes depend on Gateway-forwarded user and permission headers. Ranking and model-status routes require `x-internal-service-token`; they are not browser APIs.
>
> **Reversibility:** set `AI_RUNTIME_MODE=memory` for a dependency-light local process, set AI feature flags off at the caller/policy boundary, stop workers independently, or remove the configured model path to return ranking to deterministic fallback. Do not delete production data to disable a provider.

## 5. See It Work

### Start a dependency-light API

The memory runtime is intended for local route development and tests. It does not fabricate provider content when an API key is absent.

```powershell
cd services/ai-service
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set these local values in `.env`:

```env
AI_RUNTIME_MODE=memory
NODE_ENV=development
PORT=3009
```

Then run:

```powershell
npm run dev
```

```powershell
curl http://localhost:3009/api/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "ai-service"
}
```

### Check model readiness

The model status route is internal and uses the same token boundary as prediction:

```powershell
curl http://localhost:3009/api/v1/ranking/status `
  -H "x-internal-service-token: replace-with-local-token"
```

When the artifact is missing or incompatible, the safe status is equivalent to:

```json
{
  "ready": false,
  "fallback": true,
  "modelVersion": "ranking-fallback-v1",
  "featureCount": 9
}
```

### Predict a ranking batch

The current contract accepts exactly nine finite features per item, keeps input order, and clamps every score to `[0, 1]`.

```powershell
curl -X POST http://localhost:3009/api/v1/ranking/predict `
  -H "Content-Type: application/json" `
  -H "x-internal-service-token: replace-with-local-token" `
  -d '{
    "requestId": "demo-ranking-request",
    "items": [
      {
        "itemId": "product-001",
        "features": [0.90, 0.70, 0.80, 0.40, 0.60, 0.50, 0.75, 0.30, 0.10]
      }
    ]
  }'
```

The response includes `requestId`, `modelVersion`, and one prediction per item. With no compatible LightGBM artifact, `modelVersion` is `ranking-fallback-v1`.

### Run the included demo model

The repository includes a small synthetic dataset at
`data/ranking.demo.jsonl` so the LightGBM path can be demonstrated without
exposing real customer data. It is not a production-trained model.

The nine values must keep this order:

| Position | Feature |
| ---: | --- |
| 1 | Profile affinity |
| 2 | Session context |
| 3 | Semantic similarity |
| 4 | Co-behavior |
| 5 | Popularity |
| 6 | Freshness |
| 7 | Quality |
| 8 | Exploration |
| 9 | Negative penalty |

Build the image and run the offline training job from `services/ai-service`:

```powershell
docker build -t bin-ecommerce/ai-service:ranking-demo .
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/artifacts:/app/artifacts" `
  bin-ecommerce/ai-service:ranking-demo `
  python -m app.entrypoints.training.ranking_train `
  --input /app/data/ranking.demo.jsonl `
  --output /app/artifacts/ranking.txt `
  --version ranking-lgbm-demo-v1
```

Set `RANKING_MODEL_PATH` to `artifacts/ranking.txt` for a local process or
`/app/artifacts/ranking.txt` in Docker. The runtime loads the artifact only
when its feature count is nine; otherwise it uses the safe fallback.

## 6. Install

### Prerequisites

- Python `3.12`.
- A virtual environment for local development.
- PostgreSQL, Redis, and Kafka only for `AI_RUNTIME_MODE=service` or worker execution.
- OpenAI credentials only for provider-backed content, image, or embedding workflows.

### Local installation

```powershell
cd services/ai-service
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

The project also contains `requirements.txt` for the existing pip workflow:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### Runtime modes

| Mode | Required infrastructure | Intended use |
| --- | --- | --- |
| `memory` | No database, Redis, token, or OpenAI key for boot | Unit tests and dependency-light HTTP development |
| `service` | `DATABASE_URL`, `REDIS_URL`, `INTERNAL_SERVICE_TOKEN`, `OPENAI_API_KEY`; encryption key when image optimization is enabled | Integrated local stack and deployed runtime |

`application_lifespan` validates service mode before opening external connections. Missing required settings fail fast instead of silently running with unsafe adapters.

## 7. Getting Started

1. Start with `AI_RUNTIME_MODE=memory` and verify `/api/health`.
2. Read the route schemas before connecting a frontend or another service.
3. For seller content, configure `MEDIA_PUBLIC_CDN_URL` and the seller permission forwarded by API Gateway.
4. For image optimization, start PostgreSQL/Redis/Kafka and the `worker` plus `outbox` processes.
5. For embeddings, start `embedding-worker` and verify the requested/generated/DLQ topics.
6. For ranking rollout, train or provide a compatible nine-feature model, configure `RANKING_MODEL_PATH`, and verify `/ranking/status`.

Do not call OpenAI directly from the web client and do not treat a generated preview as an automatically applied product mutation.

## 8. How It Works

### HTTP request path

```text
API Gateway / internal caller
            │ trusted context or service token
            ▼
FastAPI router and Pydantic schema
            │ validation, allow-list and request ID
            ▼
Application use case
            ├── cache and rate limiter
            ├── provider port ──────► OpenAI adapter
            ├── media/product client ─► internal service
            └── ranking registry ───► LightGBM or fallback
            ▼
Stable response envelope
```

1. The router validates shape, size, host, permission, or internal token at the HTTP boundary.
2. A use case receives framework-independent commands.
3. Adapters perform provider, cache, database, or internal HTTP work.
4. The response mapper exposes only safe fields and the correlation `requestId`.

### Asynchronous image optimization

```text
Seller creates job
      ▼
PostgreSQL job + outbox row in one transaction
      ▼
outbox relay publishes metadata to Kafka
      ▼
image worker claims lease and resolves Product/Media assets
      ▼
provider generates output → Media Service stores asset
      ▼
seller reviews → apply / reject / rollback
```

The job state machine includes `PENDING`, `PROCESSING`, `REVIEW_REQUIRED`, `FINALIZING`, `SUCCEEDED`, `REJECTED`, `APPLIED`, `ROLLED_BACK`, and `FAILED`. Idempotency keys, leases, retries, and an outbox prevent a browser retry or Kafka redelivery from silently creating duplicate paid provider calls.

### Embedding worker

```text
Recommendation Service
  └─ recommendation.product-embedding.requested.v1
       ▼
AI embedding worker
  ├─ validate event, model version and dimensions
  ├─ reuse bounded result cache when available
  ├─ call embedding provider with retry policy
  └─ publish generated event or DLQ
       ▼
Recommendation Service → Qdrant
```

Malformed events go to the configured DLQ. Provider/transient failures are retried within the worker policy; offsets are committed only after the generated result or DLQ hand-off succeeds.

## 9. Service Boundary

### Owns

- AI provider adapters and provider-neutral application ports.
- Seller content generation use cases and public warning/safety mapping.
- Image optimization job lifecycle, outbox records, worker processing, and review transitions.
- Embedding worker transformation and version/dimension validation before publication.
- Ranking model loading, feature validation, score clamping, and model-status reporting.

### Delegates

| Responsibility | Delegated to |
| --- | --- |
| JWT verification and public route policy | API Gateway / Auth Service |
| Product ownership, product state, and inventory | Product Service |
| Upload keys, asset storage, and CDN URLs | Media Service |
| Recommendation profiles, candidates, traffic policy, and Qdrant projection | Recommendation Service |
| Canonical product catalog and seller application state | Product / Catalog / Seller Services |
| Model training data selection and offline evaluation | Offline ranking workflow; `ranking_train` only trains an artifact |

## 10. Service Communication

### Synchronous

| Direction | Dependency | Purpose |
| --- | --- | --- |
| Inbound | API Gateway | Seller content and image-optimization HTTP requests |
| Inbound | Recommendation Service | Internal ranking prediction and model status |
| Outbound | Media Service | Resolve/upload/cleanup approved media assets |
| Outbound | Product Service | Resolve product owner and source media for optimization |
| Outbound | Configured AI provider | Text, image, and embedding generation |

### Asynchronous

| Direction | Topic | Purpose |
| --- | --- | --- |
| Producer | `ai.image-optimization.requested.v1` | Queue image optimization metadata |
| Producer | `ai.image-optimization.dlq.v1` | Dead-letter malformed or exhausted image events |
| Consumer/Producer | `recommendation.product-embedding.requested.v1` → `recommendation.product-embedding.generated.v1` | Generate and publish product vectors |
| Producer | `recommendation.product-embedding.dlq.v1` | Dead-letter invalid or unrecoverable embedding events |

## 11. Data Ownership

| Storage | Ownership and purpose |
| --- | --- |
| PostgreSQL | Image optimization jobs, generated asset metadata, leases, and outbox records in service mode |
| Redis | Result cache and rate-limit state; it is not the source of truth for jobs or vectors |
| Local artifact path | Read-only ranking model artifact configured by `RANKING_MODEL_PATH` |
| Kafka | Durable transport for worker requests, generated embedding events, and DLQ records |
| S3/CDN via Media Service | Binary asset storage; AI Service keeps references and metadata rather than owning browser upload credentials |

### Main persistence concepts

- `ImageOptimizationJob`: seller-owned job aggregate and lifecycle.
- `ImageOptimizationOutboxRecord`: event delivery state and retry/dead-letter metadata.
- `GeneratedAsset`: safe reference to an output asset and its source asset.
- `RankingPredictionBatch`: in-memory application result containing model version and bounded scores.

## 12. API Reference

### Health and operations

| Method | Route | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/health` | None | Liveness of the HTTP process |
| `GET` | `/metrics` | Internal deployment | Low-cardinality Prometheus text; hidden from OpenAPI |
| `GET` | `/api/v1/ranking/status` | Internal token | Model readiness, fallback state, version, feature count |

### Seller product content

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/seller/product-content/name-suggestions` | Return three sanitized Vietnamese name suggestions |
| `POST` | `/api/v1/seller/product-content/description-suggestions` | Return one sanitized product description preview |

Request rules include 1–3 HTTPS images, configured CDN origin matching, bounded category/seller text, `vi-VN` locale, and the configured seller permission. Responses contain suggestions/description, warnings, and `requestId`; provider prompts and internal details are not exposed.

### Seller image optimization

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/seller/ai/image-optimization/jobs` | Create an idempotent asynchronous batch; requires `Idempotency-Key` |
| `GET` | `/api/v1/seller/ai/image-optimization/jobs/:jobId` | Read a seller-owned job |
| `GET` | `/api/v1/seller/ai/image-optimization/overview` | Read seller dashboard counts |
| `POST` | `/api/v1/seller/ai/image-optimization/jobs/:jobId/apply` | Apply selected generated assets with optimistic version check |
| `POST` | `/api/v1/seller/ai/image-optimization/jobs/:jobId/reject` | Reject a review result |
| `POST` | `/api/v1/seller/ai/image-optimization/jobs/:jobId/rollback` | Roll back an applied result when permitted |

Supported modes are `WHITE_BACKGROUND` and `LIFESTYLE_BACKGROUND`. Lifestyle requests use a bounded preset or description; generated assets are reviewed before apply.

### Internal ranking

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/ranking/predict` | Score up to 300 unique items, each with up to 64 features and the configured expected feature count |
| `GET` | `/api/v1/ranking/status` | Report whether a real artifact is ready or fallback is active |

The current default expected feature count is `9`. A request with duplicate IDs, mixed vector lengths, NaN/Infinity, too many items, or an invalid token is rejected before prediction.

## 13. Project Structure

```text
services/ai-service/
├── app/
│   ├── bootstrap/
│   │   ├── api.py                 # FastAPI composition root and error envelope
│   │   ├── dependencies.py        # Dependency wiring and trusted user context
│   │   └── lifespan.py             # Runtime validation and resource lifecycle
│   ├── core/
│   │   ├── config/settings.py      # Typed environment configuration
│   │   ├── errors/                 # Public-safe application errors
│   │   ├── logging/                # Logging setup
│   │   └── security/               # Header context and permission parsing
│   ├── modules/
│   │   ├── product_content/        # Seller name/description generation
│   │   ├── image_optimization/     # Job domain, use cases, adapters and router
│   │   ├── embeddings/              # Embedding provider and Kafka worker
│   │   └── ranking/                 # Prediction, model registry and training
│   ├── shared/infrastructure/      # Redis cache and rate-limit adapters
│   └── entrypoints/                # Process entrypoints grouped by runtime role
│       ├── api.py                  # FastAPI HTTP process
│       ├── workers/                # Kafka workers and outbox relay
│       └── training/               # Offline model-training commands
├── migrations/                     # Alembic schema history
├── tests/
│   ├── unit/                        # Domain/use-case/provider tests
│   └── integration/                 # HTTP route and infrastructure boundaries
├── pyproject.toml                   # Dependencies and quality gates
├── package.json                     # Python commands exposed to workspace scripts
├── Dockerfile
├── Makefile
└── .env.example
```

### Important files

| File | Why it matters |
| --- | --- |
| `app/bootstrap/api.py` | Registers routers, health, metrics, request IDs, and the stable error envelope |
| `app/bootstrap/lifespan.py` | Fails fast on incomplete service configuration and composes memory/service adapters |
| `app/modules/ranking/infrastructure/registry/model_registry.py` | Loads LightGBM only when the artifact is compatible; otherwise returns deterministic fallback |
| `app/modules/image_optimization/domain/models.py` | Protects job lifecycle, leases, source/output mapping, and apply/rollback invariants |
| `app/modules/image_optimization/infrastructure/persistence/outbox_relay.py` | Claims, retries, publishes, and dead-letters image events without holding database locks during network calls |
| `app/modules/product_content/presentation/api/router.py` | Enforces CDN host validation and maps safe seller responses |
| `app/modules/embeddings/infrastructure/messaging/worker.py` | Validates embedding events, retries provider calls, caches generated payloads, and commits Kafka offsets safely |
| `migrations/versions/` | Versioned PostgreSQL changes for image optimization persistence |

## 14. Configuration Reference

Copy [`.env.example`](./.env.example) to `.env`. The table groups the variables by responsibility; the example file remains the canonical complete inventory.

| Group | Key variables | Notes |
| --- | --- | --- |
| Runtime | `APP_NAME`, `PORT`, `NODE_ENV`, `AI_RUNTIME_MODE` | `service` is the integrated mode; `memory` is explicit local/test mode |
| Text/image provider | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_IMAGE_MODEL`, provider selectors | Secrets stay server-side |
| Content limits | `AI_MAX_IMAGES`, `AI_MAX_TEXT_CHARS`, `AI_CACHE_TTL_SECONDS`, `AI_RATE_LIMIT_*` | Bounds cost and request pressure |
| Media/Product | `MEDIA_PUBLIC_CDN_URL`, `MEDIA_SERVICE_URL`, `PRODUCT_SERVICE_URL` | CDN origins are validated before vision calls |
| Persistence | `DATABASE_URL`, `REDIS_URL` | Required in service mode |
| Internal security | `INTERNAL_SERVICE_TOKEN` | Required for ranking and trusted internal calls |
| Kafka | `KAFKA_BOOTSTRAP_SERVERS`, image and embedding topic/group variables | Worker-specific contracts are versioned |
| Embeddings | `EMBEDDING_MODEL*`, dimensions, timeout, retries, topics | Default dimensions: `1536` |
| Ranking | `RANKING_MODEL_PATH`, `RANKING_MODEL_VERSION`, `RANKING_EXPECTED_FEATURES`, limits | Default expected features: `9`; no path means fallback |
| Image jobs | `AI_IMAGE_*` | Feature flag, quotas, leases, retries, quality, output and encryption settings |

## 15. Development

### HTTP and worker commands

Run from `services/ai-service`:

```powershell
npm run dev
npm run start
npm run worker
npm run embedding-worker
npm run outbox
  npm run ranking-train -- --input data/ranking.demo.jsonl --output artifacts/ranking.txt --version ranking-lgbm-demo-v1
```

The ranking training input is JSONL with `features` and `label` fields. Training is offline; it is not imported into the FastAPI request path.

### Quality gates

```powershell
npm run lint
npm run format:check
npm run type-check
npm run architecture:check
npm run compile:check
npm run test
npm run check
```

### Testing strategy

| Test area | Coverage |
| --- | --- |
| Unit | Safety policies, prompt building, provider adapters, image state machine, cache/rate-limit behavior, ranking validation |
| Integration | Product-content, image-optimization, and ranking HTTP contracts using test application wiring |
| Architecture | Import-linter verifies domain independence from framework/infrastructure modules |
| Quality | Ruff, Mypy, UTF-8 check, compile check, and coverage through Pytest |

Tests use the application factory and do not call paid providers. Keep provider secrets out of test fixtures and logs.

## 16. Engineering Decisions

| Decision | Reason |
| --- | --- |
| Provider ports instead of SDK calls in use cases | Allows OpenAI replacement and keeps business code testable |
| Explicit memory/service runtime modes | Prevents accidental fail-open production behavior |
| Model registry with deterministic fallback | Keeps Recommendation response contract available while a model is missing or unhealthy |
| Image job + outbox in one database boundary | Prevents a committed job from losing its Kafka request silently |
| Idempotency key and job lease | Protects provider cost when HTTP retries or Kafka redelivery occur |
| Server-controlled CDN allow-list | Reduces SSRF and prevents arbitrary URLs from reaching a vision provider |
| Internal token for ranking routes | Separates service-to-service authorization from browser/user permissions |

## 17. Operational Notes

- Run the API, image worker, embedding worker, and outbox relay as separate processes when their workloads are enabled.
- Monitor provider timeout/rate-limit errors, Kafka DLQ volume, outbox attempts, job failure stages, model version, and fallback rate.
- A ranking model is considered ready only when the artifact loads and its feature count matches `RANKING_EXPECTED_FEATURES`.
- Never log prompts, raw image bytes, vectors, access tokens, provider keys, or full internal event payloads.
- If Redis or Kafka is unavailable, follow the module-specific failure policy; do not silently convert service mode to memory mode.

## 18. Documentation Findings

- `package.json` and Docker use `app.entrypoints.api:app` as the HTTP entrypoint, while the legacy `Makefile` `dev` target still references `app.main:app`. Use `npm run dev` until that target is aligned.
- Service mode validates `DATABASE_URL`, `REDIS_URL`, `INTERNAL_SERVICE_TOKEN`, and `OPENAI_API_KEY`; image optimization additionally requires `AI_IMAGE_BACKGROUND_ENCRYPTION_KEY` when enabled.
- The HTTP health route is `/api/health`; `/metrics` is intentionally hidden from the generated OpenAPI schema.

## 19. FAQ

### Does the browser call AI Service directly?

No. Browser traffic goes through API Gateway, which supplies trusted user context and the appropriate permission boundary.

### What happens when the ranking model is unavailable?

`registry/model_registry.py` returns the deterministic `ranking-fallback-v1` implementation. Recommendation Service can therefore continue using its Standard/Hybrid baseline without recording a false ML result.

### Are generated images applied automatically?

No. Image optimization creates reviewable outputs. The seller must select/apply them, and optimistic version checks protect against overwriting a newer product state.

### Can I run the service without PostgreSQL and Redis?

Yes, for local route/test work, by explicitly setting `AI_RUNTIME_MODE=memory`. Integrated service mode requires the configured dependencies and fails fast when they are missing.

## 20. Ownership

### Engineering

**Đào Ngọc Anh**

**Software Engineer**

[View portfolio](https://daongocanh.site)

Software Engineer responsible for the architecture, implementation, integration, and maintenance of this service.

### Architecture & API Design

**Đào Ngọc Anh**

Designed the provider boundary, worker contracts, ranking interface, persistence strategy, and integration with the Bin Ecommerce ecosystem.
