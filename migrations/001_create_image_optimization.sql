-- Tao metadata job/outbox; khong luu binary image, prompt raw hay signed URL.
CREATE TABLE IF NOT EXISTS image_optimization_jobs (
    job_id uuid PRIMARY KEY,
    seller_owner_id uuid NOT NULL,
    product_id uuid NOT NULL,
    source_asset_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    requested_modes jsonb NOT NULL DEFAULT '[]'::jsonb,
    generated_asset_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    generated_assets jsonb NOT NULL DEFAULT '[]'::jsonb,
    status varchar(32) NOT NULL,
    idempotency_key varchar(180) NOT NULL UNIQUE,
    provider varchar(80),
    model varchar(120),
    prompt_version varchar(40),
    attempt integer NOT NULL DEFAULT 0,
    failure_code varchar(80),
    expected_product_updated_at timestamptz,
    created_at timestamptz NOT NULL,
    started_at timestamptz,
    completed_at timestamptz,
    retention_expires_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_image_optimization_jobs_seller_status
    ON image_optimization_jobs (seller_owner_id, status);

CREATE TABLE IF NOT EXISTS image_optimization_outbox_events (
    event_id uuid PRIMARY KEY,
    aggregate_id uuid NOT NULL,
    event_type varchar(120) NOT NULL,
    payload jsonb NOT NULL,
    published_at timestamptz,
    attempts integer NOT NULL DEFAULT 0,
    last_error text
);

CREATE INDEX IF NOT EXISTS idx_image_optimization_outbox_unpublished
    ON image_optimization_outbox_events (published_at) WHERE published_at IS NULL;
