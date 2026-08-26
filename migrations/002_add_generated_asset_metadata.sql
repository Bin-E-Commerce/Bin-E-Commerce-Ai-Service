-- Them URL/CDN metadata cho preview; khong luu binary hay response raw cua provider.
ALTER TABLE image_optimization_jobs
    ADD COLUMN IF NOT EXISTS generated_assets jsonb NOT NULL DEFAULT '[]'::jsonb;
