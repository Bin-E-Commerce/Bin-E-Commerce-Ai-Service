-- Lưu preset và ciphertext của mô tả bối cảnh; không lưu prompt rõ, URL ảnh hay dữ liệu binary.
ALTER TABLE image_optimization_jobs
    ADD COLUMN IF NOT EXISTS background_preset varchar(48),
    ADD COLUMN IF NOT EXISTS background_description_ciphertext text,
    ADD COLUMN IF NOT EXISTS background_description_hash varchar(64),
    ADD COLUMN IF NOT EXISTS processing_stage varchar(32) NOT NULL DEFAULT 'QUEUED';
