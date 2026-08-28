"""Đọc và chuẩn hóa cấu hình môi trường cho AI Service, không tạo network client."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Tập trung config giúp local, Docker và production dùng cùng tên biến mà không hard-code secret.
class Settings(BaseSettings):
    """Cấu hình runtime được đọc từ biến môi trường và file `.env` local."""

    app_name: str = "ai-service"
    port: int = 3009
    node_env: Literal["development", "test", "production"] = "development"
    ai_runtime_mode: Literal["memory", "service"] = "service"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_image_model: str = "gpt-image-2"
    openai_timeout_seconds: float = 20.0
    openai_image_timeout_seconds: float = 120.0
    openai_image_quality: Literal["low", "medium", "high"] = "medium"
    # Hồ sơ preview và final được cấu hình riêng để không phải sửa code khi điều chỉnh chi phí/latency.
    ai_image_preview_quality: Literal["low", "medium", "high"] = "low"
    # Preview lifestyle phải đủ chi tiết để seller đánh giá việc giữ nguyên sản phẩm; ảnh final vẫn tạo 1024x1024.
    ai_image_preview_size: Literal["256x256", "512x512", "1024x1024", "1536x1024", "1024x1536"] = "1024x1024"
    ai_image_preview_format: Literal["jpeg", "png", "webp"] = "jpeg"
    ai_image_preview_compression: int = 75
    ai_image_preview_max_dimension: int = 1024
    ai_image_preview_input_fidelity: Literal["low", "high"] = "high"
    # Lifestyle cần đủ thời gian cho provider hoàn tất ảnh; timeout không phải SLA hiển thị cho seller.
    ai_image_preview_timeout_seconds: float = 120.0
    ai_image_final_quality: Literal["low", "medium", "high"] = "medium"
    ai_image_final_size: Literal["256x256", "512x512", "1024x1024", "1536x1024", "1024x1536"] = "1024x1024"
    ai_image_final_format: Literal["jpeg", "png", "webp"] = "jpeg"
    ai_image_final_compression: int = 85
    ai_image_final_max_dimension: int = 1024
    ai_image_final_input_fidelity: Literal["low", "high"] = "high"
    ai_image_final_timeout_seconds: float = 120.0
    openai_image_output_hosts: str = "oaidalleapiprodscus.blob.core.windows.net,cdn.openai.com"
    llm_provider: str = "openai"
    product_name_provider: Literal["openai"] = "openai"
    product_description_provider: Literal["openai"] = "openai"
    white_background_provider: Literal["rembg"] = "rembg"
    lifestyle_background_provider: Literal["openai"] = "openai"

    ai_max_images: int = 3
    ai_max_text_chars: int = 6000
    ai_cache_ttl_seconds: int = 600
    ai_rate_limit_requests: int = 5
    ai_rate_limit_window_seconds: int = 600

    media_public_cdn_url: str = ""
    redis_url: str | None = None

    required_permission: str = "seller.ai.product_content.generate"
    ai_image_optimization_enabled: bool = True
    ai_image_max_products_per_request: int = 10
    ai_image_rate_limit_requests: int = 3
    ai_image_rate_limit_window_seconds: int = 3600
    ai_image_max_retry_attempts: int = 3
    ai_image_job_lease_seconds: int = 300
    ai_image_review_retention_days: int = 30
    ai_image_max_dimension: int = 2048
    ai_image_webp_quality: int = 88
    ai_image_worker_concurrency: int = 2
    ai_image_provider_max_concurrency: int = 2
    ai_image_outbox_poll_interval_ms: int = 250
    ai_image_kafka_poll_timeout_ms: int = 250
    ai_image_outbox_max_attempts: int = 8
    ai_image_lifestyle_max_dimension: int = 1536
    ai_image_lifestyle_jpeg_quality: int = 88
    ai_image_generated_max_bytes: int = 20_000_000
    ai_image_background_encryption_key: SecretStr | None = None
    database_url: str | None = None
    kafka_bootstrap_servers: str = "localhost:29092"
    kafka_image_optimization_topic: str = "ai.image-optimization.requested.v1"
    kafka_image_optimization_dlq_topic: str = "ai.image-optimization.dlq.v1"
    kafka_consumer_group: str = "ai-service.image-optimization-worker.v1"
    media_service_url: str = "http://localhost:3004"
    product_service_url: str = "http://localhost:3008"
    internal_service_token: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Cache object Settings để dependency dùng chung config nhưng không tạo client hay gọi mạng khi import.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Đọc settings một lần, không tạo client hoặc thực hiện network call."""

    return Settings()
