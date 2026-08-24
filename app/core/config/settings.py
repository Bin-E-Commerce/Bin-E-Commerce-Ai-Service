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

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 20.0
    llm_provider: str = "openai"

    ai_max_images: int = 3
    ai_max_text_chars: int = 6000
    ai_cache_ttl_seconds: int = 600
    ai_rate_limit_requests: int = 5
    ai_rate_limit_window_seconds: int = 600

    media_public_cdn_url: str = ""
    redis_url: str | None = None

    required_permission: str = "seller.ai.product_content.generate"

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
