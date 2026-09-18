"""Kiểm tra quota image optimization theo từng môi trường triển khai."""

from app.core.config.settings import Settings


# Production phải giữ đúng hai lượt trong cửa sổ 24 giờ dù env local còn giá trị cũ.
def test_production_uses_fixed_daily_image_optimization_quota() -> None:
    settings = Settings(
        node_env="production",
        ai_image_rate_limit_requests=99,
        ai_image_rate_limit_window_seconds=60,
    )

    assert settings.effective_ai_image_rate_limit == (2, 86_400)


# Development và test vẫn cho phép dùng quota cấu hình riêng để thuận tiện kiểm thử.
def test_non_production_uses_configured_image_optimization_quota() -> None:
    settings = Settings(
        node_env="development",
        ai_image_rate_limit_requests=3,
        ai_image_rate_limit_window_seconds=3600,
    )

    assert settings.effective_ai_image_rate_limit == (3, 3600)
