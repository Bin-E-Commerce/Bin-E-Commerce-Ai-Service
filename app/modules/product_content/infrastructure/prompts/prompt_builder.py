"""Compatibility exports; prompt source of truth nằm trong application/prompts."""

from app.modules.product_content.application.prompts import (
    DESCRIPTION_PROMPT_VERSION,
    PROMPT_VERSION,
    Prompt,
    build_description_prompt,
    build_prompt,
)

__all__ = ["DESCRIPTION_PROMPT_VERSION", "PROMPT_VERSION", "Prompt", "build_description_prompt", "build_prompt"]
