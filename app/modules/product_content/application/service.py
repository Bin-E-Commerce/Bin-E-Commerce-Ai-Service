"""Compatibility aliases; code mới dùng trực tiếp các class trong application/use_cases."""

from app.modules.product_content.application.use_cases import GenerateProductDescription, GenerateProductNames

ProductNameSuggestionService = GenerateProductNames
ProductDescriptionSuggestionService = GenerateProductDescription

__all__ = ["ProductDescriptionSuggestionService", "ProductNameSuggestionService"]
