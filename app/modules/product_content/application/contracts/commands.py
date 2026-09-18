"""File này định nghĩa các command cho module product_content,
giúp gom input đã validate từ presentation layer
và chuẩn hóa dữ liệu trước khi gửi sang application layer."""

import hashlib
import json
from dataclasses import dataclass

from app.modules.product_content.domain.models import ProductContext, ProductImage

# Đồng bộ version với prompt để thay đổi công thức tên không trả lại kết quả cache của prompt cũ.
CACHE_KEY_VERSION = "product-name-v2"
DESCRIPTION_CACHE_KEY_VERSION = "product-description-v1"


# Giữ asset ID để tạo fingerprint cache nhưng loại nó khỏi context gửi sang LLM.
@dataclass(frozen=True)
class ImageCommand:
    """Thông tin ảnh dùng cho cache key và URL CDN an toàn cho provider."""

    asset_id: str
    public_url: str
    file_name: str


# Command gom input đã validate để use case xử lý mà không phải biết JSON alias của HTTP.
@dataclass(frozen=True)
class NameSuggestionCommand:
    """Input use case đã chuẩn hóa; asset ID không đi vào provider context."""

    category_name: str
    category_path: str | None
    brand: str | None
    draft_name: str | None
    short_description: str | None
    description: str | None
    attributes: tuple[tuple[str, str], ...]
    images: tuple[ImageCommand, ...]
    locale: str

    # Hash dữ liệu ổn định giúp cache không lưu prompt/raw payload và không lộ nội dung qua key.
    # Asset ID và các trường seller được sắp xếp/serialize ổn định để cùng một input luôn có cùng fingerprint.
    # Chỉ fingerprint được đưa vào cache; asset ID không bao giờ được chuyển tiếp vào prompt của provider.
    def cache_key(self) -> str:
        """Tạo fingerprint ổn định thay vì lưu payload hoặc prompt nhạy cảm."""

        payload = {
            "prompt_version": CACHE_KEY_VERSION,
            "category_name": self.category_name,
            "category_path": self.category_path,
            "brand": self.brand,
            "draft_name": self.draft_name,
            "short_description": self.short_description,
            "description": self.description,
            "attributes": self.attributes,
            "images": tuple((image.asset_id, image.file_name) for image in self.images),
            "locale": self.locale,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # Chuyển command thành context tối thiểu cho LLM, loại user ID và asset ID khỏi prompt.
    # Application layer vẫn giữ asset ID ở command để tạo cache key, nhưng provider chỉ nhận URL đã allow-list.
    # Việc tách hai object này bảo vệ ranh giới dữ liệu ngay cả khi một provider mới được thêm vào sau này.
    def to_provider_context(self) -> ProductContext:
        """Chuyển command thành context không chứa asset ID hoặc user ID."""

        return ProductContext(
            category_name=self.category_name,
            category_path=self.category_path,
            brand=self.brand,
            draft_name=self.draft_name,
            short_description=self.short_description,
            description=self.description,
            attributes=self.attributes,
            images=tuple(ProductImage(public_url=image.public_url, file_name=image.file_name) for image in self.images),
            locale=self.locale,
        )


# Command cho use case mô tả dùng cùng context an toàn nhưng có version cache riêng với tên sản phẩm.
@dataclass(frozen=True)
class DescriptionSuggestionCommand:
    """Input đã chuẩn hóa để sinh một bản mô tả hoàn chỉnh, không chứa user ID trong provider context."""

    category_name: str
    category_path: str | None
    brand: str | None
    draft_name: str | None
    description: str | None
    attributes: tuple[tuple[str, str], ...]
    images: tuple[ImageCommand, ...]
    locale: str

    # Fingerprint có version riêng để thay đổi prompt mô tả không dùng nhầm kết quả tên cũ.
    def cache_key(self) -> str:
        """Tạo hash input ổn định; không lưu payload thô hoặc prompt vào cache."""

        payload = {
            "prompt_version": DESCRIPTION_CACHE_KEY_VERSION,
            "category_name": self.category_name,
            "category_path": self.category_path,
            "brand": self.brand,
            "draft_name": self.draft_name,
            "description": self.description,
            "attributes": self.attributes,
            "images": tuple((image.asset_id, image.file_name) for image in self.images),
            "locale": self.locale,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # Chỉ chuyển public URL và facts đã chuẩn hóa sang provider; asset ID vẫn chỉ phục vụ fingerprint.
    def to_provider_context(self) -> ProductContext:
        """Chuyển command thành ProductContext không có định danh nội bộ."""

        return ProductContext(
            category_name=self.category_name,
            category_path=self.category_path,
            brand=self.brand,
            draft_name=self.draft_name,
            short_description=None,
            description=self.description,
            attributes=self.attributes,
            images=tuple(ProductImage(public_url=image.public_url, file_name=image.file_name) for image in self.images),
            locale=self.locale,
        )
