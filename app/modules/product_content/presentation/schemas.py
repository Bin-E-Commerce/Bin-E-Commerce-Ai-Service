"""File này định nghĩa schema request/response cho module product_content.
Các schema này chuẩn hóa dữ liệu từ frontend trước khi gửi sang application layer.
Các schema này được thiết kế để:
- Giới hạn độ dài text để tránh phình prompt và chi phí token.
- Trim text ở boundary để application layer nhận dữ liệu ổn định và không phải lặp xử lý khoảng trắng.
- Sử dụng alias camelCase để tương thích với contract từ frontend.
"""

# Thư viện typing để định nghĩa kiểu dữ liệu,
# bao gồm Annotated để thêm metadata cho các trường,
# Literal để giới hạn giá trị của một trường.
from typing import Annotated, Literal

# BaseModel dùng để định nghĩa các schema request/response,
# ConfigDict để cấu hình các schema,
# Field để định nghĩa các trường với ràng buộc và alias,
# HttpUrl để validate các trường URL,
# field_validator để định nghĩa các hàm validate cho các trường.
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# Trim text ở boundary để application layer nhận dữ liệu ổn định và không phải lặp xử lý khoảng trắng.
def _trim(value: str | None) -> str | None:
    """Chuẩn hóa text ở boundary trước khi đi vào application layer."""

    return value.strip() if value is not None else None


#  Schema ngành hàng đảm bảo tên không rỗng và giới hạn độ dài để tránh phình prompt và chi phí token.
class CategoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Annotated[str, Field(min_length=1, max_length=120)]
    path: Annotated[str | None, Field(default=None, max_length=240)] = None

    _trim_name = field_validator("name", mode="before")(_trim)
    _trim_path = field_validator("path", mode="before")(_trim)


# Schema thuộc tính giới hạn độ dài để không phình prompt và chi phí token.
class AttributeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    label: Annotated[str, Field(min_length=1, max_length=80)]
    value: Annotated[str, Field(min_length=1, max_length=240)]

    _trim_label = field_validator("label", mode="before")(_trim)
    _trim_value = field_validator("value", mode="before")(_trim)


# Gom các text seller nhập, dùng alias camelCase tương thích contract từ frontend.
class SellerInputRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    draft_name: Annotated[str | None, Field(default=None, alias="draftName", max_length=200)] = None
    short_description: Annotated[str | None, Field(default=None, alias="shortDescription", max_length=600)] = None
    description: Annotated[str | None, Field(default=None, max_length=3000)] = None
    attributes: Annotated[list[AttributeRequest], Field(default_factory=list, max_length=20)]

    _trim_draft_name = field_validator("draft_name", mode="before")(_trim)
    _trim_short_description = field_validator("short_description", mode="before")(_trim)
    _trim_description = field_validator("description", mode="before")(_trim)


# Schema ảnh bắt buộc asset ID để cache và public URL để gửi vision sau khi kiểm tra CDN.
class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    asset_id: Annotated[str, Field(min_length=1, max_length=128, alias="assetId")]
    public_url: Annotated[HttpUrl, Field(alias="publicUrl")]
    file_name: Annotated[str, Field(min_length=1, max_length=180, alias="fileName")]

    _trim_asset_id = field_validator("asset_id", mode="before")(_trim)
    _trim_file_name = field_validator("file_name", mode="before")(_trim)


# Boundary request giới hạn tối đa ba ảnh trước khi request chạm application/provider.
class NameSuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    category: CategoryRequest
    brand: Annotated[str | None, Field(default=None, max_length=120)] = None
    seller_input: Annotated[SellerInputRequest | None, Field(default=None, alias="sellerInput")] = None
    images: Annotated[list[ImageRequest], Field(min_length=1, max_length=3)]
    locale: Literal["vi-VN"] = "vi-VN"

    _trim_brand = field_validator("brand", mode="before")(_trim)


# Request dùng chung context với gợi ý tên nhưng tách contract để frontend/backend dễ tiến hóa độc lập.
class DescriptionSuggestionRequest(NameSuggestionRequest):
    """Payload tạo một mô tả sản phẩm hoàn chỉnh bằng tiếng Việt."""


# Schema một đề xuất để frontend hiển thị title, lý do và cờ đề xuất tốt nhất.
class SuggestionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    reason: str
    recommended: bool


# Warning public chỉ chứa mã/trường/thông báo an toàn, không trả giá trị nhạy cảm gốc.
class WarningResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code: str
    field: str
    message: str


# Response ổn định cho frontend với requestId dùng truy vết mà không lộ prompt.
class NameSuggestionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    suggestions: list[SuggestionResponse]
    warnings: list[WarningResponse]
    request_id: str = Field(alias="requestId")


# Response chỉ trả nội dung đã safety-validate và warning công khai, không trả prompt/provider detail.
class DescriptionSuggestionResponse(BaseModel):
    """Mô tả đề xuất và warning để seller xem trước trước khi áp dụng vào form."""

    description: str
    warnings: list[WarningResponse]
    request_id: str = Field(alias="requestId")
