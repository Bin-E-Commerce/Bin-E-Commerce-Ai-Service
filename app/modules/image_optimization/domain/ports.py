"""Protocol cho persistence, Kafka, media va image providers de use case khong bi khoa."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.modules.image_optimization.domain.enums import ImageOptimizationStatus, LifestyleBackgroundPreset
from app.modules.image_optimization.domain.models import GeneratedAsset, ImageOptimizationJob


@dataclass(frozen=True)
class GeneratedImage:
    """Binary tam thoi cua provider, chi ton tai trong worker truoc khi upload media."""

    content: bytes
    content_type: str
    file_name: str


# Chỉ mang dữ liệu cần thiết vào provider ngay trước lời gọi trả phí; không ghi object này vào log, Kafka hay database.
@dataclass(frozen=True)
class LifestyleBackgroundRequest:
    """Ngữ cảnh bối cảnh đã giải mã, chỉ tồn tại trong bộ nhớ của worker."""

    preset: LifestyleBackgroundPreset | None
    description: str | None


class ImageOptimizationJobRepository(Protocol):
    """Port persistence cho aggregate va idempotency lookup."""

    async def save(self, job: ImageOptimizationJob) -> None:
        """Luu aggregate moi hoac ban cap nhat trang thai."""

    async def find_by_id(self, job_id: UUID, seller_owner_id: UUID | None = None) -> ImageOptimizationJob | None:
        """Doc job theo ID va tuy chon chan truy cap cheo seller."""

    async def find_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> ImageOptimizationJob | None:
        """Tra job cu de request retry khong tao job trung."""

    async def find_batch_by_idempotency(self, seller_owner_id: UUID, idempotency_key: str) -> tuple[ImageOptimizationJob, ...]:
        """Tra toan bo job cua batch retry, khong chi job dau tien."""

    async def count_applied(self, seller_owner_id: UUID) -> int:
        """Dem san pham da apply cho metric dashboard."""

    async def count_status(self, seller_owner_id: UUID, status: ImageOptimizationStatus) -> int:
        """Dem job theo state de dashboard khong dung so lieu gia."""


class ImageOptimizationRateLimiter(Protocol):
    """Port sliding-window gioi han chi phi theo seller."""

    async def check(self, key: str, limit: int, window_seconds: int) -> None:
        """Raise loi 429 khi seller vuot quota."""


# Mã hóa mô tả seller trước persistence và chỉ giải mã trong worker ngay trước khi gọi provider.
class BackgroundDescriptionCipher(Protocol):
    """Port bảo vệ mô tả bối cảnh vì worker bất đồng bộ cần đọc lại dữ liệu này."""

    def encrypt(self, value: str) -> str:
        """Mã hóa chuỗi mô tả để database không giữ dữ liệu rõ."""

    def decrypt(self, value: str) -> str:
        """Giải mã chuỗi trong bộ nhớ worker trước khi tạo prompt."""


class OptimizationEventPublisher(Protocol):
    """Port publish event, co the la Kafka hoac adapter outbox local."""

    async def publish_requested(self, job: ImageOptimizationJob) -> None:
        """Day event nho qua broker, khong gui binary image."""


class ImageOptimizationProvider(Protocol):
    """Hop dong provider cho output deterministic va generative."""

    async def generate_white_background(self, source: bytes, file_name: str) -> GeneratedImage:
        """Tao anh nen trang khong dung LLM."""

    async def generate_lifestyle_background(
        self, source: bytes, file_name: str, request: LifestyleBackgroundRequest
    ) -> GeneratedImage:
        """Tao anh lifestyle bang provider vision/image da cau hinh."""


class WhiteBackgroundProviderPort(Protocol):
    """Port nho cho provider local chi xu ly nen trang."""

    async def generate_white_background(self, source: bytes, file_name: str) -> GeneratedImage:
        """Tao output nen trang tu source bytes."""


class LifestyleBackgroundProviderPort(Protocol):
    """Port nho cho provider lifestyle, giup worker khong ep provider local cai method khong dung."""

    async def generate_lifestyle_background(
        self, source: bytes, file_name: str, request: LifestyleBackgroundRequest
    ) -> GeneratedImage:
        """Tao output lifestyle tu source bytes."""


class ProductOwnerClient(Protocol):
    """Port xac minh ownership truoc khi tao job va apply output."""

    async def assert_owned_and_get_updated_at(self, seller_owner_id: UUID, product_id: UUID) -> datetime:
        """Xac minh san pham thuoc seller va tra version hien tai."""

    async def get_cover_asset_id(self, seller_owner_id: UUID, product_id: UUID) -> UUID:
        """Lay asset ID anh dai dien da duoc Product Service xac minh ownership."""

    async def get_product_asset_ids(
        self, seller_owner_id: UUID, product_id: UUID, requested_asset_ids: tuple[UUID, ...]
    ) -> tuple[UUID, ...]:
        """Xac minh cac asset seller chon thuoc product va tra lai danh sach da chuan hoa."""


class ProductMediaClient(Protocol):
    """Port goi Product Service de apply/rollback sau khi seller xac nhan."""

    async def apply_media(
        self,
        *,
        seller_owner_id: UUID,
        product_id: UUID,
        job_id: UUID,
        expected_product_updated_at: datetime | None,
        assets: tuple[GeneratedAsset, ...],
        permissions: tuple[str, ...] = (),
    ) -> None:
        """Apply output qua ownership va optimistic concurrency cua Product Service."""

    async def rollback_media(self, *, seller_owner_id: UUID, product_id: UUID, job_id: UUID) -> None:
        """Khoi phuc snapshot anh goc do Product Service quan ly."""


class MediaAssetClient(Protocol):
    """Port Media Service dung de doc source va upload output, tranh worker truy cap S3 truc tiep."""

    async def download_source(self, *, seller_owner_id: UUID, asset_id: UUID) -> tuple[bytes, str, str]:
        """Tai source bytes qua endpoint noi bo da kiem tra owner/purpose."""

    async def upload_output(
        self,
        *,
        seller_owner_id: UUID,
        job_id: UUID,
        output: GeneratedImage,
    ) -> GeneratedAsset:
        """Upload output va tra asset ID/CDN URL da duoc Media Service cap."""

    async def cleanup_outputs(self, *, seller_owner_id: UUID, job_id: UUID) -> None:
        """Don output tam khi seller tu choi hoac het retention."""
