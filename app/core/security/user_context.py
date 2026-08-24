"""Đọc user context từ Gateway và kiểm tra permission trước khi gọi LLM trả phí."""

from dataclasses import dataclass

from app.core.errors import AuthorizationError


# Context bất biến giúp các lớp bên trong chỉ dùng identity đã được Gateway xác thực.
@dataclass(frozen=True)
class UserContext:
    """Identity và permission đã được chuẩn hóa từ header nội bộ."""

    user_id: str
    permissions: frozenset[str]


# Kiểm tra user trước, sau đó mới tách permission để request thiếu quyền không thể tiêu quota LLM.
# Header chỉ được xem là context nội bộ do Gateway đã xác thực; service vẫn kiểm tra lại để tránh tin mù quáng.
# Kết quả là context bất biến, giúp các lớp nghiệp vụ không phải tự parse chuỗi quyền hoặc đọc header lần nữa.
def build_user_context(
    user_id: str | None,
    permission_header: str | None,
    required_permission: str,
) -> UserContext:
    """Kiểm tra user context và permission trước khi gọi LLM trả phí."""

    if not user_id or len(user_id) > 128:
        raise AuthorizationError()

    permissions = frozenset(permission.strip() for permission in (permission_header or "").split(",") if permission.strip())
    if required_permission not in permissions:
        raise AuthorizationError()

    return UserContext(user_id=user_id, permissions=permissions)
