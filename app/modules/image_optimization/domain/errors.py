"""Loi nghiep vu rieng cua quy trinh toi uu anh san pham."""


class ImageOptimizationDomainError(ValueError):
    """Loi domain khong duoc gan voi HTTP hay provider cu the."""


class InvalidJobTransitionError(ImageOptimizationDomainError):
    """Tu choi chuyen trang thai khong nam trong state machine."""


class ProductOwnershipError(ImageOptimizationDomainError):
    """Chan job neu san pham khong thuoc seller dang yeu cau."""


# Từ chối asset output không thuộc job hoặc bị gửi trùng trong cùng request apply.
class InvalidOutputSelectionError(ImageOptimizationDomainError):
    """Bảo vệ Product Service khỏi danh sách output do browser tự sửa."""
