"""Loi nghiep vu rieng cua quy trinh toi uu anh san pham."""


class ImageOptimizationDomainError(ValueError):
    """Loi domain khong duoc gan voi HTTP hay provider cu the."""


class InvalidJobTransitionError(ImageOptimizationDomainError):
    """Tu choi chuyen trang thai khong nam trong state machine."""


class ProductOwnershipError(ImageOptimizationDomainError):
    """Chan job neu san pham khong thuoc seller dang yeu cau."""
