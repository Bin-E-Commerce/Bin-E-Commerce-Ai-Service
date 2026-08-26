"""Mã hóa mô tả bối cảnh lifestyle trước khi lưu job và không chịu trách nhiệm tạo prompt hay gọi OpenAI."""

from cryptography.fernet import Fernet, InvalidToken

from app.core.errors import ConfigurationError, ProviderUnavailableError


# Adapter Fernet dùng khóa từ cấu hình runtime để database chỉ nhận ciphertext có thể giải mã bởi worker cùng môi trường.
class FernetBackgroundDescriptionCipher:
    """Bảo vệ mô tả seller khi workflow bất đồng bộ bắt buộc phải lưu lại dữ liệu giữa API và worker."""

    # Khởi tạo cipher từ secret đã inject; không nhận khóa từ HTTP request hay log lại giá trị này.
    def __init__(self, secret: str | None) -> None:
        if not secret:
            raise ConfigurationError()
        self._fernet = Fernet(secret.encode("utf-8"))

    # Mã hóa trước persistence để raw mô tả không xuất hiện trong PostgreSQL hoặc event broker.
    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    # Chỉ giải mã ngay trong worker trước khi provider xây prompt; ciphertext lỗi được map thành lỗi provider an toàn.
    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode("utf-8")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError) as error:
            raise ProviderUnavailableError() from error
