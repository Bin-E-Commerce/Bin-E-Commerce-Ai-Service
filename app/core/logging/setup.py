"""Thiết lập logging tối thiểu và tránh ghi prompt, ảnh hoặc secret vào log."""

import logging


# Thiết lập format chung một lần khi app bắt đầu, không log dữ liệu sản phẩm nhạy cảm.
def configure_logging() -> None:
    """Thiết lập logging một lần và không đưa prompt, URL ảnh hoặc secret vào log."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
