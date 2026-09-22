"""Thiết lập structured logging cho AI Service.

File này sở hữu format/redaction của log stdout; không sở hữu business logging,
không ghi prompt, media URL, token hoặc dữ liệu nhận diện đầy đủ.
"""

import json
import logging
import re


# Redact token/password và che email trước khi log đi vào stdout/Loki.
def _sanitize(value: object) -> str:
    text = str(value)
    text = re.sub(
        r"(authorization|cookie|password|token|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+)",
        r"\1***\2",
        text,
    )


# Formatter giữ schema log ổn định để Loki tìm theo request_id mà không cần regex theo từng service.
class JsonFormatter(logging.Formatter):
    # Chuyển LogRecord thành JSON line với field bounded và không serialize exception payload tùy ý.
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "service": "ai-service",
            "logger": record.name,
            "message": _sanitize(record.getMessage()),
        }
        for key in ("request_id", "method", "route", "status_code", "duration_ms", "event"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = _sanitize(value)
        if record.exc_info:
            payload["exception"] = _sanitize(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


# Thiết lập logging một lần với JSON line và policy không log dữ liệu nhạy cảm.
def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
