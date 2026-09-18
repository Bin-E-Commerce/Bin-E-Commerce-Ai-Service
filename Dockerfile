# Image của AI Service: cài dependency trong một stage riêng, sau đó chỉ copy
# virtual environment và các file cần thiết khi chạy vào image cuối cùng.
#
# Image này được dùng chung cho HTTP API và các background worker. Compose hoặc
# Kubernetes chỉ thay đổi command của từng process, còn dependency và các lớp
# bảo mật vẫn được giữ đồng nhất giữa mọi runtime của AI Service.

FROM python:3.12-slim AS builder

# Giữ hành vi Python ổn định trong container và ngăn pip tạo cache. Cache pip
# không giúp process lúc runtime nhưng sẽ làm tăng kích thước các layer của image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Compiler chỉ cần trong trường hợp dependency Python không có wheel tương thích
# với Python 3.12/Linux. Toolchain này chỉ tồn tại ở builder và không đi vào image
# runtime, nhờ đó image chạy cuối cùng nhỏ hơn và có bề mặt tấn công thấp hơn.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes build-essential \
    && python -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*

# Copy metadata của dependency trước source code để Docker tái sử dụng layer cài
# đặt tốn thời gian khi chỉ có code ứng dụng thay đổi.
COPY pyproject.toml ./

# Chỉ cài dependency runtime được khai báo trong pyproject.toml. Nhóm tùy chọn
# `dev` được loại khỏi image: pytest, mypy, Ruff và import-linter chỉ dành cho
# local/CI, không cần thiết trong image chạy production.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

FROM python:3.12-slim AS runtime

# Dùng cùng các biến môi trường với builder để lệnh `python` và các console
# script đã cài đặt luôn trỏ vào virtual environment được copy sang image này.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

# LightGBM và ONNX Runtime dùng OpenMP khi chạy. Chỉ cài thư viện runtime chung
# (`libgomp1`) vào image cuối; toàn bộ compiler toolchain vẫn nằm ở builder và
# không bị đưa lên môi trường production.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Không chạy FastAPI hoặc worker bằng root. Điều này giới hạn mức độ ảnh hưởng
# nếu dependency có lỗ hổng, đồng thời đặt owner rõ ràng cho thư mục model/cache
# được mount vào container.
RUN addgroup --system aiworker \
    && adduser --system --ingroup aiworker --home /home/aiworker aiworker \
    && mkdir -p /home/aiworker/.u2net /app/artifacts \
    && chown -R aiworker:aiworker /home/aiworker /app

# Copy virtual environment đã build nhưng không copy build tool, compiler hoặc
# pip cache. Image runtime vẫn có đủ thư viện cho FastAPI, LightGBM và pipeline
# xử lý ảnh rembg/ONNX.
COPY --from=builder /opt/venv /opt/venv

# Giữ migration và cấu hình Alembic trong image để migration job dùng đúng artifact
# với API. Các file này không được import trong quá trình API khởi động.
COPY --chown=aiworker:aiworker app ./app
COPY --chown=aiworker:aiworker migrations ./migrations
COPY --chown=aiworker:aiworker alembic.ini ./alembic.ini

USER aiworker

EXPOSE 3009

# Probe này chỉ kiểm tra process đã lắng nghe HTTP, không gọi OpenAI, Kafka,
# PostgreSQL hoặc Redis. Việc kiểm tra dependency thuộc về application; liveness
# probe giá rẻ không nên tạo thêm request tốn chi phí hoặc phụ thuộc bên ngoài.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3009/api/health', timeout=3)"

# Khởi động bằng exec form giúp SIGTERM đi thẳng tới Uvicorn khi rollout hoặc
# shutdown. Nhờ vậy FastAPI có thể đóng HTTP client, Redis connection và pool
# database một cách sạch sẽ.
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["app.entrypoints.api:app", "--host", "0.0.0.0", "--port", "3009"]
