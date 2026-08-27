"""Kiểm thử boundary background lifestyle, bảo đảm mô tả riêng không thể được dùng cho batch hoặc nền trắng."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.modules.image_optimization.presentation.api.schemas import CreateImageOptimizationRequest


# Tạo payload lifestyle hợp lệ để từng test chỉ thay đổi một rule nghiệp vụ.
def _payload() -> dict[str, object]:
    """Trả payload một sản phẩm với preset và mô tả bối cảnh trong giới hạn cho phép."""

    return {
        "productIds": [str(uuid4())],
        "sourceAssetPolicy": "COVER_IMAGE",
        "modes": ["LIFESTYLE_BACKGROUND"],
        "background": {
            "preset": "WARM_HOME",
            "description": "Bàn gỗ sáng cạnh cửa sổ với ánh nắng buổi sáng dịu nhẹ.",
        },
    }


# Xác nhận request lifestyle hợp lệ giữ nguyên preset và mô tả sau khi Pydantic parse boundary.
def test_accepts_single_product_custom_lifestyle_background() -> None:
    """AAA: seller chọn một sản phẩm thì được phép gửi mô tả bối cảnh tùy chỉnh."""

    payload = _payload()

    target = CreateImageOptimizationRequest.model_validate(payload)

    assert target.background is not None
    assert target.background.preset.value == "WARM_HOME"
    assert target.background.description == "Bàn gỗ sáng cạnh cửa sổ với ánh nắng buổi sáng dịu nhẹ."


# Chặn batch có prompt riêng vì một mô tả không thể đại diện chính xác cho nhiều sản phẩm khác nhau.
def test_rejects_custom_background_for_multiple_products() -> None:
    """AAA: mô tả bối cảnh riêng phải bị từ chối khi seller chọn nhiều sản phẩm."""

    payload = _payload()
    payload["productIds"] = [str(uuid4()), str(uuid4())]

    with pytest.raises(ValidationError, match="exactly one product"):
        CreateImageOptimizationRequest.model_validate(payload)


# Chặn background ở mode nền trắng để pipeline rembg local luôn quyết định, không phát sinh prompt hay chi phí LLM.
def test_rejects_background_for_white_background_mode() -> None:
    """AAA: background chỉ hợp lệ khi request có LIFESTYLE_BACKGROUND."""

    payload = _payload()
    payload["modes"] = ["WHITE_BACKGROUND"]

    with pytest.raises(ValidationError, match="only available for lifestyle"):
        CreateImageOptimizationRequest.model_validate(payload)


# Đồng bộ với frontend: mô tả ngắn bị từ chối, còn để trống vẫn hợp lệ vì trường này không bắt buộc.
def test_rejects_non_empty_background_description_shorter_than_ten_characters() -> None:
    """AAA: nội dung thực sự có ký tự phải đạt tối thiểu 10 ký tự."""

    payload = _payload()
    payload["background"] = {"preset": "WARM_HOME", "description": "phòng đẹp"}

    with pytest.raises(ValidationError, match="at least 10 characters"):
        CreateImageOptimizationRequest.model_validate(payload)


# Chuỗi chỉ có khoảng trắng được xem như không nhập để UI và API không áp dụng hai quy tắc khác nhau.
def test_normalizes_whitespace_only_background_description_to_empty() -> None:
    """AAA: khoảng trắng không tạo prompt tùy chỉnh và được chuẩn hóa thành None."""

    payload = _payload()
    payload["background"] = {"preset": "WARM_HOME", "description": "   "}

    target = CreateImageOptimizationRequest.model_validate(payload)

    assert target.background is not None
    assert target.background.description is None
