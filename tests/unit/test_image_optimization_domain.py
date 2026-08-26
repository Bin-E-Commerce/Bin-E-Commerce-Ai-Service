"""Kiem thu state machine domain image optimization, khong can FastAPI hay database."""

from uuid import uuid4

import pytest

from app.modules.image_optimization.domain.enums import ImageOptimizationMode, ImageOptimizationStatus
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError
from app.modules.image_optimization.domain.models import ImageOptimizationJob


def _job() -> ImageOptimizationJob:
    """Tao aggregate toi thieu de test state transition."""

    return ImageOptimizationJob.create(
        seller_owner_id=uuid4(),
        product_id=uuid4(),
        source_asset_ids=(uuid4(),),
        requested_modes=(ImageOptimizationMode.WHITE_BACKGROUND,),
        idempotency_key="idempotency-domain-test",
        expected_product_updated_at=None,
    )


def test_job_allows_processing_review_and_apply_flow() -> None:
    """AAA: state hop le phai giu dung thu tu xu ly va apply."""

    target = _job()

    processing = target.transition(ImageOptimizationStatus.PROCESSING)
    review = processing.transition(ImageOptimizationStatus.REVIEW_REQUIRED)
    applied = review.transition(ImageOptimizationStatus.APPLIED)

    assert processing.status is ImageOptimizationStatus.PROCESSING
    assert review.status is ImageOptimizationStatus.REVIEW_REQUIRED
    assert applied.status is ImageOptimizationStatus.APPLIED
    assert applied.completed_at is not None


def test_job_rejects_apply_before_review() -> None:
    """AAA: khong cho seller apply khi worker chua tao output review."""

    target = _job()

    with pytest.raises(InvalidJobTransitionError):
        target.transition(ImageOptimizationStatus.APPLIED)
