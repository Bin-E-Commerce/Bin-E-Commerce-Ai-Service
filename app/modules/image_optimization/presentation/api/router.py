"""Router image optimization: validate request, go application va map response."""

import re
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from app.bootstrap.dependencies import (
    get_image_apply_user,
    get_image_generate_user,
    get_image_optimization_service,
    get_image_rollback_user,
    get_image_user,
)
from app.core.security import UserContext
from app.modules.image_optimization.application.commands import CreateOptimizationJobsCommand
from app.modules.image_optimization.application.service import ImageOptimizationApplicationService
from app.modules.image_optimization.domain.errors import InvalidJobTransitionError
from app.modules.image_optimization.presentation.api.schemas import (
    ApplyImageOptimizationRequest,
    CreateImageOptimizationRequest,
    CreateImageOptimizationResponse,
    ImageOptimizationOverviewResponse,
    OptimizationJobResponse,
)

router = APIRouter(prefix="/api/v1/seller/ai/image-optimization", tags=["seller-image-optimization"])


# Chuyển aggregate domain sang response ổn định, đồng thời trả version sản phẩm để frontend apply đúng optimistic lock.
def _to_job_response(job: object) -> OptimizationJobResponse:
    """Chuyển aggregate domain sang schema JSON, không lộ raw persistence fields."""

    from app.modules.image_optimization.domain.models import ImageOptimizationJob

    if not isinstance(job, ImageOptimizationJob):
        raise TypeError("Unsupported job type")
    return OptimizationJobResponse(
        job_id=job.job_id,
        product_id=job.product_id,
        status=job.status,
        processing_stage=job.processing_stage,
        generation_profile=job.generation_profile,
        background_preset=job.background_preset,
        generated_asset_ids=list(job.generated_asset_ids),
        generated_assets=[
            {
                "assetId": str(asset.asset_id),
                "imageUrl": asset.public_url,
                "mode": asset.mode,
                "outputId": str(asset.output_id),
                "sourceAssetId": str(asset.source_asset_id) if asset.source_asset_id else None,
            }
            for asset in job.generated_assets
        ],
        created_at=job.created_at,
        expected_product_updated_at=job.expected_product_updated_at,
        failure_code=job.failure_code,
    )


@router.post(
    "/jobs", response_model=CreateImageOptimizationResponse, response_model_by_alias=True, status_code=status.HTTP_202_ACCEPTED
)
async def create_image_optimization_jobs(
    payload: CreateImageOptimizationRequest,
    user: Annotated[UserContext, Depends(get_image_generate_user)],
    service: Annotated[ImageOptimizationApplicationService, Depends(get_image_optimization_service)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> CreateImageOptimizationResponse:
    """Tao batch job idempotent va tra ve ngay de worker xu ly bat dong bo.

    Route khong tai anh, khong goi OpenAI va khong tu kiem tra ownership; cac boundary
    do application/Product Service xu ly. Header idempotency ngan retry cua browser tao
    them job va tieu chi phi provider.
    """

    if not idempotency_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Idempotency-Key is required")
    if len(idempotency_key) > 160 or re.fullmatch(r"[A-Za-z0-9._:-]+", idempotency_key) is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Idempotency-Key has an invalid format")
    command = CreateOptimizationJobsCommand(
        seller_owner_id=UUID(user.user_id),
        product_ids=tuple(payload.product_ids),
        source_asset_policy=payload.source_asset_policy.value,
        modes=tuple(payload.modes),
        source_asset_ids=tuple(payload.source_asset_ids or ()),
        idempotency_key=idempotency_key,
        expected_product_updated_at=None,
        background_preset=payload.background.preset if payload.background else None,
        background_description=payload.background.description if payload.background else None,
        force_regenerate=payload.force_regenerate,
        permissions=user.permissions,
        seller_email=user.email,
    )
    batch_id, jobs = await service.create_jobs(command)
    return CreateImageOptimizationResponse(batch_id=batch_id, jobs=[_to_job_response(job) for job in jobs])


@router.get("/jobs/{job_id}", response_model=OptimizationJobResponse, response_model_by_alias=True)
async def get_image_optimization_job(
    job_id: UUID,
    user: Annotated[UserContext, Depends(get_image_user)],
    service: Annotated[ImageOptimizationApplicationService, Depends(get_image_optimization_service)],
) -> OptimizationJobResponse:
    """Doc job cua seller hien tai, khong cho phep truy cap cheo seller."""

    job = await service.get_job(job_id, UUID(user.user_id))
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Optimization job not found")
    return _to_job_response(job)


@router.get("/overview", response_model=ImageOptimizationOverviewResponse, response_model_by_alias=True)
async def get_image_optimization_overview(
    user: Annotated[UserContext, Depends(get_image_user)],
    service: Annotated[ImageOptimizationApplicationService, Depends(get_image_optimization_service)],
) -> ImageOptimizationOverviewResponse:
    """Tra metric tong quan tu application service cho dashboard Seller Center."""

    overview = await service.get_overview(UUID(user.user_id))
    return ImageOptimizationOverviewResponse.model_validate(overview)


@router.post("/jobs/{job_id}/reject", response_model=OptimizationJobResponse, response_model_by_alias=True)
async def reject_image_optimization_job(
    job_id: UUID,
    user: Annotated[UserContext, Depends(get_image_apply_user)],
    service: Annotated[ImageOptimizationApplicationService, Depends(get_image_optimization_service)],
) -> OptimizationJobResponse:
    """Danh dau reject va khong cham vao anh goc trong Media Service."""

    try:
        job = await service.reject_job(job_id, UUID(user.user_id))
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Optimization job not found") from error
    except InvalidJobTransitionError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Optimization job is not ready for rejection") from error
    return _to_job_response(job)


@router.post("/jobs/{job_id}/apply", response_model=OptimizationJobResponse, response_model_by_alias=True)
# Nhận xác nhận apply từ seller và giữ nguyên version server đã chụp lúc tạo job.
async def apply_image_optimization_job(
    job_id: UUID,
    payload: ApplyImageOptimizationRequest,
    user: Annotated[UserContext, Depends(get_image_apply_user)],
    service: Annotated[ImageOptimizationApplicationService, Depends(get_image_optimization_service)],
    response: Response,
) -> OptimizationJobResponse:
    """Map selection sang asset IDs; use case tự kiểm tra owner, version và lifecycle."""

    updated = await service.apply_job(
        job_id,
        UUID(user.user_id),
        expected_product_updated_at=payload.expected_product_updated_at,
        selected_asset_ids=tuple(image.asset_id for image in payload.images),
        permissions=user.permissions,
        seller_email=user.email,
    )
    if updated.status.value == "FINALIZING":
        response.status_code = status.HTTP_202_ACCEPTED
    return _to_job_response(updated)


@router.post("/jobs/{job_id}/rollback", response_model=OptimizationJobResponse, response_model_by_alias=True)
async def rollback_image_optimization_job(
    job_id: UUID,
    user: Annotated[UserContext, Depends(get_image_rollback_user)],
    service: Annotated[ImageOptimizationApplicationService, Depends(get_image_optimization_service)],
) -> OptimizationJobResponse:
    """Danh dau rollback sau khi Product Service khoi phuc snapshot anh goc."""

    try:
        job = await service.rollback_job(job_id, UUID(user.user_id), user.permissions, user.email)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Optimization job not found") from error
    except InvalidJobTransitionError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Optimization job cannot be rolled back") from error
    return _to_job_response(job)
