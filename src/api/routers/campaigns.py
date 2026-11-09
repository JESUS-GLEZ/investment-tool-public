"""Optional, user-owned grouping for existing and future runs."""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter
from sqlalchemy import and_, func, select

from api.campaigns import owned_campaign
from api.models import CampaignRow, RunRow
from api.routers.runs import CurrentUser, DB
from api.schemas import CampaignFields, CampaignSummary

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.get("", response_model=list[CampaignSummary])
def list_campaigns(user: CurrentUser, db: DB) -> list[CampaignSummary]:
    rows = db.execute(
        select(CampaignRow, func.count(RunRow.id))
        .outerjoin(
            RunRow,
            and_(RunRow.campaign_id == CampaignRow.id, RunRow.user_id == user.id),
        )
        .where(CampaignRow.user_id == user.id)
        .group_by(CampaignRow.id)
        .order_by(CampaignRow.updated_at.desc(), CampaignRow.id)
    )
    return [
        CampaignSummary.model_validate(row).model_copy(update={"run_count": count})
        for row, count in rows
    ]


@router.post("", response_model=CampaignSummary, status_code=201)
def create_campaign(body: CampaignFields, user: CurrentUser, db: DB) -> CampaignSummary:
    now = datetime.now(timezone.utc)
    row = CampaignRow(
        id=uuid4().hex[:12],
        user_id=user.id,
        created_at=now,
        updated_at=now,
        **body.model_dump(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return CampaignSummary.model_validate(row)


@router.put("/{campaign_id}", response_model=CampaignSummary)
def update_campaign(
    campaign_id: str, body: CampaignFields, user: CurrentUser, db: DB
) -> CampaignSummary:
    row = owned_campaign(db, campaign_id, user.id)
    row.name, row.hypothesis, row.notes = body.name, body.hypothesis, body.notes
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    count = db.scalar(
        select(func.count(RunRow.id)).where(
            RunRow.campaign_id == row.id, RunRow.user_id == user.id
        )
    )
    return CampaignSummary.model_validate(row).model_copy(update={"run_count": count})
