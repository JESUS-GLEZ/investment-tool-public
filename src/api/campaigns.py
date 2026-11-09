"""Campaign ownership checks shared by run-submission routes."""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.db_store import DBRunStore
from api.models import CampaignRow
from api.store import RunStore


def owned_campaign(db: Session, campaign_id: str, user_id: int) -> CampaignRow:
    row = db.query(CampaignRow).filter_by(id=campaign_id, user_id=user_id).first()
    if row is None:
        raise HTTPException(404, "Campaign not found")
    return row


def validate_campaign(
    store: DBRunStore | RunStore,
    db: Session,
    campaign_id: str | None,
    user_id: int,
) -> None:
    if campaign_id is None:
        return
    if not isinstance(store, DBRunStore):
        raise HTTPException(501, "Campaigns require database-backed run storage")
    owned_campaign(db, campaign_id, user_id)
