"""Authenticated CRUD for reusable run configurations."""

from typing import Annotated, Any, Self

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import get_db
from api.models import PresetRow, UserRow
from api.schemas import (
    ExecutionRequest,
    PredictionComparisonRequest,
    RunType,
    SolverComparisonRequest,
    StrategyComparisonRequest,
)

router = APIRouter(prefix="/presets", tags=["presets"])
CurrentUser = Annotated[UserRow, Depends(get_current_user)]
Database = Annotated[Session, Depends(get_db)]
REQUEST_TYPES = {
    RunType.EXECUTION: ExecutionRequest,
    RunType.STRATEGY_COMPARISON: StrategyComparisonRequest,
    RunType.SOLVER_COMPARISON: SolverComparisonRequest,
    RunType.PREDICTION_COMPARISON: PredictionComparisonRequest,
}


class PresetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    run_type: RunType
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Preset name cannot be blank")
        schema = REQUEST_TYPES[self.run_type]
        unknown = self.parameters.keys() - schema.model_fields.keys()
        if unknown:
            raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
        self.parameters = schema.model_validate(self.parameters).model_dump(mode="json")
        return self


class PresetResponse(BaseModel):
    """Return stored snapshots without applying today's input-only validators.

    Legacy configurations must stay visible so owners can correct or delete
    them; new runs and preset writes still validate through PresetRequest.
    """

    id: int
    name: str
    run_type: RunType
    parameters: dict[str, Any]
    model_config = {"from_attributes": True}


def owned_preset(db: Session, user: UserRow, preset_id: int) -> PresetRow:
    row = db.query(PresetRow).filter_by(id=preset_id, user_id=user.id).first()
    if row is None:
        raise HTTPException(404, "Preset not found")
    return row


def save(db: Session, row: PresetRow) -> PresetRow:
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            409, "A preset with this name already exists for this run type"
        )
    db.refresh(row)
    return row


@router.get("", response_model=list[PresetResponse])
def list_presets(run_type: RunType, user: CurrentUser, db: Database) -> list[PresetRow]:
    rows: list[PresetRow] = (
        db.query(PresetRow)
        .filter_by(user_id=user.id, run_type=run_type.value)
        .order_by(PresetRow.name)
        .all()
    )
    return rows


@router.post("", response_model=PresetResponse, status_code=201)
def create_preset(body: PresetRequest, user: CurrentUser, db: Database) -> PresetRow:
    return save(db, PresetRow(user_id=user.id, **body.model_dump(mode="json")))


@router.put("/{preset_id}", response_model=PresetResponse)
def update_preset(
    preset_id: int, body: PresetRequest, user: CurrentUser, db: Database
) -> PresetRow:
    row = owned_preset(db, user, preset_id)
    if row.run_type != body.run_type.value:
        raise HTTPException(422, "Cannot change a preset's run type")
    row.name = body.name
    row.parameters = body.parameters
    return save(db, row)


@router.delete("/{preset_id}", status_code=204)
def delete_preset(preset_id: int, user: CurrentUser, db: Database) -> Response:
    db.delete(owned_preset(db, user, preset_id))
    db.commit()
    return Response(status_code=204)
