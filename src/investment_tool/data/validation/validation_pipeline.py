"""Validation pipeline that orchestrates multiple validators."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Sequence

from .validation_report import ValidationReport, ValidationIssue, IssueSeverity
from .validators import (
    BaseValidator,
    RequiredColumnsValidator,
    MissingDataValidator,
    TimeSeriesConsistencyValidator,
    NumericRangeValidator,
    MinimumHistoryValidator,
    NewsDataValidator,
)
from .exceptions import ValidationError
from .bar_quality import BarQualityValidator

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Controls how validation errors are handled."""

    STRICT = "strict"  # Raise exception on any error
    WARN = "warn"  # Log warnings, continue on errors
    SKIP = "skip"  # Skip validation entirely


class ValidationPipeline:
    """Orchestrates multiple validators in sequence.

    The pipeline runs validators in order, passing the (possibly modified) data
    from one validator to the next. Each validator adds issues to a shared report.

    Attributes
    ----------
    validators : list[BaseValidator]
        Ordered list of validators to run.
    level : ValidationLevel
        How to handle validation errors.
    """

    def __init__(
        self,
        validators: Sequence[BaseValidator] | None = None,
        level: ValidationLevel = ValidationLevel.WARN,
        *,
        point_in_time_history: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        validators : Sequence[BaseValidator] | None
            Custom list of validators. If None, uses default validators.
        level : ValidationLevel
            How to handle validation errors.
        """
        self.validators = (
            list(validators)
            if validators is not None
            else self._default_validators(point_in_time_history)
        )
        self.level = level

    @staticmethod
    def _default_validators(point_in_time_history: bool = False) -> list[BaseValidator]:
        """Create the default validator chain."""
        if point_in_time_history:
            # Missing observations and young listings are assessed at each
            # decision. Future missingness or series length cannot drop them
            # globally, and missing execution quotes must not be filled.
            return [
                RequiredColumnsValidator(drop_invalid=True),
                TimeSeriesConsistencyValidator(
                    check_future_dates=True,
                    drop_duplicates=True,
                ),
                MissingDataValidator(fill_method="none", drop_if_exceeds=False),
                BarQualityValidator(),
                NumericRangeValidator(clip_outliers=False),
                NewsDataValidator(check_future_dates=True),
            ]
        return [
            RequiredColumnsValidator(drop_invalid=True),
            TimeSeriesConsistencyValidator(
                check_future_dates=True,
                drop_duplicates=True,
            ),
            MissingDataValidator(
                max_missing_pct=0.3,
                max_consecutive_missing=5,
                fill_method="ffill",
                drop_if_exceeds=True,
            ),
            NumericRangeValidator(clip_outliers=False),
            MinimumHistoryValidator(min_periods=30, drop_insufficient=True),
            NewsDataValidator(check_future_dates=True),
        ]

    def validate(self, data: dict[str, Any]) -> tuple[dict[str, Any], ValidationReport]:
        """Run all validators on the input data.

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary with keys: stock_dict, crypto_dict, news_dict.

        Returns
        -------
        tuple[dict[str, Any], ValidationReport]
            Validated data and the validation report.

        Raises
        ------
        ValidationError
            If level is STRICT and critical issues are found.
        """
        if self.level == ValidationLevel.SKIP:
            logger.info("Validation skipped (level=SKIP)")
            return data, ValidationReport()

        report = ValidationReport()

        # Count original symbols
        stock_count = len(data.get("stock_dict", {}))
        crypto_count = len(data.get("crypto_dict", {}))
        report.original_symbol_count = stock_count + crypto_count

        logger.info(
            f"Starting validation pipeline with {len(self.validators)} validators"
        )
        logger.info(f"Input: {stock_count} stocks, {crypto_count} crypto symbols")

        # Run validators in sequence
        validated_data = data
        for validator in self.validators:
            logger.debug(f"Running validator: {validator.name}")
            try:
                validated_data = validator.validate(validated_data, report)
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")
                report.add_issue(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        validator=validator.name,
                        message=f"Validator failed with exception: {e}",
                        action="block_backtest",
                    )
                )
                if self.level == ValidationLevel.STRICT:
                    raise ValidationError(
                        f"Validation failed in {validator.name}: {e}",
                        report=report,
                    ) from e

        # Count final symbols
        final_stock_count = len(validated_data.get("stock_dict", {}))
        final_crypto_count = len(validated_data.get("crypto_dict", {}))
        report.final_symbol_count = final_stock_count + final_crypto_count

        logger.info(
            f"Validation complete. "
            f"Symbols: {report.original_symbol_count} -> {report.final_symbol_count}"
        )

        # Handle based on level
        if report.has_critical_issues():
            logger.error("Critical validation issues found!")
            logger.error(report.summary())
            if self.level == ValidationLevel.STRICT:
                raise ValidationError(
                    "Critical validation issues found",
                    report=report,
                )
        elif report.has_errors():
            logger.warning("Validation errors found (non-critical)")
            if self.level == ValidationLevel.STRICT:
                raise ValidationError(
                    "Validation errors found",
                    report=report,
                )

        # Log summary at INFO level
        if logger.isEnabledFor(logging.INFO):
            logger.info(report.summary())

        return validated_data, report

    def add_validator(
        self, validator: BaseValidator, position: int | None = None
    ) -> None:
        """Add a validator to the pipeline.

        Parameters
        ----------
        validator : BaseValidator
            Validator to add.
        position : int | None
            Position to insert at. If None, appends to end.
        """
        if position is None:
            self.validators.append(validator)
        else:
            self.validators.insert(position, validator)

    def remove_validator(self, validator_name: str) -> bool:
        """Remove a validator by name.

        Parameters
        ----------
        validator_name : str
            Name of the validator to remove.

        Returns
        -------
        bool
            True if validator was found and removed.
        """
        for i, v in enumerate(self.validators):
            if v.name == validator_name:
                self.validators.pop(i)
                return True
        return False
