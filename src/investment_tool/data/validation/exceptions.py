"""Custom exceptions for the validation module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from investment_tool.exceptions import ValidationError as BaseValidationError

if TYPE_CHECKING:
    from .validation_report import ValidationReport


class ValidationError(BaseValidationError):
    """Raised when validation fails at a critical level.

    Inherits from the main ValidationError in investment_tool.exceptions
    to maintain a consistent exception hierarchy.

    Attributes
    ----------
    report : ValidationReport | None
        The validation report containing all issues found.
    message : str
        Human-readable error message.
    """

    def __init__(
        self,
        message: str,
        report: "ValidationReport | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, report=report, **kwargs)

    def __str__(self) -> str:
        """Return message with summary if report is available."""
        if self.report:
            return f"{self.message}\n{str(self.report.summary())}"
        return str(self.message)
