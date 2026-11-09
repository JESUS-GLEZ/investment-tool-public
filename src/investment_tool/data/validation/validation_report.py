"""Validation report data structures."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any


class IssueSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Data was modified or has minor issues
    ERROR = "error"  # Critical issue that may affect results
    CRITICAL = "critical"  # Data is unusable, must be dropped


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the data.

    Attributes
    ----------
    severity : IssueSeverity
        How severe the issue is.
    validator : str
        Name of the validator that found the issue.
    symbol : str | None
        Symbol affected (if applicable).
    field_name : str | None
        Field/column affected (if applicable).
    message : str
        Human-readable description of the issue.
    details : dict[str, Any]
        Additional details (e.g., affected rows, values).
    """

    severity: IssueSeverity
    validator: str
    message: str
    symbol: str | None = None
    field_name: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)
    action: str = "review"

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}]", f"({self.validator})"]
        if self.symbol:
            parts.append(f"[{self.symbol}]")
        if self.field_name:
            parts.append(f"[{self.field_name}]")
        parts.append(self.message)
        return " ".join(parts)


@dataclass
class ValidationReport:
    """Aggregates all validation issues and tracks data modifications.

    Attributes
    ----------
    issues : list[ValidationIssue]
        All issues found during validation.
    dropped_symbols : dict[str, str]
        Symbols that were dropped and why (symbol -> reason).
    modified_symbols : dict[str, list[str]]
        Symbols that were modified and how (symbol -> list of modifications).
    original_symbol_count : int
        Number of symbols before validation.
    final_symbol_count : int
        Number of symbols after validation.
    """

    issues: list[ValidationIssue] = dataclass_field(default_factory=list)
    dropped_symbols: dict[str, str] = dataclass_field(default_factory=dict)
    modified_symbols: dict[str, list[str]] = dataclass_field(default_factory=dict)
    original_symbol_count: int = 0
    final_symbol_count: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)

    def drop_symbol(self, symbol: str, reason: str) -> None:
        """Record that a symbol was dropped."""
        self.dropped_symbols[symbol] = reason
        self.add_issue(
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                validator="ValidationPipeline",
                symbol=symbol,
                message=f"Symbol dropped: {reason}",
            )
        )

    def record_modification(self, symbol: str, modification: str) -> None:
        """Record that a symbol's data was modified."""
        if symbol not in self.modified_symbols:
            self.modified_symbols[symbol] = []
        self.modified_symbols[symbol].append(modification)

    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    def has_errors(self) -> bool:
        """Check if there are any errors or critical issues."""
        return any(
            i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)
            for i in self.issues
        )

    def get_issues_by_severity(self, severity: IssueSeverity) -> list[ValidationIssue]:
        """Get all issues of a specific severity."""
        return [i for i in self.issues if i.severity == severity]

    def get_issues_by_symbol(self, symbol: str) -> list[ValidationIssue]:
        """Get all issues for a specific symbol."""
        return [i for i in self.issues if i.symbol == symbol]

    def summary(self) -> str:
        """Generate a human-readable summary of the validation."""
        lines = ["=" * 60, "VALIDATION REPORT", "=" * 60]

        # Counts by severity
        for severity in IssueSeverity:
            count = len(self.get_issues_by_severity(severity))
            if count > 0:
                lines.append(f"  {severity.value.upper()}: {count}")

        # Symbol statistics
        lines.append(
            f"\nSymbols: {self.original_symbol_count} -> {self.final_symbol_count}"
        )
        if self.dropped_symbols:
            lines.append(f"  Dropped: {len(self.dropped_symbols)}")
            for symbol, reason in list(self.dropped_symbols.items())[:5]:
                lines.append(f"    - {symbol}: {reason}")
            if len(self.dropped_symbols) > 5:
                lines.append(f"    ... and {len(self.dropped_symbols) - 5} more")

        if self.modified_symbols:
            lines.append(f"  Modified: {len(self.modified_symbols)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a dictionary for serialization."""
        return {
            "issues": [
                {
                    "severity": i.severity.value,
                    "validator": i.validator,
                    "symbol": i.symbol,
                    "field": i.field_name,
                    "message": i.message,
                    "details": i.details,
                    "action": i.action,
                }
                for i in self.issues
            ],
            "dropped_symbols": self.dropped_symbols,
            "modified_symbols": self.modified_symbols,
            "original_symbol_count": self.original_symbol_count,
            "final_symbol_count": self.final_symbol_count,
        }
