"""Backward-compatible name for a wholly cash security redemption."""

from dataclasses import dataclass

from .security_action import SecurityAction


@dataclass(frozen=True)
class CashSettlement(SecurityAction):
    pass
