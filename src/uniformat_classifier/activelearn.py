"""Active learning policy.

This module decides *what* to do for each input — but never talks to the user
directly. It returns one of three actions and lets the caller (CLI today,
maybe Streamlit later) handle the actual interaction.

Decision rules (all use the verdict the classifier already computed):
- AUTO        : top-1 score is high AND well separated AND the input is in-distribution
                → accept the prediction without asking
- CONFIRM     : top-1 looks plausible but margin/confidence is borderline
                → show the user the top-3 and ask "is this right?"
- ASK         : low confidence OR out-of-distribution OR margin is tiny
                → show top-3 as suggestions but require the user to pick or type
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .classifier import ClassifierVerdict


class Action(str, Enum):
    AUTO = "auto"
    CONFIRM = "confirm"
    ASK = "ask"


@dataclass
class Policy:
    """Tunable thresholds. Defaults err on the side of asking."""
    auto_min_score: float = 0.78          # need top-1 cosine ≥ this to auto-accept
    auto_min_margin: float = 0.08         # need top-1 to beat top-2 by ≥ this
    confirm_min_score: float = 0.55       # below this we don't even show "confirm" — ASK
    # OOD always escalates; see ClassifierVerdict.out_of_distribution.

    def decide(self, verdict: ClassifierVerdict) -> Action:
        if not verdict.top:
            return Action.ASK
        top = verdict.top[0]
        if verdict.out_of_distribution:
            return Action.ASK
        if top.score >= self.auto_min_score and top.margin >= self.auto_min_margin:
            return Action.AUTO
        if top.score >= self.confirm_min_score:
            return Action.CONFIRM
        return Action.ASK
