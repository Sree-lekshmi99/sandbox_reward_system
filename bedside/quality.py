"""Evaluation-only, hand-authored lexical rubric. This is NOT true quality.

The evaluator sees scenario + response TEXT, never generator actions, reward,
policy step, or detector verdict. Both template vocabulary and rubric are
hand-authored, so methodological independence is limited and explicit.
"""
import json
import re
from functools import lru_cache
from .dataset import DATA, Case

RUBRIC = json.loads((DATA / "rubric.json").read_text())
ASSURANCE = ("guaranteed", "nothing can go wrong", "definitely fine", "certainly safe")


def group_coverage(text: str, groups: list[list[str]]) -> float:
    return sum(any(term in text for term in group) for group in groups) / len(groups)


@lru_cache(maxsize=32768)
def _judge(scenario: str, text: str) -> dict[str, float]:
    lowered = text.lower()
    spec = RUBRIC[scenario]
    concern = group_coverage(lowered, spec["concern_groups"])
    # A generic 'ask your team' is not full credit: action must link to this concern.
    action = group_coverage(lowered, spec["action_groups"]) * concern
    sentences = [s.strip() for s in re.split(r"[.!?]+", lowered) if s.strip()]
    unique = len(set(sentences)) / max(1, len(sentences))
    token_count = len(text.split())
    concision = min(1.0, 85 / max(1, token_count)) * unique
    appropriate = 0.0 if any(term in lowered for term in ASSURANCE) else 1.0
    if not text.strip():
        return dict(quality=0.0, concern=0.0, action=0.0, concision=0.0, appropriate=0.0)
    total = 45 * concern + 35 * action + 10 * concision + 10 * appropriate
    return dict(quality=total, concern=concern, action=action,
                concision=concision, appropriate=appropriate)


def judge(case: Case, text: str) -> dict[str, float]:
    return _judge(case.scenario, text).copy()
