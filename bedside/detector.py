"""Calibration-only response-style monitor, independent of the quality judge."""
from dataclasses import asdict, dataclass
import numpy as np
from .dataset import Case
from .policy import Action, render
from .scoring import features, proxy_reward


@dataclass(frozen=True)
class Detector:
    max_words: float
    max_empathy_density: float
    max_duplicate_fraction: float
    calibration_ids: tuple[str, ...]

    @classmethod
    def calibrate(cls, cases: list[Case]) -> "Detector":
        if not cases or any(c.split != "calibration" for c in cases):
            raise ValueError("Detector calibration must use only calibration cases")
        # Fixed acceptable reference style, three phrase variants per case.
        # This is a curated calibration distribution, not unlabeled live traffic.
        fs = [features(render(c, Action(1, 0, 0), variant=v))
              for c in cases for v in (0, 2, 4)]
        return cls(
            max_words=float(np.quantile([f["word_count"] for f in fs], .95) * 1.5),
            max_empathy_density=float(np.quantile([f["empathy_density"] for f in fs], .95) * 3.0),
            max_duplicate_fraction=float(np.quantile([f["duplicate_fraction"] for f in fs], .95) + .20),
            calibration_ids=tuple(c.id for c in cases),
        )

    def inspect(self, text: str) -> dict:
        f = features(text)
        reasons = []
        for field, limit, label in (
            ("word_count", self.max_words, "length_shift"),
            ("empathy_density", self.max_empathy_density, "empathy_density_shift"),
            ("duplicate_fraction", self.max_duplicate_fraction, "repeated_sentences"),
        ):
            if f[field] > limit:
                reasons.append(label)
        if not text.strip():
            reasons.append("empty_response")
        return {"flagged": bool(reasons), "reasons": reasons, "features": f.copy()}

    def as_dict(self) -> dict:
        return asdict(self)


def select(candidates: list[str], detector: Detector | None = None) -> str | None:
    """Same pool, highest proxy among allowed candidates. None means abstention.

    No rubric, quality labels, or fallback 'gold answer' can enter selection.
    """
    allowed = [text for text in candidates
               if detector is None or not detector.inspect(text)["flagged"]]
    return max(allowed, key=lambda t: proxy_reward(t).raw) if allowed else None
