"""Typed, packaged synthetic data; strict split and schema validation."""
from dataclasses import dataclass
from pathlib import Path
import json

DATA = Path(__file__).with_name("data")

@dataclass(frozen=True)
class Case:
    id: str
    split: str
    scenario: str
    message: str
    synthetic: bool = True


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                records.append(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return records


def load_cases(split: str | None = None) -> list[Case]:
    cases = [Case(**row) for row in read_jsonl(DATA / "patients.jsonl")]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    if len({c.message for c in cases}) != len(cases):
        raise ValueError("Duplicate messages")
    scenarios: dict[str, str] = {}
    for case in cases:
        if not case.synthetic or case.split not in {"train", "calibration", "test"}:
            raise ValueError(f"Invalid case: {case.id}")
        if scenarios.setdefault(case.scenario, case.split) != case.split:
            raise ValueError("A scenario crosses split boundaries")
    if split is not None and split not in {"train", "calibration", "test"}:
        raise ValueError(f"Unknown split: {split}")
    return [c for c in cases if split is None or c.split == split]
