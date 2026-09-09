import csv
import json
import re
from pathlib import Path
import pytest
from bedside.dataset import load_cases, read_jsonl
from bedside.detector import Detector
from bedside.experiment import Config, estimate, run_experiment, selection_experiment
from bedside.report import build_report


def test_seed_interval_handles_single_seed():
    assert estimate([3.0]) == {"mean": 3.0, "low": 3.0, "high": 3.0}


def test_paired_nested_selection_and_abstention_accounting():
    gate = Detector.calibrate(load_cases("calibration"))
    config = Config(seeds=(7,), steps=2, selection_trials=2, budgets=(1, 2, 4))
    metrics, raw, _ = selection_experiment(config, gate)
    assert len(metrics) == 3
    for case in load_cases("test"):
        for trial in range(2):
            group = [r for r in raw if r["case_id"] == case.id and r["trial"] == trial]
            assert [r["naive_proxy"] for r in group] == sorted(r["naive_proxy"] for r in group)
            assert [r["coverage"] for r in group] == sorted(r["coverage"] for r in group)
            for row in group:
                assert row["guarded_utility"] == (row["guarded_quality"] if row["coverage"] else 0)


def test_all_abstentions_are_not_hidden():
    strict = Detector(0, 0, 0, ())
    _, raw, _ = selection_experiment(Config(seeds=(7,), selection_trials=1, budgets=(1, 2)), strict)
    assert all(r["guarded_quality"] is None for r in raw)
    assert all(r["guarded_utility"] == 0 and r["coverage"] == 0 for r in raw)


def test_candidate_bank_requires_all_test_cases():
    with pytest.raises(ValueError, match="Bank needs"):
        selection_experiment(Config(seeds=(7,), selection_trials=1, budgets=(1,)),
                             Detector.calibrate(load_cases("calibration")), bank={})


def test_end_to_end_reproduces_raw_results_and_builds_report(tmp_path):
    config = Config(seeds=(7, 19), steps=3, batch_size=8, eval_every=2,
                    eval_samples=2, selection_trials=1, budgets=(1, 2, 4))
    a, b = tmp_path / "a", tmp_path / "b"
    first, second = run_experiment(config, a), run_experiment(config, b)
    for file in ("training_metrics.csv", "training_evaluations.csv", "training_log.csv",
                 "selection_evaluations.csv", "training_examples.jsonl", "human_review.csv"):
        assert (a / file).read_bytes() == (b / file).read_bytes()
    assert first["training"] == second["training"]
    assert first["selection"] == second["selection"]
    build_report(a)
    text = (a / "index.html").read_text()
    assert not re.search(r"__[A-Z_]+__", text)
    assert "data:image/png;base64," in text
    assert "not a pretrained LLM" in text
    assert (a / "reward_hacking.png").stat().st_size > 1000
    reviews = list(csv.DictReader((a / "human_review.csv").open()))
    assert len(reviews) == 24
    assert all(not r["concern_0_4"] for r in reviews)
    with pytest.raises(FileExistsError):
        run_experiment(config, a)
