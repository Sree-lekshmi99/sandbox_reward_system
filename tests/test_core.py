import inspect
import json
import numpy as np
import pytest
from bedside.dataset import DATA, load_cases, read_jsonl
from bedside.detector import Detector, select
from bedside.experiment import Config, audit_detector
from bedside.policy import Action, Policy, render, train_step
from bedside.quality import judge
from bedside.scoring import features, proxy_reward


def test_dataset_is_synthetic_unique_and_group_disjoint():
    cases = load_cases()
    assert len(cases) == 48
    assert len({c.id for c in cases}) == len({c.message for c in cases}) == 48
    assert all(c.synthetic for c in cases)
    assert [len(load_cases(s)) for s in ("train", "calibration", "test")] == [24, 12, 12]
    groups = [{c.scenario for c in load_cases(s)} for s in ("train", "calibration", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_templates_and_rubrics_cover_all_cases():
    scenarios = {c.scenario for c in load_cases()}
    for file in ("generator_templates.json", "rubric.json"):
        assert scenarios == set(json.loads((DATA / file).read_text()))


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        load_cases("validation_typo")


def test_proxy_rewards_repeated_empathy():
    assert proxy_reward("I understand. " * 10).raw > proxy_reward("I understand.").raw


def test_proxy_deliberately_ignores_negation():
    assert features("This is not good.")["positive_hits"] == 1


def test_fixed_display_scale_and_empty_reward():
    assert proxy_reward("").displayed == 0
    assert proxy_reward("I understand. " * 100).displayed == 100
    assert proxy_reward("I understand.").displayed == 2 * proxy_reward("I understand.").raw


def test_rubric_prefers_concern_specific_content_to_spam():
    c = load_cases("test")[0]
    good = render(c, Action(1, 0, 0))
    spam = render(c, Action(5, 3, 1))
    assert judge(c, good)["quality"] > judge(c, spam)["quality"] + 50
    assert proxy_reward(spam).raw > proxy_reward(good).raw
    assert judge(c, "")["quality"] == 0


def test_rubric_has_documented_keyword_overlap_blind_spot():
    c = load_cases("test")[0]
    # Semantically poor text receives high lexical credit: not 'true quality'.
    assert judge(c, "diagnosis overwhelmed ask team")["quality"] >= 90


def test_rubric_penalizes_narrow_guarantee_pattern():
    c = load_cases("test")[0]
    base = render(c, Action(1, 0, 0))
    assert judge(c, base + " You are guaranteed to be fine.")["appropriate"] == 0


@pytest.mark.parametrize("empathy", range(6))
def test_generator_respects_budget(empathy):
    for c in load_cases():
        text = render(c, Action(empathy, 3, 0))
        assert len(text.split()) <= 80
        assert text.endswith(".")


def test_policy_probabilities_normalize():
    for p in Policy().probabilities():
        assert np.isclose(p.sum(), 1)
        assert np.all(p > 0)


def test_policy_gradient_increases_rewarded_action_probability():
    p = Policy()
    before = p.probabilities()[0][5]
    p.update([Action(0, 0, 0), Action(5, 0, 0)], np.array([0., 50.]), .3)
    assert p.probabilities()[0][5] > before


def test_training_is_deterministic():
    p, q = Policy(), Policy()
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(12):
        assert train_step(p, load_cases("train"), a) == train_step(q, load_cases("train"), b)
    assert p.state() == q.state()


@pytest.mark.parametrize("split", ["calibration", "test"])
def test_training_rejects_nontraining_cases(split):
    with pytest.raises(ValueError, match="training split"):
        train_step(Policy(), load_cases(split), np.random.default_rng(7))


@pytest.mark.parametrize("split", ["train", "test"])
def test_detector_rejects_noncalibration_cases(split):
    with pytest.raises(ValueError, match="calibration cases"):
        Detector.calibrate(load_cases(split))


def test_no_evaluation_dependency_in_optimizer_or_detector():
    import bedside.policy as policy
    import bedside.detector as detector
    for module in (policy, detector):
        source = inspect.getsource(module)
        assert "from .quality" not in source
        assert "judge(" not in source


def test_training_and_selection_do_not_call_judge(monkeypatch):
    import bedside.quality as quality
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation leaked into training or selection")
    monkeypatch.setattr(quality, "judge", forbidden)
    train_step(Policy(), load_cases("train"), np.random.default_rng(7))
    gate = Detector.calibrate(load_cases("calibration"))
    assert select(["Ask the clinic a question."], gate) is not None


def test_guard_abstains_when_every_candidate_is_flagged():
    gate = Detector.calibrate(load_cases("calibration"))
    assert select(["I understand. " * 20], gate) is None
    assert select([], gate) is None
    assert gate.inspect("")["flagged"]


def test_guard_uses_same_pool_and_proxy_not_gold_fallback():
    gate = Detector.calibrate(load_cases("calibration"))
    good = render(load_cases("test")[0], Action(1, 0, 0))
    spam = "I understand. " * 20
    assert select([good, spam]) == spam
    assert select([good, spam], gate) == good


def test_detector_known_miss_remains_visible():
    gate = Detector.calibrate(load_cases("calibration"))
    assert not gate.inspect("Good great wonderful excellent positive hopeful.")["flagged"]


def test_audit_reports_all_denominators():
    audit = audit_detector(Detector.calibrate(load_cases("calibration")))
    assert audit["n"] == 24 == sum(audit[k] for k in ("tp", "tn", "fp", "fn"))
    assert audit["fn"] > 0 and audit["fp"] > 0
    assert audit["precision"] == audit["tp"] / (audit["tp"] + audit["fp"])


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"batch_size": 1}, {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"seeds": ()}, {"seeds": (7, 7)}, {"seeds": (-1,)}, {"budgets": (2, 1)},
    {"budgets": (0,)}, {"eval_samples": 0}, {"selection_trials": 0},
])
def test_invalid_configs_fail_early(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs).validate()


def test_jsonl_reports_line_numbers(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{}\nnot-json\n')
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(path)
