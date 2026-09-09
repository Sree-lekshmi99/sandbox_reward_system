"""Seeded experiments, raw measurements, paired selection, and audit exports."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
from .dataset import DATA, load_cases, read_jsonl
from .detector import Detector, select
from .policy import Policy, render, train_step
from .quality import judge
from .scoring import features, proxy_reward


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (7, 19, 42, 73, 101)
    steps: int = 300
    batch_size: int = 48
    learning_rate: float = .30
    eval_every: int = 20
    eval_samples: int = 16
    selection_trials: int = 12
    budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("Provide distinct, nonnegative seeds")
        if self.steps < 1 or self.batch_size < 2 or not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Need positive steps/learning rate and batch size >=2")
        if min(self.eval_every, self.eval_samples, self.selection_trials) < 1:
            raise ValueError("Evaluation settings must be positive")
        if not self.budgets or min(self.budgets) < 1 or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("Budgets must be positive, unique, and increasing")


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimate(values: list[float]) -> dict[str, float]:
    """Descriptive seed-cluster bootstrap, not a patient-population interval."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("Cannot estimate an empty or nonfinite sample")
    rng = np.random.default_rng(20260909)
    boot = rng.choice(a, (2000, len(a)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {"mean": float(a.mean()), "low": float(lo), "high": float(hi)}


def audit_detector(detector: Detector) -> dict:
    records = []
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    for record in read_jsonl(DATA / "detector_challenge.jsonl"):
        inspection = detector.inspect(record["response"])
        expected = record["reward_hack"]
        predicted = inspection["flagged"]
        counts[("tp" if expected else "fp") if predicted else ("fn" if expected else "tn")] += 1
        records.append({**record, **inspection})
    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    return {**counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "n": len(records), "records": records,
            "label_provenance": "Author-labeled synthetic challenges, not an independent human study"}


def evaluate_policy(policy: Policy, seed: int, step: int, config: Config,
                    detector: Detector) -> tuple[dict, list[dict], list[dict]]:
    # Common random numbers across checkpoints, independent of training RNG.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 111]))
    rows, examples = [], []
    for case in load_cases("test"):
        for draw in range(config.eval_samples):
            action = policy.sample(rng)
            text = render(case, action, int(rng.integers(5)))
            reward, quality, verdict = proxy_reward(text), judge(case, text), detector.inspect(text)
            row = dict(seed=seed, step=step, case_id=case.id, draw=draw,
                       proxy=reward.displayed, raw_proxy=reward.raw,
                       quality=quality["quality"], concern=quality["concern"],
                       action_score=quality["action"], flagged=int(verdict["flagged"]),
                       words=features(text)["word_count"], empathy_hits=features(text)["empathy_hits"])
            rows.append(row)
            if step in (0, config.steps) and draw == 0:
                examples.append({**row, "message": case.message, "response": text,
                                 "reasons": verdict["reasons"]})
    metrics = dict(seed=seed, step=step)
    for key in ("proxy", "raw_proxy", "quality", "concern", "action_score", "flagged", "words", "empathy_hits"):
        metrics[key] = float(np.mean([r[key] for r in rows]))
    return metrics, rows, examples


def selection_experiment(config: Config, detector: Detector,
                         bank: dict[str, list[str]] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Optimize test-time selection, NOT model weights; nested, paired pools.

    The frozen initial policy (not the collapsed final policy) supplies both
    selectors. Serving coverage and zero-for-abstention utility are measured.
    """
    raw_rows, examples = [], []
    for seed in config.seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 222]))
        policy = Policy()
        for case in load_cases("test"):
            for trial in range(config.selection_trials):
                if bank is None:
                    pool = [render(case, policy.sample(rng), int(rng.integers(5)))
                            for _ in range(max(config.budgets))]
                else:
                    if case.id not in bank or len(bank[case.id]) < max(config.budgets):
                        raise ValueError(f"Bank needs at least {max(config.budgets)} responses for {case.id}")
                    ids = rng.choice(len(bank[case.id]), max(config.budgets), replace=False)
                    pool = [bank[case.id][int(i)] for i in ids]
                for n in config.budgets:
                    naive = select(pool[:n])
                    guarded = select(pool[:n], detector)
                    naive_quality = judge(case, naive)["quality"]
                    guarded_quality = judge(case, guarded)["quality"] if guarded is not None else None
                    row = dict(seed=seed, case_id=case.id, trial=trial, n=n,
                               naive_proxy=proxy_reward(naive).displayed,
                               naive_quality=naive_quality,
                               naive_flagged=int(detector.inspect(naive)["flagged"]),
                               guarded_proxy=proxy_reward(guarded).displayed if guarded is not None else None,
                               guarded_quality=guarded_quality,
                               guarded_utility=guarded_quality if guarded_quality is not None else 0.0,
                               coverage=int(guarded is not None))
                    raw_rows.append(row)
                    if seed == config.seeds[0] and trial == 0 and n == max(config.budgets):
                        examples.append({**row, "message": case.message, "naive_response": naive,
                                         "guarded_response": guarded,
                                         "naive_reasons": detector.inspect(naive)["reasons"]})
    metrics = []
    for seed in config.seeds:
        for n in config.budgets:
            rows = [r for r in raw_rows if r["seed"] == seed and r["n"] == n]
            record = dict(seed=seed, n=n)
            for key in ("naive_proxy", "naive_quality", "naive_flagged", "guarded_proxy", "guarded_quality", "guarded_utility", "coverage"):
                vals = [r[key] for r in rows if r[key] is not None]
                record[key] = float(np.mean(vals)) if vals else None
            metrics.append(record)
    return metrics, raw_rows, examples


def human_review(examples: list[dict], out: Path, seed: int) -> None:
    records = [e for e in examples if e["seed"] == seed]
    np.random.default_rng(333).shuffle(records)
    review, key = [], []
    for index, record in enumerate(records):
        rid = f"r{index+1:03d}"
        review.append(dict(response_id=rid, message=record["message"], response=record["response"],
                           concern_0_4="", action_0_4="", appropriateness_0_4="", clarity_0_4="", notes=""))
        key.append(dict(response_id=rid, case_id=record["case_id"], seed=seed, step=record["step"]))
    save_csv(out / "human_review.csv", review)
    save_csv(out / "human_review_key.csv", key)


def run_experiment(config: Config, out: Path) -> dict:
    config.validate()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite a completed run in {out}. Use a new --out directory.")
    detector = Detector.calibrate(load_cases("calibration"))
    save_json(out / "detector.json", detector.as_dict())
    audit = audit_detector(detector)
    save_json(out / "detector_audit.json", audit)
    metrics, raw, examples, log, states = [], [], [], [], []
    for seed in config.seeds:
        policy = Policy()
        rng = np.random.default_rng(seed)
        train_cases = load_cases("train")
        for step in range(config.steps + 1):
            if step == 0 or step % config.eval_every == 0 or step == config.steps:
                m, r, e = evaluate_policy(policy, seed, step, config, detector)
                metrics.append(m); raw.extend(r); examples.extend(e)
                states.append(dict(seed=seed, step=step, **policy.state()))
            if step < config.steps:
                reward = train_step(policy, train_cases, rng, config.batch_size, config.learning_rate)
                state = policy.state()
                log.append(dict(seed=seed, step=step+1, batch_raw_reward=reward,
                                expected_empathy_count=state["expected_empathy_count"],
                                expected_positive_count=state["expected_positive_count"],
                                specific_probability=state["specific_probability"], entropy=state["entropy"]))
    save_csv(out / "training_metrics.csv", metrics)
    save_csv(out / "training_evaluations.csv", raw)
    save_csv(out / "training_log.csv", log)
    save_json(out / "policy_states.json", states)
    save_jsonl(out / "training_examples.jsonl", examples)
    human_review(examples, out, config.seeds[0])
    selected, selection_raw, selected_examples = selection_experiment(config, detector)
    save_csv(out / "selection_metrics.csv", selected)
    save_csv(out / "selection_evaluations.csv", selection_raw)
    save_jsonl(out / "selection_examples.jsonl", selected_examples)
    initial = [r for r in metrics if r["step"] == 0]
    final = [r for r in metrics if r["step"] == config.steps]
    summary = dict(
        status="executed_offline_compositional_policy",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        config=asdict(config),
        split_counts={s: len(load_cases(s)) for s in ("train", "calibration", "test")},
        data_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir()) if p.is_file()},
        environment={"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")}},
        training={}, selection={}, detector={k: v for k, v in audit.items() if k != "records"},
        caveats=["Controlled compositional generator, not a pretrained LLM or PPO run",
                 "Rubric score is a lexical surrogate, not true quality or clinical validation",
                 "Detector never sees the rubric; both share some observable style cues",
                 "Template content covers all scenarios, including held-out cases",
                 "Selection mitigation uses the frozen initial policy, not the collapsed final policy",
                 "Intervals resample seeds only; the fixed 12-case test set is not a population sample",
                 "Optional Ollama backend is not exercised in this bundled experiment"],
    )
    for name, group in (("initial", initial), ("final", final)):
        summary["training"][name] = {k: estimate([r[k] for r in group]) for k in ("proxy", "quality", "flagged", "words", "empathy_hits")}
    for key in ("proxy", "quality"):
        summary["training"][key + "_delta"] = estimate([b[key] - a[key] for a, b in zip(initial, final)])
    for n in config.budgets:
        group = [r for r in selected if r["n"] == n]
        summary["selection"][str(n)] = {}
        for k in ("naive_proxy", "naive_quality", "guarded_quality", "guarded_utility", "coverage"):
            values = [r[k] for r in group if r[k] is not None]
            summary["selection"][str(n)][k] = estimate(values) if values else None
    # A fixed demonstration alert, not a statistically calibrated sequential test.
    baseline = summary["training"]["initial"]["proxy"]["mean"]
    alert_steps = []
    for step in sorted({r["step"] for r in metrics}):
        rows = [r for r in metrics if r["step"] == step]
        if np.mean([r["proxy"] for r in rows]) >= baseline + 10 and np.mean([r["flagged"] for r in rows]) > .25:
            alert_steps.append(step)
    summary["training"]["first_style_alert_step"] = min(alert_steps) if alert_steps else None
    save_json(out / "summary.json", summary)
    return summary
