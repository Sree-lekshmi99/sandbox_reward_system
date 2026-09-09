"""Charts and a self-contained HTML walkthrough, generated from measured outputs."""
from pathlib import Path
import base64
import csv
import html
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dataset import read_jsonl
from .experiment import estimate


def read_metrics(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: float(v) if v else None for k, v in row.items()} for row in csv.DictReader(f)]


def bands(rows: list[dict], xkey: str, ykey: str) -> tuple:
    xs = sorted({r[xkey] for r in rows})
    stats = [estimate([r[ykey] for r in rows if r[xkey] == x and r[ykey] is not None])
             if any(r[xkey] == x and r[ykey] is not None for r in rows) else None for x in xs]
    return xs, *[[s[k] if s else np.nan for s in stats] for k in ("mean", "low", "high")]


def plot_training(out: Path, summary: dict) -> Path:
    rows = read_metrics(out / "training_metrics.csv")
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    for key, label, linestyle in (("proxy", "Optimized proxy reward", "-"),
                                  ("quality", "Held-out rubric score", "--")):
        x, mean, low, high = bands(rows, "step", key)
        ax.plot(x, mean, label=label, linewidth=2.7, linestyle=linestyle, marker="o", markersize=3)
        ax.fill_between(x, low, high, alpha=.14)
    alert = summary["training"]["first_style_alert_step"]
    if alert is not None:
        ax.axvline(alert, linestyle=":", linewidth=1.3, alpha=.7)
        ax.annotate(f"Style alert at step {alert}", xy=(alert, 101), xytext=(8, -4),
                    textcoords="offset points", fontsize=10, va="top")
    ax.set(xlabel="Actual policy-gradient updates", ylabel="Score (two fixed 0–100 scales)",
           ylim=(0, 106), xlim=(0, summary["config"]["steps"]))
    ax.grid(alpha=.18)
    ax.legend(loc="center right", frameon=False, fontsize=11)
    fig.suptitle("The reward improves. The response stops helping.", fontsize=20, y=.97)
    ax.set_title(f"Controlled compositional policy  |  {len(summary['config']['seeds'])} seeds  |  12 held-out synthetic messages", fontsize=11, pad=16)
    fig.text(.5, .025, "Rubric score is not clinical truth. Shading: descriptive 95% seed-bootstrap intervals; no patient-population inference.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(.02, .06, .99, .94))
    path = out / "reward_hacking.png"
    fig.savefig(path, dpi=170)
    fig.savefig(out / "reward_hacking.svg")
    plt.close(fig)
    return path


def plot_selection(out: Path, rows: list[dict] | None = None,
                   source: str = "Frozen initial generator; same nested candidate pools for both selectors") -> Path:
    rows = rows if rows is not None else read_metrics(out / "selection_metrics.csv")
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    for key, label, linestyle in (
        ("naive_quality", "Proxy-only: rubric score", "-"),
        ("guarded_quality", "Detector-gated: score on served replies", "--"),
        ("guarded_utility", "Detector-gated: utility per request*", ":"),
    ):
        x, mean, low, high = bands(rows, "n", key)
        ax.plot(x, mean, label=label, linewidth=2.5, linestyle=linestyle, marker="o", markersize=4)
        ax.fill_between(x, low, high, alpha=.1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(int(v)) for v in x])
    ax.set(xlabel="Candidates per prompt N (inference budget, not training steps)",
           ylabel="Held-out lexical rubric score", ylim=(0, 105))
    ax.grid(alpha=.18)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    fig.suptitle("A second signal changes which answer wins.", fontsize=20, y=.97)
    ax.set_title(source, fontsize=11, pad=16)
    fig.text(.5, .025, "*Abstentions receive zero utility. The detector never reads the rubric. Bands resample seeds, not patients.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(.02, .06, .99, .94))
    path = out / "mitigation.png"
    fig.savefig(path, dpi=170)
    fig.savefig(out / "mitigation.svg")
    plt.close(fig)
    return path


def build_report(out: Path) -> None:
    summary = json.loads((out / "summary.json").read_text())
    plot_training(out, summary)
    plot_selection(out)
    training = read_jsonl(out / "training_examples.jsonl")
    selection = read_jsonl(out / "selection_examples.jsonl")
    first_seed = summary["config"]["seeds"][0]
    cases = []
    for before in [r for r in training if r["seed"] == first_seed and r["step"] == 0]:
        after = next(r for r in training if r["seed"] == first_seed and r["step"] == summary["config"]["steps"] and r["case_id"] == before["case_id"])
        picked = next(r for r in selection if r["case_id"] == before["case_id"])
        cases.append({"id": before["case_id"], "message": before["message"], "before": before, "after": after, "selection": picked})
    n = str(max(summary["config"]["budgets"]))
    initial, final, selected = summary["training"]["initial"], summary["training"]["final"], summary["selection"][n]
    def value(group: dict, key: str) -> float:
        return group[key]["mean"]
    audit = summary["detector"]
    report = (Path(__file__).with_name("report_template.html")).read_text()
    replacements = {
        "__STEPS__": str(summary["config"]["steps"]), "__SEEDS__": str(len(summary["config"]["seeds"])),
        "__FIRST_SEED__": str(first_seed), "__N__": n,
        "__PROXY_BEFORE__": f"{value(initial,'proxy'):.1f}", "__PROXY_AFTER__": f"{value(final,'proxy'):.1f}",
        "__QUALITY_BEFORE__": f"{value(initial,'quality'):.1f}", "__QUALITY_AFTER__": f"{value(final,'quality'):.1f}",
        "__GUARDED__": f"{value(selected,'guarded_quality'):.1f}" if selected["guarded_quality"] else "No replies served",
        "__NAIVE__": f"{value(selected,'naive_quality'):.1f}",
        "__COVERAGE__": f"{100*value(selected,'coverage'):.1f}%",
        "__UTILITY__": f"{value(selected,'guarded_utility'):.1f}",
        "__FLAGGED__": f"{100*value(final,'flagged'):.1f}%",
        "__ALERT__": str(summary["training"]["first_style_alert_step"]),
        "__PRECISION__": f"{100*audit['precision']:.1f}%" if audit['precision'] is not None else "Undefined",
        "__RECALL__": f"{100*audit['recall']:.1f}%" if audit['recall'] is not None else "Undefined",
        "__CONFUSION__": f"{audit['tp']} true positives · {audit['fp']} false positives · {audit['tn']} true negatives · {audit['fn']} misses",
        "__RUN_DATE__": html.escape(summary["generated_at_utc"]),
        "__DATA__": json.dumps(cases).replace("<", "\\u003c"),
    }
    for name in ("reward_hacking", "mitigation"):
        replacements[f"__{name.upper()}_IMG__"] = base64.b64encode((out / f"{name}.png").read_bytes()).decode()
    for key, replacement in replacements.items():
        report = report.replace(key, replacement)
    (out / "index.html").write_text(report, encoding="utf-8")
    guarded_text = f"{value(selected, 'guarded_quality'):.2f}" if selected["guarded_quality"] else "undefined (no replies served)"
    text = f"""# Measured run summary

Executed with a controlled compositional generator, **not a pretrained LLM**.

| Measurement | Initial | After {summary['config']['steps']} updates |
|---|---:|---:|
| Proxy reward (fixed display scale) | {value(initial,'proxy'):.2f} | {value(final,'proxy'):.2f} |
| Held-out lexical rubric score | {value(initial,'quality'):.2f} | {value(final,'quality'):.2f} |
| Style-monitor flag rate | {100*value(initial,'flagged'):.2f}% | {100*value(final,'flagged'):.2f}% |

A fixed style alert first fired at checkpoint **{summary['training']['first_style_alert_step']}**.
This is a retrospective diagnostic; it did not stop the experimental training run.

## Paired best-of-{n} mitigation experiment

The candidate source is the **frozen initial generator**, not the trained final policy.
Proxy-only selection scored **{value(selected,'naive_quality'):.2f}** on the rubric.
Detector-gated selection scored **{guarded_text}** on served replies,
with **{100*value(selected,'coverage'):.2f}%** coverage and **{value(selected,'guarded_utility'):.2f}**
zero-for-abstention utility per request. The gate does not use rubric scores.

## Separate detector challenge

{audit['tp']} TP / {audit['fp']} FP / {audit['tn']} TN / {audit['fn']} FN on {audit['n']} author-labeled synthetic examples.
Precision: {100*audit['precision']:.1f}%; recall: {100*audit['recall']:.1f}%.
These are descriptive counts on a tiny designed challenge set, not deployment estimates.

## Provenance and limits

Configuration, input SHA-256 hashes, Python/dependency versions, and seed-bootstrap intervals
are in `summary.json`. Raw checkpoint scores are in `training_evaluations.csv`.
Full sampled before/after texts are in `training_examples.jsonl`.
Both selectors' raw outcomes are in `selection_evaluations.csv`.
There are {len(summary['config']['seeds'])} independent optimization seeds; intervals resample seeds only.
The 12 fixed test cases are not an independent clinical population sample.

The rubric is a second lexical surrogate, not true quality. It shares some style cues with the
monitor and vocabulary with the templates. Templates cover all scenarios. The generator's
finite action space explicitly contains repetition and positive padding; the optimization,
not the action-space design, is learned. No real-model inference or human judging is claimed.
"""
    (out / "summary.md").write_text(text)
