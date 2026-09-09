# Measured run summary

Executed with a controlled compositional generator, **not a pretrained LLM**.

| Measurement | Initial | After 300 updates |
|---|---:|---:|
| Proxy reward (fixed display scale) | 15.22 | 98.87 |
| Held-out lexical rubric score | 88.27 | 29.17 |
| Style-monitor flag rate | 8.13% | 99.69% |

A fixed style alert first fired at checkpoint **80**.
This is a retrospective diagnostic; it did not stop the experimental training run.

## Paired best-of-64 mitigation experiment

The candidate source is the **frozen initial generator**, not the trained final policy.
Proxy-only selection scored **70.22** on the rubric.
Detector-gated selection scored **90.98** on served replies,
with **100.00%** coverage and **90.98**
zero-for-abstention utility per request. The gate does not use rubric scores.

## Separate detector challenge

8 TP / 2 FP / 10 TN / 4 FN on 24 author-labeled synthetic examples.
Precision: 80.0%; recall: 66.7%.
These are descriptive counts on a tiny designed challenge set, not deployment estimates.

## Provenance and limits

Configuration, input SHA-256 hashes, Python/dependency versions, and seed-bootstrap intervals
are in `summary.json`. Raw checkpoint scores are in `training_evaluations.csv`.
Full sampled before/after texts are in `training_examples.jsonl`.
Both selectors' raw outcomes are in `selection_evaluations.csv`.
There are 5 independent optimization seeds; intervals resample seeds only.
The 12 fixed test cases are not an independent clinical population sample.

The rubric is a second lexical surrogate, not true quality. It shares some style cues with the
monitor and vocabulary with the templates. Templates cover all scenarios. The generator's
finite action space explicitly contains repetition and positive padding; the optimization,
not the action-space design, is learned. No real-model inference or human judging is claimed.
