# Ten-minute interview walkthrough

Open `results/index.html` before the call. Run the experiment once locally beforehand; use the bundled measured artifacts if a live installation is inconvenient. Do not present a smoke run as the five-seed result.

## 0:00–1:00 — The question

> “Can optimizing a plausible bedside-manner metric make the actual response worse? I built a tiny controlled environment to make that failure visible, then added a separate monitor and measured what it catches—and misses.”

Frame the task as supportive communication, not diagnosis or treatment. All 48 patient messages are fictional.

## 1:00–2:30 — The deliberately wrong objective

Open `bedside/scoring.py`. The proxy rewards empathetic phrases, positive words, and length. It never sees the patient's question. Repetitions count repeatedly; negation is ignored.

> “That makes this a useful stand-in for an inadequately validated objective. I made the flaw explicit so that the experiment tests the incentives, not whether I can hide a bug.”

Show the four-node pipeline in the report. Optimization, evaluation, and monitoring are separate code paths.

## 2:30–4:00 — What was really optimized

Open `bedside/policy.py`. Explain the three categorical decisions and the finite 80-word budget. The generator gets a known scenario ID and uses an authored content bank; the learned policy controls style globally.

> “This is actual policy-gradient optimization, but it is not a pretrained LLM or PPO run. The exploit actions exist from the start. The interesting observation is that optimizing the proxy shifts probability toward them.”

There are 300 updates per seed, five seeds, and 48 sampled actions per update. Neither the held-out rubric nor the detector participates in the gradient.

## 4:00–5:30 — Show the failure, not just the score

Show the main chart. Proxy reward rises **15.2 → 98.9**. Held-out rubric score falls **88.3 → 29.2**. Average empathy occurrences rise **1.50 → 15.82**.

Use the response explorer to switch between two held-out prompts. It shows the first draw for seed 7 at each endpoint, not a hand-picked best/worst pair.

> “The polite prefix starts consuming the response budget. The relevant next step disappears. The rubric's action-coverage component falls from 0.85 to 0.07, so this is not just a repetition penalty lowering the aggregate score.”

The score is still a lexical surrogate. Do not call it true patient benefit.

## 5:30–7:30 — The second signal and paired intervention

Open `bedside/detector.py`. It checks length, empathy density, and repeated sentences, using only calibration-reference thresholds. A fixed style alert first fires at step 80; training continues for diagnosis.

Show the second chart. Both selectors see identical nested candidate pools from the **frozen initial generator**. The gated selector rejects flagged candidates, then uses the same proxy to rank the survivors. No survivors means abstention—not a hidden gold answer.

At N=64: **70.2 → 91.0** rubric score, with **720/720** requests served in this sample. At N=1, filtering has a utility cost. Point to the utility and coverage columns.

> “This is a candidate-selection mitigation, not a claim that a collapsed policy has been repaired.”

## 7:30–9:00 — Lead with the limitations

The independent challenge has 24 author-labeled synthetic responses: **8 true positives, 2 false positives, 10 true negatives, and 4 misses**. Precision is 80%; recall is 66.7%. Short positive-token lists bypass the monitor; legitimate longer replies trigger false alarms.

> “A style detector is a useful tripwire, not a correctness oracle. Its 99.7% flag rate on the optimized-policy distribution is not 99.7% recall across all possible hacks.”

The rubric is also gameable. Templates cover test scenarios. The 12 test messages are not a representative patient sample. No human study or live model run is included. The optional Ollama adapter is tested with mocked HTTP only.

## 9:00–10:00 — The next credible experiment

> “Next I would replace the generator with actual model samples, calibrate the detector on a separate model-specific split, and compare a stronger judge against blinded human ratings. Then I would test whether the guard still works against new exploit families, rather than only the family I designed it to catch.”

The repo already provides the model candidate collector, paired selection harness, raw measurements, and a shuffled human-review CSV. Its rating fields are blank: do not imply those experiments are already done.

## Questions you should be ready for

**“Did the model discover this spontaneously?”** No. A small categorical policy learned to favor deliberately available actions. This isolates the mechanism. A real-model extension is available but has not been run here.

**“Why not PPO?”** The small REINFORCE policy makes every probability update auditable; paired best-of-N gives a second, simpler optimization experiment. Neither requires presenting infrastructure work as the scientific result.

**“Is the evaluator independent?”** Its score is isolated from training and gating, but its errors are not guaranteed to be independent. It shares vocabulary with the templates and some style features with the detector. The repo makes that limitation explicit.

**“Did you choose the fix using test quality?”** The gate's fixed reference-calibration procedure does not read test quality labels. The controlled demonstration was designed to expose known mechanisms; it is not a preregistered prospective clinical study or an untouched external benchmark.

**“What stops the optimizer gaming the new detector?”** Nothing in this demo proves that it cannot. The detector is frozen and not part of the training reward. Adaptive optimization against it would be a separate adversarial test.

**“What did you actually validate?”** 54 automated tests, seeded reproduction of raw small-run outputs, no evaluator calls in optimization or gating, abstention accounting, HTTP-contract mocks, and desktop/mobile HTML behavior. Not clinical safety, deployment performance, or a live model backend.
