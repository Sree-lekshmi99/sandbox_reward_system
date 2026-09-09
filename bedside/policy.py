"""Compositional generator + actual factorized categorical policy gradients.

This is a controlled action-space experiment, NOT a pretrained language model,
PPO implementation, or evidence of an LLM discovering a novel exploit.
"""
from dataclasses import dataclass
import json
import numpy as np
from .dataset import DATA, Case
from .scoring import EMPATHY, proxy_reward

TEMPLATES = json.loads((DATA / "generator_templates.json").read_text())
EMPATHY_COUNTS = np.array([0, 1, 2, 4, 8, 16])
POSITIVE_COUNTS = np.array([0, 1, 2, 4])
# Initial support includes the exploit; optimization must discover its high reward.
INITIAL = (
    np.array([0.06, 0.70, 0.15, 0.06, 0.02, 0.01]),
    np.array([0.82, 0.13, 0.04, 0.01]),
    np.array([0.85, 0.15]),  # specific / generic content
)
WORD_BUDGET = 80


@dataclass(frozen=True)
class Action:
    empathy: int
    positivity: int
    content: int

    def indices(self) -> tuple[int, int, int]:
        return self.empathy, self.positivity, self.content


def render(case: Case, action: Action, variant: int = 0, budget: int = WORD_BUDGET) -> str:
    if budget < 1:
        raise ValueError("Word budget must be positive")
    parts = [EMPATHY[(variant + j) % len(EMPATHY)]
             for j in range(int(EMPATHY_COUNTS[action.empathy]))]
    parts += ["Good, positive, hopeful, supported, and calm."] * int(POSITIVE_COUNTS[action.positivity])
    if action.content == 0:
        template = TEMPLATES[case.scenario]
        parts += [template["reflection"], template["next_step"]]
    else:
        parts += ["A good and positive outlook can feel hopeful."]
    # Fixed sentence ordering + finite context is the resource competition.
    # Use complete sentences only, not a fabricated score penalty for an action ID.
    result: list[str] = []
    used = 0
    for sentence in parts:
        size = len(sentence.split())
        if used + size > budget:
            break
        result.append(sentence)
        used += size
    return " ".join(result)


class Policy:
    def __init__(self) -> None:
        self.logits = [np.log(p.copy()) for p in INITIAL]

    def probabilities(self) -> list[np.ndarray]:
        result = []
        for logits in self.logits:
            exps = np.exp(logits - logits.max())
            result.append(exps / exps.sum())
        return result

    def sample(self, rng: np.random.Generator) -> Action:
        return Action(*(int(rng.choice(len(p), p=p)) for p in self.probabilities()))

    def update(self, actions: list[Action], rewards: np.ndarray, lr: float) -> None:
        """REINFORCE with a leave-one-out baseline; no quality/detector input.

        Divide by a FIXED reward scale (25), not a learned evaluator score.
        Leave-one-out avoids the small bias from a same-batch mean baseline.
        """
        if len(actions) < 2 or len(actions) != len(rewards) or lr <= 0:
            raise ValueError("Need >=2 aligned actions/rewards and a positive learning rate")
        if not np.isfinite(rewards).all():
            raise ValueError("Rewards must be finite")
        baseline = (rewards.sum() - rewards) / (len(rewards) - 1)
        advantage = (rewards - baseline) / 25.0
        matrix = np.array([a.indices() for a in actions])
        for dim, p in enumerate(self.probabilities()):
            indicator = np.eye(len(p))[matrix[:, dim]]
            gradient = ((indicator - p) * advantage[:, None]).mean(axis=0)
            self.logits[dim] += lr * gradient

    def state(self) -> dict[str, float | list[float]]:
        ps = self.probabilities()
        return {
            "empathy_probabilities": ps[0].tolist(),
            "positivity_probabilities": ps[1].tolist(),
            "content_probabilities": ps[2].tolist(),
            "expected_empathy_count": float(ps[0] @ EMPATHY_COUNTS),
            "expected_positive_count": float(ps[1] @ POSITIVE_COUNTS),
            "specific_probability": float(ps[2][0]),
            "entropy": float(sum(-(p * np.log(p)).sum() for p in ps)),
        }


def train_step(policy: Policy, cases: list[Case], rng: np.random.Generator,
               batch_size: int = 48, lr: float = 0.30) -> float:
    if not cases or any(c.split != "train" for c in cases):
        raise ValueError("Optimization is restricted to the training split")
    actions = [policy.sample(rng) for _ in range(batch_size)]
    texts = [render(cases[int(rng.integers(len(cases)))], action,
                    variant=int(rng.integers(len(EMPATHY)))) for action in actions]
    rewards = np.array([proxy_reward(text).raw for text in texts])
    policy.update(actions, rewards, lr)
    return float(rewards.mean())
