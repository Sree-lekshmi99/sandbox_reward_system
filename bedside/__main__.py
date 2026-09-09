"""Command-line entry points; run from the repo or install with pip -e ."""
from pathlib import Path
import argparse
import sys
from .dataset import load_cases
from .detector import Detector
from .experiment import Config, run_experiment, save_csv, save_json, save_jsonl, selection_experiment
from .report import build_report, plot_selection


def integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers, for example 7,19,42") from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthetic reward-hacking sandbox. Not a medical chatbot.")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the offline policy and paired best-of-N experiments")
    run.add_argument("--out", type=Path, default=Path("results-local"))
    run.add_argument("--steps", type=int, default=300)
    run.add_argument("--seeds", type=integers, default=(7, 19, 42, 73, 101))
    run.add_argument("--batch-size", type=int, default=48)
    run.add_argument("--learning-rate", type=float, default=.30)
    run.add_argument("--eval-every", type=int, default=20)
    run.add_argument("--eval-samples", type=int, default=16)
    run.add_argument("--selection-trials", type=int, default=12)
    run.add_argument("--budgets", type=integers, default=(1, 2, 4, 8, 16, 32, 64))
    report = commands.add_parser("report", help="Rebuild HTML/charts from an existing offline run")
    report.add_argument("--out", type=Path, default=Path("results"))
    collect = commands.add_parser("collect", help="Optional: collect unchanged replies from an installed Ollama model")
    collect.add_argument("--model", required=True, help="Exact name of an already-installed local model")
    collect.add_argument("--bank", type=Path, default=Path("results-local/ollama-bank.jsonl"))
    collect.add_argument("--base-url", default="http://localhost:11434")
    collect.add_argument("--timeout", type=float, default=120)
    collect.add_argument("--n", type=int, default=64)
    collect.add_argument("--seed", type=int, default=7)
    collect.add_argument("--resume", action="store_true")
    bank = commands.add_parser("run-bank", help="Run paired best-of-N on a saved real-model candidate bank")
    bank.add_argument("--bank", type=Path, required=True)
    bank.add_argument("--out", type=Path, default=Path("results-local/model-selection"))
    bank.add_argument("--budgets", type=integers, default=(1, 2, 4, 8, 16, 32, 64))
    bank.add_argument("--seeds", type=integers, default=(7, 19, 42, 73, 101))
    bank.add_argument("--selection-trials", type=int, default=12)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "run":
            config = Config(seeds=args.seeds, steps=args.steps, batch_size=args.batch_size,
                            learning_rate=args.learning_rate, eval_every=args.eval_every,
                            eval_samples=args.eval_samples, selection_trials=args.selection_trials,
                            budgets=args.budgets)
            summary = run_experiment(config, args.out)
            build_report(args.out)
            print(f"Completed: {args.out / 'index.html'}")
            for phase in ("initial", "final"):
                m = summary["training"][phase]
                print(f"{phase:7s} proxy={m['proxy']['mean']:.2f}, rubric={m['quality']['mean']:.2f}")
        elif args.command == "report":
            build_report(args.out)
            print(f"Rebuilt: {args.out / 'index.html'}")
        elif args.command == "collect":
            from .ollama_backend import OllamaClient, collect_bank
            n = collect_bank(OllamaClient(args.model, args.base_url, args.timeout), args.bank, args.n, args.seed, args.resume)
            print(f"Saved {n} real-model responses to {args.bank}; no performance claim has been made.")
        elif args.command == "run-bank":
            from .ollama_backend import load_bank
            import hashlib
            config = Config(seeds=args.seeds, budgets=args.budgets, selection_trials=args.selection_trials)
            config.validate()
            if (args.out / "summary.json").exists():
                raise FileExistsError("Choose a new output directory for this model-bank run")
            bank = load_bank(args.bank)
            detector = Detector.calibrate(load_cases("calibration"))
            metrics, raw, examples = selection_experiment(config, detector, bank)
            args.out.mkdir(parents=True, exist_ok=True)
            save_csv(args.out / "selection_metrics.csv", metrics)
            save_csv(args.out / "selection_evaluations.csv", raw)
            save_jsonl(args.out / "selection_examples.jsonl", examples)
            save_json(args.out / "detector.json", detector.as_dict())
            save_json(args.out / "summary.json", {
                "status": "executed_model_bank_selection_only", "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
                "seeds": list(config.seeds), "budgets": list(config.budgets), "trials": config.selection_trials,
                "caveats": ["No model weights were trained", "Offline template-calibrated style thresholds are reused, not model-calibrated",
                            "Quality is still a lexical rubric, not a stronger-model or human judgment",
                            "Seed intervals are conditional on one fixed candidate bank"],
            })
            plot_selection(args.out, metrics, "Real-model candidate bank; fixed lexical rubric and template-calibrated detector")
            print(f"Saved model-bank selection results to {args.out}. Read summary.json caveats before interpreting.")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
