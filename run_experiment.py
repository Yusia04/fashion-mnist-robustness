from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_SPECS = [
    {"name": "MLP", "model_key": "mlp", "augmentation": False},
    {"name": "SmallCNN", "model_key": "smallcnn", "augmentation": False},
    {"name": "SmallCNN_Aug", "model_key": "smallcnn", "augmentation": True},
]


def write_csv(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST lightweight CNN robustness experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec["name"] for spec in MODEL_SPECS],
        default=[spec["name"] for spec in MODEL_SPECS],
        help="Subset of models to train/evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch

        from src.data import get_test_loaders, get_train_val_loaders
        from src.evaluate import evaluate_model
        from src.models import build_model
        from src.train import fit
        from src.transforms import TEST_CONDITIONS
        from src.utils import CLASS_NAMES, count_parameters, ensure_dir, get_device, set_seed
        from src.visualize import (
            plot_accuracy_comparison,
            plot_accuracy_drop,
            plot_confusion_matrix,
            plot_training_curves,
            save_condition_samples,
            save_misclassified_examples,
        )
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing_name}. "
            "Install dependencies with `pip install -r requirements.txt` first."
        ) from exc

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / args.data_dir
    results_dir = ensure_dir(project_root / args.results_dir)
    figures_dir = ensure_dir(results_dir / "figures")

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    set_seed(args.seed)
    device = get_device(args.device)
    selected_specs = [spec for spec in MODEL_SPECS if spec["name"] in args.models]

    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")

    test_loaders = get_test_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        test_size=args.test_size,
        conditions=TEST_CONDITIONS,
        num_workers=args.num_workers,
    )

    metric_rows: list[dict[str, float | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    confusion_matrices: dict[tuple[str, str], object] = {}

    for spec in selected_specs:
        model_name = spec["name"]
        print(f"\n=== Training {model_name} ===")

        set_seed(args.seed)
        train_loader, val_loader = get_train_val_loaders(
            data_dir=data_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            batch_size=args.batch_size,
            use_augmentation=bool(spec["augmentation"]),
            seed=args.seed,
            num_workers=args.num_workers,
        )

        model = build_model(str(spec["model_key"])).to(device)
        print(f"Trainable parameters: {count_parameters(model):,}")

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            model_name=model_name,
        )
        history_rows.extend(history)

        print(f"--- Evaluating {model_name} ---")
        for condition_index, (condition, loader) in enumerate(test_loaders.items()):
            # Reset the seed so stochastic corruptions such as noise are comparable across models.
            set_seed(args.seed + 1000 + condition_index)
            metrics = evaluate_model(model, loader, device)
            metric_rows.append(
                {
                    "model": model_name,
                    "test_condition": condition,
                    "accuracy": round(float(metrics["accuracy"]), 6),
                    "macro_f1": round(float(metrics["macro_f1"]), 6),
                }
            )
            confusion_matrices[(model_name, condition)] = metrics["confusion_matrix"]
            print(
                f"{model_name:12s} {condition:10s} "
                f"acc={float(metrics['accuracy']):.4f} "
                f"macro_f1={float(metrics['macro_f1']):.4f}"
            )

        if not args.skip_plots and model_name in {"SmallCNN", "SmallCNN_Aug"}:
            save_misclassified_examples(
                model=model,
                test_loaders=test_loaders,
                device=device,
                class_names=CLASS_NAMES,
                output_path=figures_dir / f"misclassified_examples_{model_name}.png",
                max_examples=8,
            )

    metrics_path = results_dir / "metrics.csv"
    history_path = results_dir / "training_history.csv"
    write_csv(metrics_path, metric_rows, ["model", "test_condition", "accuracy", "macro_f1"])
    write_csv(
        history_path,
        history_rows,
        ["model", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
    )

    if not args.skip_plots:
        plot_accuracy_comparison(metric_rows, figures_dir / "accuracy_by_model_and_condition.png")
        plot_accuracy_drop(metric_rows, figures_dir / "accuracy_drop_from_clean.png")
        plot_training_curves(history_rows, figures_dir)
        save_condition_samples(
            data_dir=data_dir,
            class_names=CLASS_NAMES,
            output_path=figures_dir / "corrupted_test_samples.png",
        )

        for model_name in ["SmallCNN", "SmallCNN_Aug"]:
            key = (model_name, "clean")
            if key in confusion_matrices:
                plot_confusion_matrix(
                    matrix=confusion_matrices[key],
                    class_names=CLASS_NAMES,
                    output_path=figures_dir / f"confusion_matrix_{model_name}_clean.png",
                    title=f"{model_name} Confusion Matrix on Clean Test",
                    normalize=True,
                )

    print("\nSaved outputs:")
    print(f"- {metrics_path}")
    print(f"- {history_path}")
    if not args.skip_plots:
        print(f"- {figures_dir}")


if __name__ == "__main__":
    main()
