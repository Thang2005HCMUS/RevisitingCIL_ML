import argparse
import copy
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def load_json(settings_path: str) -> dict:
    """Read Experiment 3 sweep settings from a JSON file."""
    with open(settings_path, "r", encoding="utf-8") as data_file:
        return json.load(data_file)


def setup_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Experiment 3 (2D alpha-PCA sensitivity sweep)."""
    parser = argparse.ArgumentParser(
        description="Experiment 3: 2D Sensitivity Sweep over weighted alpha-mixing and Stage-1 PCA dimension k."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/exp3_analysis/sweep_cifar224_adapter.json",
        help="Path to Experiment 3 JSON sweep config.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to save/load cached Stage-0 dual-branch features.",
    )
    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="Re-run Stage 0 adaptation and feature extraction even if cache_path exists.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write CSV sweep table and PNG figures.",
    )
    parser.add_argument(
        "--alpha_grid",
        nargs="+",
        type=float,
        default=None,
        help="Override mixing coefficient grid alpha in [0, 1].",
    )
    parser.add_argument(
        "--k_grid",
        nargs="+",
        type=int,
        default=None,
        help="Override PCA/subsampling dimension grid k.",
    )
    parser.add_argument(
        "--random_seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds for Random Subsampling baseline.",
    )
    parser.add_argument(
        "--tuned_epoch",
        type=int,
        default=None,
        help="Override Stage 0 adaptation epochs.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (wandb.Table, plots, and summary metrics).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="nnm-hcmus",
        help="Weights & Biases entity (team or username).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Group7-ML-Lab",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Weights & Biases run group.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb_tags",
        nargs="+",
        default=None,
        help="Additional Weights & Biases tags.",
    )
    return parser


def run_experiment_3(args_dict: dict) -> dict:
    """Execute feature caching (or load cache), 2D sensitivity sweep, plotting, and optional W&B logging."""
    from analysis.sensitivity_sweep import (
        DEFAULT_ALPHA_GRID,
        DEFAULT_K_GRID,
        DEFAULT_RANDOM_SEEDS,
        check_cache_compatibility,
        export_sweep_csv,
        extract_and_cache_features,
        generate_sensitivity_plots,
        load_feature_cache,
        log_sensitivity_to_wandb,
        run_2d_sensitivity_sweep,
    )

    run_args = copy.deepcopy(args_dict)
    if "convnet_type" not in run_args and run_args.get("backbone_type"):
        run_args["convnet_type"] = run_args["backbone_type"]
    run_args["use_wandb"] = bool(run_args.get("wandb") or run_args.get("use_wandb"))

    seed_raw = run_args.get("seed", 1993)
    seed = seed_raw[0] if isinstance(seed_raw, (list, tuple)) and seed_raw else seed_raw
    run_args["seed"] = int(seed)

    dataset = str(run_args.get("dataset", "unknown"))
    output_dir = run_args.get("output_dir") or os.path.join("experiments", "exp3_analysis", dataset)
    os.makedirs(output_dir, exist_ok=True)
    cache_path = run_args.get("cache_path") or os.path.join(output_dir, f"{dataset}_seed{seed}_features.pt")
    force_reextract = bool(run_args.get("force_reextract", False))

    cached_features = None
    effective_force_reextract = force_reextract
    if cache_path and os.path.exists(cache_path) and not force_reextract:
        candidate = load_feature_cache(cache_path)
        compatible, mismatches = check_cache_compatibility(candidate, run_args)
        if compatible:
            logging.info(f"[Exp 3] Reusing cached features from {cache_path}")
            cached_features = candidate
        else:
            logging.warning(
                f"[Exp 3] Cache at {cache_path} is incompatible with current config "
                f"({'; '.join(mismatches)}); re-extracting features."
            )
            effective_force_reextract = True

    if cached_features is None:
        from data.task_manager import ContinualTaskManager
        from framework.learner import APERLearner, _set_device, _set_random

        _set_random(run_args["seed"])
        _set_device(run_args)
        task_manager = ContinualTaskManager(run_args)
        learner = APERLearner(run_args)
        cached_features = extract_and_cache_features(
            learner,
            task_manager,
            cache_path=cache_path,
            force_reextract=effective_force_reextract,
        )

    alpha_grid = run_args.get("alpha_grid") or DEFAULT_ALPHA_GRID
    k_grid = run_args.get("k_grid") or DEFAULT_K_GRID
    random_seeds = run_args.get("random_seeds") or DEFAULT_RANDOM_SEEDS
    sweep_method = run_args.get("sweep_method", "both")

    sweep_result = run_2d_sensitivity_sweep(
        cached_features,
        alpha_grid=alpha_grid,
        k_grid=k_grid,
        method=sweep_method,
        random_seeds=random_seeds,
    )

    csv_path = os.path.join(output_dir, f"sweep_{dataset}_seed{seed}.csv")
    export_sweep_csv(sweep_result, csv_path)
    plot_paths = generate_sensitivity_plots(sweep_result, output_dir)
    log_sensitivity_to_wandb(sweep_result, run_args, plot_paths=plot_paths, csv_path=csv_path)

    summary = sweep_result.get("summary", {})
    best = summary.get("best_overall")
    default_aper = summary.get("default_aper")
    if best:
        logging.info(
            f"[Exp 3 | {dataset}] Best (alpha*={best['alpha']:.1f}, k*={best['k']}, eff_k={best['effective_k']}): "
            f"avg_acc={best['avg_acc']:.2f}%, last_acc={best['last_acc']:.2f}%"
        )
    if default_aper:
        logging.info(
            f"[Exp 3 | {dataset}] Default APER (alpha={default_aper['alpha']:.1f}, k={default_aper['k']}): "
            f"avg_acc={default_aper['avg_acc']:.2f}%, last_acc={default_aper['last_acc']:.2f}%"
        )

    return {
        "sweep_result": sweep_result,
        "csv_path": csv_path,
        "plot_paths": plot_paths,
        "cache_path": cache_path,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s")
    args = setup_parser().parse_args()
    param = load_json(args.config)
    cli_args = vars(args)

    args_dict = copy.deepcopy(param)
    optional_keys = (
        "cache_path",
        "output_dir",
        "alpha_grid",
        "k_grid",
        "random_seeds",
        "tuned_epoch",
        "wandb_group",
        "wandb_name",
        "wandb_tags",
    )
    for key in optional_keys:
        if cli_args.get(key) is not None:
            args_dict[key] = cli_args[key]

    args_dict["config"] = cli_args["config"]
    args_dict["force_reextract"] = bool(cli_args.get("force_reextract"))
    args_dict["wandb_entity"] = cli_args.get("wandb_entity") or args_dict.get("wandb_entity") or "nnm-hcmus"
    args_dict["wandb_project"] = cli_args.get("wandb_project") or args_dict.get("wandb_project") or "Group7-ML-Lab"
    args_dict["wandb"] = bool(cli_args.get("wandb") or args_dict.get("wandb"))
    args_dict["use_wandb"] = bool(args_dict["wandb"] or args_dict.get("use_wandb"))

    seeds = args_dict.get("seed", [1993])
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]

    results = []
    for seed in seeds:
        per_seed_args = copy.deepcopy(args_dict)
        per_seed_args["seed"] = seed
        if len(seeds) > 1 and cli_args.get("cache_path") is None and "cache_path" in param:
            base, ext = os.path.splitext(param["cache_path"])
            per_seed_args["cache_path"] = f"{base}_seed{seed}{ext}"
        results.append(run_experiment_3(per_seed_args))
    return results


if __name__ == "__main__":
    main()
