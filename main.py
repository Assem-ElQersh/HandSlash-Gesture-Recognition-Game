"""
Gesture Trajectory Engine — Command-Line Interface

Usage
-----
  python main.py collect                        Record gesture samples via webcam
  python main.py train --model lstm             Train a single model
  python main.py train --model all              Train all 5 models
  python main.py benchmark                      Compare all trained models
  python main.py demo                           Launch the fruit-slash game
  python main.py infer --model transformer      Live gesture inference overlay
  python main.py load-external --format csv --path my_gestures.csv
"""

import argparse
import sys
from pathlib import Path


def cmd_collect(args):
    from data.collector import run_collector
    run_collector(config_path=args.config)


def cmd_train(args):
    if args.model == "all":
        from training.trainer import train_all
        train_all(config_path=args.config)
    else:
        from training.trainer import train
        train(args.model, config_path=args.config)


def cmd_benchmark(args):
    from benchmarks.compare_models import run_benchmark
    run_benchmark(config_path=args.config)


def cmd_demo(args):
    demo_path = Path("demos/fruit_slash_game.py")
    if not demo_path.exists():
        print("Demo not found. Expected: demos/fruit_slash_game.py")
        sys.exit(1)
    import runpy
    runpy.run_path(str(demo_path), run_name="__main__")


def cmd_infer(args):
    import yaml
    from pathlib import Path as P

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gestures = cfg["gestures"]
    ckpt_dir = cfg["paths"]["checkpoints"]
    name = args.model

    ext = ".pkl" if name in ("dtw", "hmm") else ".pt"
    ckpt = P(ckpt_dir) / f"{name}{ext}"

    if not ckpt.exists():
        print(f"No checkpoint for '{name}' at {ckpt}.")
        print(f"Run: python main.py train --model {name}")
        sys.exit(1)

    model = _load_model(name, gestures, ckpt, cfg)
    from realtime.inference import run_live_inference
    run_live_inference(model, config_path=args.config)


def cmd_load_external(args):
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    raw_dir = cfg["paths"]["raw_data"]

    fmt = args.format.lower()
    if fmt == "csv":
        from data.loader import load_csv, save_external_samples
        samples = load_csv(args.path)
    elif fmt == "dollar_n":
        from data.loader import load_dollar_n, save_external_samples
        samples = load_dollar_n(args.path)
    elif fmt == "txt":
        from data.loader import load_txt_directory, save_external_samples
        samples = load_txt_directory(args.path)
    else:
        print(f"Unknown format: {fmt}. Choose from: csv, dollar_n, txt")
        sys.exit(1)

    from data.loader import save_external_samples
    n = save_external_samples(samples, raw_dir)
    print(f"Loaded {n} samples from {args.path} → {raw_dir}/")


def _load_model(name, gestures, ckpt, cfg):
    if name == "dtw":
        from models.dtw import DTWClassifier
        return DTWClassifier(gestures).load(ckpt)
    if name == "hmm":
        from models.hmm import HMMClassifier
        return HMMClassifier(gestures).load(ckpt)
    if name == "cnn":
        from models.cnn import CNNClassifier
        c = cfg["models"]["cnn"]
        return CNNClassifier(
            gestures,
            channels=tuple(c.get("channels", [64, 128])),
            dropout=c.get("dropout", 0.3),
        ).load(ckpt)
    if name == "lstm":
        from models.lstm import LSTMClassifier
        c = cfg["models"]["lstm"]
        return LSTMClassifier(
            gestures,
            hidden_size=c.get("hidden_size", 128),
            num_layers=c.get("num_layers", 2),
            dropout=c.get("dropout", 0.3),
            bidirectional=c.get("bidirectional", True),
        ).load(ckpt)
    if name == "transformer":
        from models.transformer import TransformerClassifier
        c = cfg["models"]["transformer"]
        return TransformerClassifier(
            gestures,
            d_model=c.get("d_model", 64),
            nhead=c.get("nhead", 4),
            num_encoder_layers=c.get("num_encoder_layers", 2),
            dim_feedforward=c.get("dim_feedforward", 128),
            dropout=c.get("dropout", 0.1),
        ).load(ckpt)
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(
        prog="gesture_engine",
        description="Gesture Trajectory Learning Engine",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # collect
    sub.add_parser("collect", help="Record gesture samples via webcam")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument(
        "--model", required=True,
        choices=["dtw", "hmm", "cnn", "lstm", "transformer", "all"],
        help="Model to train, or 'all'",
    )

    # benchmark
    sub.add_parser("benchmark", help="Compare all trained models")

    # demo
    sub.add_parser("demo", help="Launch the fruit-slash game demo")

    # infer
    p_infer = sub.add_parser("infer", help="Live gesture inference with webcam overlay")
    p_infer.add_argument(
        "--model", required=True,
        choices=["dtw", "hmm", "cnn", "lstm", "transformer"],
    )

    # load-external
    p_ext = sub.add_parser("load-external", help="Import an external gesture dataset")
    p_ext.add_argument("--format", required=True, choices=["csv", "dollar_n", "txt"])
    p_ext.add_argument("--path", required=True, help="Path to file or directory")

    args = parser.parse_args()

    dispatch = {
        "collect": cmd_collect,
        "train": cmd_train,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
        "infer": cmd_infer,
        "load-external": cmd_load_external,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
