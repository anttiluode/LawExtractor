from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import collect_hidden_rotor, load_public
from .scientist import discover
from .evaluate import evaluate


def _ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        prog="lawextractor",
        description="Invent a coordinate system, then see whether a small law survives intervention.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="make a hidden-world benchmark")
    c.add_argument("--out", default="runs/gate0/data")
    c.add_argument("--seed", type=int, default=0)

    d = sub.add_parser("discover", help="train candidate rulers using public data only")
    d.add_argument("--data", default="runs/gate0/data/public.npz")
    d.add_argument("--out", default="runs/gate0/discovery")
    d.add_argument("--dims", default="1,2,3,4,5")
    d.add_argument("--epochs", type=int, default=450)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--device", default="cpu")

    e = sub.add_parser("evaluate", help="compare raw and invented-coordinate laws")
    e.add_argument("--data", default="runs/gate0/data/public.npz")
    e.add_argument("--private", default="runs/gate0/data/private_truth.npz")
    e.add_argument("--model", required=True)
    e.add_argument("--out", default="runs/gate0/receipt.json")
    e.add_argument("--device", default="cpu")

    g = sub.add_parser("gate0", help="run collection, discovery, and the receipt")
    g.add_argument("--run-dir", default="runs/gate0")
    g.add_argument("--dims", default="1,2,3,4,5")
    g.add_argument("--epochs", type=int, default=450)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu")

    args = p.parse_args()

    if args.cmd == "collect":
        paths = collect_hidden_rotor(args.out, seed=args.seed)
        print(paths.public)
        return

    if args.cmd == "discover":
        report = discover(
            load_public(args.data),
            args.out,
            latent_dims=_ints(args.dims),
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
        print(report.to_json())
        return

    if args.cmd == "evaluate":
        result = evaluate(
            args.data,
            args.model,
            private_path=args.private,
            out_path=args.out,
            device=args.device,
        )
        print(json.dumps(result, indent=2))
        return

    run = Path(args.run_dir)
    paths = collect_hidden_rotor(run / "data", seed=args.seed)
    report = discover(
        load_public(paths.public),
        run / "discovery",
        latent_dims=_ints(args.dims),
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    result = evaluate(
        paths.public,
        report.winner.model_path,
        private_path=paths.private,
        out_path=run / "receipt.json",
        device=args.device,
    )
    print("\nWINNER")
    print(report.to_json())
    print("\nRECEIPT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
