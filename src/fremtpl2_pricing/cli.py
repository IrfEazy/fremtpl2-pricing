"""Command line interface: `fremtpl2 <command> --config PATH`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .audit import run_audit
from .config import DEFAULT_CONFIG, ConfigError, load_config
from .data import DataError
from .download import DownloadError, run_download
from .evaluate import run_evaluate, run_publish, run_rehearsal
from .freeze import FreezeError, run_freeze
from .gbm import run_gbm
from .interpret import run_interpret
from .report import run_report
from .sensitivities import run_sensitivities
from .split import run_split
from .train import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fremtpl2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    names = (
        "download", "audit", "split", "train", "sensitivities", "gbm", "interpret", "freeze",
        "evaluate", "report",
    )  # fmt: skip
    for name in names:
        p = sub.add_parser(name, help=f"{name} step")
        p.add_argument(
            "--config", default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)"
        )
        if name == "download":
            p.add_argument("--force", action="store_true", help="re-download even if files exist")
        if name == "freeze":
            p.add_argument(
                "--force",
                action="store_true",
                help="replace an existing manifest (refused once the test set was accessed)",
            )
        if name == "evaluate":
            p.add_argument(
                "--technical-rerun",
                action="store_true",
                help="rerun after a technical failure of an attempt that did not complete",
            )
            p.add_argument("--reason", help="reason of the technical rerun (required with it)")
            p.add_argument(
                "--rehearsal",
                action="store_true",
                help="run the identical code path on the train fold freeze.rehearsal.fold "
                "(no test row; outputs under paths.processed)",
            )
            p.add_argument(
                "--publish-only",
                action="store_true",
                help="republish the COMPLETE attempt's staged, hashed outputs (no recomputation)",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        if args.command == "audit":
            run_audit(cfg)
        elif args.command == "split":
            run_split(cfg)
        elif args.command == "train":
            run_train(cfg)
        elif args.command == "gbm":
            run_gbm(cfg)
        elif args.command == "sensitivities":
            run_sensitivities(cfg)
        elif args.command == "interpret":
            res = run_interpret(cfg)
            print(
                "interpret: " + ", ".join(f"{k} {v:.1f}s" for k, v in res["timing_seconds"].items())
            )
        elif args.command == "report":
            run_report(cfg)
        elif args.command == "freeze":
            run_freeze(cfg, args.config, force=args.force)
        elif args.command == "evaluate":
            if args.reason and not args.technical_rerun:
                raise FreezeError("--reason is only used with --technical-rerun")
            if args.rehearsal + args.publish_only + args.technical_rerun > 1:
                raise FreezeError(
                    "--rehearsal, --publish-only and --technical-rerun are mutually exclusive"
                )
            if args.rehearsal:
                run_rehearsal(cfg, args.config)
            elif args.publish_only:
                run_publish(cfg, args.config)
            else:
                run_evaluate(cfg, args.config, args.technical_rerun, args.reason)
        else:
            run_download(cfg, force=args.force)
    except (DownloadError, DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # network errors from requests
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
