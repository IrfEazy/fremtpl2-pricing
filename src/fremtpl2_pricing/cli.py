"""Command line interface: `fremtpl2 <command> --config PATH`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .audit import run_audit
from .config import DEFAULT_CONFIG, ConfigError, load_config
from .data import DataError
from .download import DownloadError, run_download
from .split import run_split
from .train import run_train

STUBS = ("evaluate", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fremtpl2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "audit", "split", "train", *STUBS):
        p = sub.add_parser(name, help=f"{name} step")
        p.add_argument(
            "--config", default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)"
        )
        if name == "download":
            p.add_argument("--force", action="store_true", help="re-download even if files exist")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.command in STUBS:
        print(f"{args.command}: not implemented yet", file=sys.stderr)
        return 2
    try:
        if args.command == "audit":
            run_audit(cfg)
        elif args.command == "split":
            run_split(cfg)
        elif args.command == "train":
            run_train(cfg)
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
