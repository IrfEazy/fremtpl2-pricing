"""Command line interface: `fremtpl2 <command> --config PATH`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import requests

from .audit import run_audit
from .config import (
    DEFAULT_CONFIG,
    DEFAULT_DOCS_CONFIG,
    DEFAULT_REPRODUCE_CONFIG,
    ConfigError,
    load_config,
    load_docs_config,
    load_reproduce_config,
    with_docs,
    with_reproduce,
)
from .data import DataError
from .docs_render import run_docs
from .download import DownloadError, run_download
from .evaluate import run_evaluate, run_publish, run_rehearsal
from .freeze import FreezeError, run_freeze, run_freeze_check
from .gbm import run_gbm
from .interpret import run_interpret
from .report import run_report
from .reproduce import run_reproduce
from .sensitivities import run_sensitivities
from .split import run_split
from .train import run_train

NETWORK_EXIT_CODE = (
    75  # EX_TEMPFAIL: a download step could not reach the network (not a data mismatch)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fremtpl2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    names = (
        "download", "audit", "split", "train", "sensitivities", "gbm", "interpret", "freeze",
        "evaluate", "report", "reproduce", "docs",
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
            p.add_argument(
                "--check",
                action="store_true",
                help="strict read-only verification of the committed manifest against the "
                "regenerated artifacts (writes nothing; allowed after test access)",
            )
        if name in ("freeze", "reproduce"):
            p.add_argument(
                "--reproduce-config",
                default=DEFAULT_REPRODUCE_CONFIG,
                help="YAML verification policy used by `freeze --check` and `reproduce` "
                "(default: %(default)s)",
            )
        if name == "reproduce":
            p.add_argument(
                "--force-clean",
                action="store_true",
                help="delete a non-empty reproduce.workdir (only inside paths.processed) first",
            )
        if name == "docs":
            p.add_argument(
                "--docs-config",
                default=DEFAULT_DOCS_CONFIG,
                help="YAML rendering policy of the generated documents (default: %(default)s)",
            )
            p.add_argument(
                "--check",
                action="store_true",
                help="render to memory and exit non-zero if a target differs (writes nothing)",
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
        elif args.command == "docs":
            res = run_docs(with_docs(cfg, load_docs_config(args.docs_config)), check=args.check)
            if args.check:
                print(
                    "docs: "
                    + (f"drift in {', '.join(res['drift'])}" if res["drift"] else "up to date")
                )
                return 1 if res["drift"] else 0
            print(f"docs: wrote {', '.join(res['written']) or 'nothing (up to date)'}")
        elif args.command == "freeze":
            if args.check:
                if args.force:
                    raise FreezeError("--check is read-only and cannot be combined with --force")
                res = run_freeze_check(
                    with_reproduce(cfg, load_reproduce_config(args.reproduce_config)), args.config
                )
                print(json.dumps(res, indent=2))
                return 0 if res["passed"] else 1
            run_freeze(cfg, args.config, force=args.force)
        elif args.command == "reproduce":
            rcfg = with_reproduce(cfg, load_reproduce_config(args.reproduce_config))
            rec = run_reproduce(rcfg, args.config, force_clean=args.force_clean)
            print(f"reproduce: {rec['verdict']} ({rec['total_seconds']:.0f}s), see the record")
            return 0 if rec["verdict"] in ("EXACT_PASS", "PORTABLE_NUMERIC_PASS") else 1
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
    except requests.RequestException as exc:  # network unavailable: a distinct exit code
        print(f"error: network unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return NETWORK_EXIT_CODE
    except (DownloadError, DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # network errors from requests
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
