"""`drusilla models` subcommands: list / download / rm / path."""

from __future__ import annotations

import argparse

from .. import registry


def add_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    sub.add_parser("list", help="List available models and their local cache status.")

    p_dl = sub.add_parser("download", help="Download a model into the local cache.")
    p_dl.add_argument("name", help="Model name (see `drusilla models list`).")
    p_dl.add_argument("--force", action="store_true",
                      help="Re-download even if the cached file is valid.")

    p_rm = sub.add_parser("rm", help="Remove a cached model.")
    p_rm.add_argument("name")

    p_pa = sub.add_parser("path", help="Print the local path to a cached model.")
    p_pa.add_argument("name")
    p_pa.add_argument("--download", action="store_true",
                      help="Download the model if it is not yet cached.")


def _cmd_list(_args: argparse.Namespace) -> int:
    entries = registry.load_registry()
    if not entries:
        print("no models registered.")
        return 0
    for name in sorted(entries):
        e = entries[name]
        st = registry.local_status(name)
        cache = "cached" if st["cached"] else "not cached"
        clades = ",".join(e.clades) if e.clades else "-"
        print(f"{name:20s} v{e.version:6s}  {cache:12s}  clades={clades}")
        if e.description:
            print(f"  {e.description}")
    print()
    print(f"cache dir: {registry.cache_dir()}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    path = registry.resolve_weights(args.name, force=args.force)
    print(str(path))
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    removed = registry.clear(args.name)
    if removed is None:
        print(f"{args.name}: nothing to remove")
    else:
        print(f"removed {removed}")
    return 0


def _cmd_path(args: argparse.Namespace) -> int:
    if args.download:
        path = registry.resolve_weights(args.name)
    else:
        st = registry.local_status(args.name)
        if not st["cached"]:
            raise SystemExit(
                f"{args.name}: not cached. Run `drusilla models download {args.name}` "
                f"or pass --download."
            )
        path = st["path"]
    print(str(path))
    return 0


_DISPATCH = {
    "list": _cmd_list,
    "download": _cmd_download,
    "rm": _cmd_rm,
    "path": _cmd_path,
}


def run(args: argparse.Namespace) -> int:
    return _DISPATCH[args.action](args)
