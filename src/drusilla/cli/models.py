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
                      help="Re-download even if a cached copy is present.")

    p_rm = sub.add_parser("rm", help="Remove a cached model.")
    p_rm.add_argument("name")

    p_pa = sub.add_parser("path",
                          help="Print local paths (weights + arch config) for a cached model.")
    p_pa.add_argument("name")
    p_pa.add_argument("--download", action="store_true",
                      help="Download the model if it is not yet cached.")


def _cmd_list(_args: argparse.Namespace) -> int:
    manifests = registry.list_manifests()
    if not manifests:
        print("no models registered.")
        return 0
    for name in sorted(manifests):
        mf = manifests[name]
        st = registry.local_status(name)
        cache = "cached" if st["cached"] else "not cached"
        clade = mf.data.get("target_clade") or "-"
        arch = mf.data.get("architecture") or "-"
        print(f"{name:20s} v{mf.version:6s}  {cache:12s}  clade={clade}  arch={arch}")
        if mf.data.get("comment"):
            print(f"  {mf.data['comment']}")
    print()
    print(f"cache dir: {registry.cache_dir()}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    resolved = registry.resolve_model(args.name, force=args.force)
    print(f"weights  : {resolved.weights_path}")
    print(f"arch cfg : {resolved.arch_config_path}")
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    removed = registry.clear(args.name)
    if not removed:
        print(f"{args.name}: nothing to remove")
    else:
        for p in removed:
            print(f"removed {p}")
    return 0


def _cmd_path(args: argparse.Namespace) -> int:
    if args.download:
        resolved = registry.resolve_model(args.name)
        print(f"weights  : {resolved.weights_path}")
        print(f"arch cfg : {resolved.arch_config_path}")
        return 0
    st = registry.local_status(args.name)
    if not st["cached"]:
        raise SystemExit(
            f"{args.name}: not cached. Run `drusilla models download {args.name}` "
            f"or pass --download."
        )
    print(f"weights  : {st['weights_path']}")
    print(f"arch cfg : {st['arch_config_path']}")
    return 0


_DISPATCH = {
    "list": _cmd_list,
    "download": _cmd_download,
    "rm": _cmd_rm,
    "path": _cmd_path,
}


def run(args: argparse.Namespace) -> int:
    return _DISPATCH[args.action](args)
