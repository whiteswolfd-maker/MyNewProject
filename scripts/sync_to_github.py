#!/usr/bin/env python3
"""Mirror the working source tree into the GitHub upload repository.

The working folder is where code is edited; the upload folder is the local
clone that is committed and pushed.  This script copies only source paths, so
generated products (``outputs/``), caches (``__pycache__``) and machine-local
files never reach the upload clone.

What is synced
--------------
directories : src/, tests/, scripts/, configs/
files       : README.md, AGENTS.md, CLAUDE.md, pyproject.toml,
              requirements.txt, conftest.py, .gitignore

Deletions
---------
A file that was tracked in the upload repository under one of the synced
directories but no longer exists in the working tree is deleted there, so a
removed module does not stay on GitHub.  Untracked extra files are reported
but only deleted with ``--prune-untracked``.

The script never runs ``git add/commit/push``; it prints the next commands.
Use ``--check`` to preview every change without writing.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

SYNC_DIRS = ("src", "tests", "scripts", "configs")
SYNC_FILES = (
    "README.md", "AGENTS.md", "CLAUDE.md",
    "pyproject.toml", "requirements.txt", "conftest.py", ".gitignore",
)
IGNORE_DIRS = {"__pycache__", ".pytest_cache", "outputs"}
IGNORE_SUFFIXES = (".pyc", ".pyo")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_tree(base: Path, root: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        for name in sorted(filenames):
            if name.endswith(IGNORE_SUFFIXES):
                continue
            path = Path(dirpath) / name
            yield path.relative_to(root).as_posix()


def _iter_source_files(source: Path) -> set[str]:
    files: set[str] = set()
    for dirname in SYNC_DIRS:
        base = source / dirname
        if base.is_dir():
            files.update(_iter_tree(base, source))
    for name in SYNC_FILES:
        path = source / name
        if path.is_file():
            files.add(name)
    return files


def _git_tracked(repo: Path) -> set[str] | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        entry.decode("utf-8", "surrogateescape").replace("\\", "/")
        for entry in completed.stdout.split(b"\0") if entry
    }


def _git_status(repo: Path) -> list[str] | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "status", "--short"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _under_synced_dirs(rel: str) -> bool:
    return rel.split("/", 1)[0] in SYNC_DIRS


def _remove_empty_dirs(target: Path) -> list[str]:
    removed: list[str] = []
    for dirname in SYNC_DIRS:
        base = target / dirname
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=False):
            path = Path(dirpath)
            if path == base:
                continue
            if not any(path.iterdir()):
                path.rmdir()
                removed.append(path.relative_to(target).as_posix())
    return removed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync the working source tree into the GitHub upload clone.",
    )
    parser.add_argument(
        "--source", default=str(Path(__file__).resolve().parent.parent),
        help="working code folder (default: this repository root)",
    )
    parser.add_argument(
        "--target", default=r"F:\MyNewProject\MyNewProject",
        help="local clone of the GitHub upload repository",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report what would change; write and delete nothing",
    )
    parser.add_argument(
        "--no-prune", action="store_true",
        help="do not delete tracked upload files that no longer exist here",
    )
    parser.add_argument(
        "--prune-untracked", action="store_true",
        help="also delete untracked extra files under the synced directories",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="allow a target that does not look like a git clone",
    )
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()

    if not (source / "pyproject.toml").is_file() or not (source / "src").is_dir():
        print(f"error: {source} does not look like the working repository", file=sys.stderr)
        return 2
    if source == target:
        print("error: source and target are the same folder", file=sys.stderr)
        return 2
    if not target.is_dir():
        print(f"error: target {target} does not exist", file=sys.stderr)
        return 2
    if not (target / ".git").exists() and not args.force:
        print(
            f"error: {target} has no .git; pass --force only if this is intentional",
            file=sys.stderr,
        )
        return 2

    source_files = _iter_source_files(source)
    tracked = _git_tracked(target)

    created: list[str] = []
    updated: list[str] = []
    unchanged = 0
    for rel in sorted(source_files):
        src = source / rel
        dst = target / rel
        if dst.is_file() and _sha256(dst) == _sha256(src):
            unchanged += 1
            continue
        (created if not dst.exists() else updated).append(rel)
        if not args.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    pruned: list[str] = []
    if tracked is not None and not args.no_prune:
        for rel in sorted(tracked):
            if not _under_synced_dirs(rel) or rel in source_files:
                continue
            dst = target / rel
            if not dst.is_file():
                continue
            pruned.append(rel)
            if not args.check:
                dst.unlink()

    target_files: set[str] = set()
    for dirname in SYNC_DIRS:
        base = target / dirname
        if base.is_dir():
            target_files.update(_iter_tree(base, target))
    tracked_set = tracked or set()
    extra = sorted(
        rel for rel in target_files
        if rel not in source_files and rel not in tracked_set
    )
    deleted_extra: list[str] = []
    if args.prune_untracked and not args.check:
        for rel in extra:
            (target / rel).unlink()
            deleted_extra.append(rel)
    removed_dirs = [] if args.check else _remove_empty_dirs(target)

    verb = "would " if args.check else ""
    print(f"source: {source}")
    print(f"target: {target}")
    print(f"{verb}add:       {len(created)}")
    for rel in created:
        print(f"  + {rel}")
    print(f"{verb}update:    {len(updated)}")
    for rel in updated:
        print(f"  M {rel}")
    print(f"unchanged: {unchanged}")
    print(f"{verb}prune tracked: {len(pruned)}")
    for rel in pruned:
        print(f"  - {rel}")
    if extra:
        print(f"untracked extras under synced dirs: {len(extra)}")
        for rel in extra[:20]:
            print(f"  ? {rel}")
        if len(extra) > 20:
            print(f"  ... {len(extra) - 20} more")
        if args.prune_untracked:
            print(f"{verb}prune untracked: {len(deleted_extra)}")
    if removed_dirs:
        print(f"removed empty dirs: {len(removed_dirs)}")

    status = _git_status(target)
    if status:
        print("upload repository status after sync:")
        for line in status[:40]:
            print(f"  {line}")
        if len(status) > 40:
            print(f"  ... {len(status) - 40} more")
    elif status is not None:
        print("upload repository is clean after sync")

    if not args.check:
        print("\nnext steps (not run automatically):")
        print(f'  git -C "{target}" add -A')
        print(f'  git -C "{target}" commit -m "sync from working tree"')
        print(f'  git -C "{target}" push origin main')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
