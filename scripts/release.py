#!/usr/bin/env python3
"""One-command local release helper for physipy.

Automates the manual flow that used to live in the development guide:

    bump -> roll changelog -> build -> TestPyPI -> PyPI -> commit/tag/push

Design notes
------------
* **Pure helpers are importable and unit-tested** (``test/test_release.py``):
  version parsing/bumping, the tag<->version consistency guard, and the
  changelog roll are side-effect-free functions. The steps that touch the world
  (``git``, ``uv``) are thin subprocess wrappers gated behind confirmations.
* **Stdlib only** -- it shells out to ``git`` and ``uv`` (the project's tools),
  so there is nothing extra to install.
* **Safe ordering**: the version bump / changelog edits stay *uncommitted*
  until after PyPI publish succeeds, so a failure before the upload leaves
  nothing to unwind beyond
  ``git checkout -- physipy/_version.py CHANGELOG.md``.

Usage
-----
    uv run python scripts/release.py patch          # 0.2.9 -> 0.2.10
    uv run python scripts/release.py minor          # 0.2.9 -> 0.3.0
    uv run python scripts/release.py major          # 0.2.9 -> 1.0.0
    uv run python scripts/release.py 0.3.1          # explicit version
    uv run python scripts/release.py patch --dry-run        # show plan only
    uv run python scripts/release.py patch --no-testpypi    # skip TestPyPI
    uv run python scripts/release.py patch --no-push        # don't push tag
    uv run python scripts/release.py check        # tag<->version guard
    uv run python scripts/release.py check --tag 0.3.0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "physipy" / "_version.py"
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.md"
DEFAULT_BRANCH = "master"

_REVERT_HINT = (
    "Aborting before commit. Revert the working tree with:\n"
    "  git checkout -- physipy/_version.py CHANGELOG.md"
)

_VERSION_RE = re.compile(
    r"""^__version__\s*=\s*["'](?P<v>[^"']+)["']""", re.MULTILINE
)
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_UNRELEASED_RE = re.compile(
    r"^## \[Unreleased\][^\n]*\n(?P<body>.*?)(?=^## \[|\Z)",
    re.MULTILINE | re.DOTALL,
)
# Template placeholder kept in the (empty) [Unreleased] section. Must match the
# comment seeded in CHANGELOG.md so rolls are stable and it is never treated as
# a real entry.
_UNRELEASED_PLACEHOLDER = (
    "<!-- Add notes here as you work; the release script moves them under "
    "the\n     new version heading automatically. -->"
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in test/test_release.py)
# --------------------------------------------------------------------------- #
def read_version(text: str) -> str:
    """Extract the ``__version__`` string from the contents of _version.py."""
    m = _VERSION_RE.search(text)
    if not m:
        raise ValueError("no __version__ assignment found")
    return m.group("v")


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a strict ``MAJOR.MINOR.PATCH`` string into a tuple of ints."""
    m = _SEMVER_RE.match(version)
    if not m:
        raise ValueError(
            f"{version!r} is not a MAJOR.MINOR.PATCH version (e.g. '0.3.1')"
        )
    return int(m[1]), int(m[2]), int(m[3])


def bump_version(current: str, part: str) -> str:
    """Return the next version.

    ``part`` is ``major`` / ``minor`` / ``patch`` (semantic bump of
    ``current``) or an explicit ``MAJOR.MINOR.PATCH`` string (validated and
    required to be strictly greater than ``current``).
    """
    if part in ("major", "minor", "patch"):
        major, minor, patch = parse_version(current)
        if part == "major":
            major, minor, patch = major + 1, 0, 0
        elif part == "minor":
            minor, patch = minor + 1, 0
        else:
            patch += 1
        return f"{major}.{minor}.{patch}"
    # explicit version
    new = part
    if parse_version(new) <= parse_version(current):
        raise ValueError(
            f"explicit version {new!r} is not greater than current {current!r}"
        )
    return new


def replace_version(text: str, new_version: str) -> str:
    """Return ``text`` with the ``__version__`` value replaced by ``new``."""
    if not _VERSION_RE.search(text):
        raise ValueError("no __version__ assignment to replace")
    return _VERSION_RE.sub(f'__version__ = "{new_version}"', text, count=1)


def tag_for(version: str) -> str:
    """The git tag name for a version.

    physipy tags releases with the bare version, no ``v`` prefix (``0.3.1``),
    matching the existing tag history.
    """
    return version


def version_from_tag(tag: str) -> str:
    """Inverse of :func:`tag_for` (tolerates a stray ``v`` prefix)."""
    return tag[1:] if tag.startswith("v") else tag


def assert_tag_matches_version(tag: str, version: str) -> None:
    """Guard: raise if ``tag`` does not name ``version`` (``vX.Y.Z``)."""
    expected = tag_for(version)
    if tag != expected:
        raise ValueError(
            f"tag/version mismatch: tag {tag!r} != expected {expected!r} "
            f"for __version__ {version!r}"
        )


def roll_changelog(text: str, version: str, date: str) -> str:
    """Move ``[Unreleased]`` notes under a new ``[version] - date`` heading.

    A fresh, empty ``[Unreleased]`` section is left at the top. Returns the
    new changelog text. Raises if there is no ``[Unreleased]`` section.
    """
    m = _UNRELEASED_RE.search(text)
    if not m:
        raise ValueError("no '## [Unreleased]' section found in changelog")
    # Keep only real notes: drop the template placeholder comment(s) so they
    # don't get copied into the dated section.
    body = re.sub(r"<!--.*?-->", "", m.group("body"), flags=re.DOTALL).strip()
    fresh_unreleased = f"## [Unreleased]\n\n{_UNRELEASED_PLACEHOLDER}\n"
    new_section = f"## [{version}] - {date}\n"
    if body:
        new_section += f"\n{body}\n"
    replacement = f"{fresh_unreleased}\n{new_section}\n"
    return text[: m.start()] + replacement + text[m.end() :]


def changelog_has_entries(text: str) -> bool:
    """True if the ``[Unreleased]`` section has content beyond the comment."""
    m = _UNRELEASED_RE.search(text)
    if not m:
        return False
    body = m.group("body")
    # strip HTML comments and whitespace
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
    return bool(body)


# --------------------------------------------------------------------------- #
# Side-effecting helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], *, dry_run: bool = False, check: bool = True) -> int:
    """Echo and run a subprocess command (skipped under ``dry_run``)."""
    printable = " ".join(cmd)
    print(f"  $ {printable}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check).returncode


def _capture(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _confirm(prompt: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{prompt} [auto-yes]")
        return True
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _git_clean() -> bool:
    return _capture(["git", "status", "--porcelain"]) == ""


def _current_branch() -> str:
    return _capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _tag_exists(tag: str) -> bool:
    out = _capture(["git", "tag", "--list", tag])
    return out != ""


def _latest_tag() -> str | None:
    try:
        return _capture(["git", "describe", "--tags", "--abbrev=0"])
    except subprocess.CalledProcessError:
        return None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    """Tag<->version consistency guard (reusable, e.g. from CI later)."""
    version = read_version(VERSION_FILE.read_text())
    tag = args.tag or _latest_tag()
    if tag is None:
        print("No git tag found to check against.", file=sys.stderr)
        return 1
    try:
        assert_tag_matches_version(tag, version)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"OK: tag {tag} matches __version__ {version}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    dry = args.dry_run
    current = read_version(VERSION_FILE.read_text())
    new_version = bump_version(current, args.part)
    tag = tag_for(new_version)
    today = _dt.date.today().isoformat()

    print("physipy release plan")
    print(f"  current version : {current}")
    print(f"  new version     : {new_version}")
    print(f"  git tag         : {tag}")
    print(f"  TestPyPI smoke  : {'no' if args.no_testpypi else 'yes'}")
    print(f"  push tag        : {'no' if args.no_push else 'yes'}")
    print()

    # ---- pre-flight ------------------------------------------------------- #
    branch = _current_branch()
    if branch != DEFAULT_BRANCH:
        print(f"WARNING: on branch {branch!r}, not {DEFAULT_BRANCH!r}.")
    problems = []
    if _tag_exists(tag):
        problems.append(f"tag {tag} already exists")
    if not args.allow_dirty and not _git_clean():
        problems.append(
            "working tree is not clean (commit/stash, or pass --allow-dirty)"
        )
    changelog_empty = not changelog_has_entries(CHANGELOG_FILE.read_text())

    if problems:
        for p in problems:
            print(
                f"  {'would block' if dry else 'ERROR'}: {p}",
                file=None if dry else sys.stderr,
            )
        if not dry:
            return 1
    if changelog_empty:
        print("  note: the [Unreleased] changelog section looks empty")
        if not dry and not _confirm(
            "Release with no changelog entries?", assume_yes=args.yes
        ):
            return 1
    if not dry and not _confirm(
        f"Proceed with release {new_version}?", assume_yes=args.yes
    ):
        return 1
    if dry:
        print("\n(--dry-run: showing every step; nothing is executed)")

    # ---- [1/6] mutate (uncommitted) -------------------------------------- #
    print("\n[1/6] bump version + roll changelog")
    print(f"  - physipy/_version.py: {current} -> {new_version}")
    print(f"  - CHANGELOG.md: roll [Unreleased] -> [{new_version}] - {today}")
    if not dry:
        VERSION_FILE.write_text(
            replace_version(VERSION_FILE.read_text(), new_version)
        )
        CHANGELOG_FILE.write_text(
            roll_changelog(CHANGELOG_FILE.read_text(), new_version, today)
        )

    # ---- [2/6] build ------------------------------------------------------ #
    print("[2/6] build distributions (clean dist/ first)")
    dist = REPO_ROOT / "dist"
    if not dry and dist.exists():
        shutil.rmtree(dist)
    _run(["uv", "build"], dry_run=dry)

    # ---- [3/6] TestPyPI smoke -------------------------------------------- #
    if not args.no_testpypi:
        print("[3/6] publish to TestPyPI (smoke test)")
        _run(
            ["uv", "publish", "--index", "testpypi"], dry_run=dry, check=False
        )
        if not dry and not _confirm(
            "TestPyPI looks good?", assume_yes=args.yes
        ):
            print(_REVERT_HINT)
            return 1
    else:
        print("[3/6] TestPyPI smoke skipped (--no-testpypi)")

    # ---- [4/6] PyPI ------------------------------------------------------- #
    print("[4/6] publish to PyPI")
    if not dry and not _confirm(
        f"Upload {new_version} to PyPI (irreversible)?", assume_yes=args.yes
    ):
        print(_REVERT_HINT)
        return 1
    _run(["uv", "publish"], dry_run=dry)

    # ---- [5/6] commit + tag ---------------------------------------------- #
    print("[5/6] commit + tag")
    _run(["git", "add", str(VERSION_FILE), str(CHANGELOG_FILE)], dry_run=dry)
    _run(["git", "commit", "-m", f"Release {tag}"], dry_run=dry)
    _run(["git", "tag", "-a", tag, "-m", f"Release {tag}"], dry_run=dry)

    # ---- [6/6] push ------------------------------------------------------- #
    if args.no_push:
        print("[6/6] push skipped (--no-push)")
    else:
        print("[6/6] push")
        _run(["git", "push"], dry_run=dry)
        _run(["git", "push", "origin", tag], dry_run=dry)

    verb = "would release" if dry else "Done. Released"
    print(f"\n{verb} physipy {new_version} (tag {tag}).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="release.py",
        description="Local release helper for physipy.",
        epilog="examples: release.py patch | release.py 0.3.1 --dry-run | "
        "release.py check",
    )
    p.add_argument(
        "part",
        nargs="?",
        help="major | minor | patch | MAJOR.MINOR.PATCH  (or 'check' for the "
        "tag<->version guard)",
    )
    p.add_argument(
        "--tag", help="('check' only) tag to verify; default latest"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and stop, change nothing",
    )
    p.add_argument(
        "--no-testpypi",
        action="store_true",
        help="skip the TestPyPI smoke upload",
    )
    p.add_argument(
        "--no-push", action="store_true", help="commit + tag but do not push"
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="proceed even if the working tree is not clean",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="assume yes to all confirmations (non-interactive)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.part is None:
        parser.print_help()
        return 2
    if args.part == "check":
        return cmd_check(args)
    return cmd_release(args)


if __name__ == "__main__":
    raise SystemExit(main())
