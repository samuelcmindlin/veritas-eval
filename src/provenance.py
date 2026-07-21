"""Shared provenance primitives — CLAUDE.md invariant 2, DEV plan §7.

Every analysis entry point records git state via git_sha()/git_dirty() (a SHA
alone can point at code that did not produce the number — the dirty flag makes
that visible), hashes its config via config_hash(), and asserts external clones
against configs/external_pins.yaml via pin_provenance() (pins are asserted,
not decorative).

config_hash()'s recipe is byte-identical to stage 0b's original private helper
(sha256 over sort_keys JSON): committed reports embed hashes under this recipe,
so it must never change silently — tests/test_provenance.py pins it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

PINS_PATH = Path("configs/external_pins.yaml")


def git_sha(path: str = ".") -> str:
    return subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def git_dirty(path: str = ".") -> bool:
    """True if TRACKED files differ from HEAD (staged or unstaged). Untracked
    files are ignored on purpose: several docs are deliberately kept untracked
    in this repo, and they cannot change what committed code computes."""
    out = subprocess.run(
        ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout
    return bool(out.strip())


def git_provenance(path: str = ".") -> dict:
    return {"our_git_sha": git_sha(path), "our_git_dirty": git_dirty(path)}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def pin_provenance(name: str, clone_path: str | Path,
                   pins_path: Path = PINS_PATH) -> dict:
    """Assert an external clone against its configs/external_pins.yaml entry.

    If `clone_path` is the pinned clone location, its HEAD must equal the
    pinned commit — mismatch raises (re-clone at the pin; results produced
    from a drifted clone are not the pinned computation). A different path
    (e.g. a custom --experiment-dir) is recorded as explicitly un-pinned
    rather than silently passing.
    """
    pin = yaml.safe_load(Path(pins_path).read_text())[name]
    clone_path = Path(clone_path)
    head = git_sha(str(clone_path))
    fields = {"external_git_sha": head, "external_pin_name": name}
    if clone_path.resolve() != Path(pin["cloned_to"]).resolve():
        return {**fields, "external_pin_ok": False,
                "external_pin_note": f"clone path {clone_path} is not the "
                f"pinned location {pin['cloned_to']} — un-pinned input"}
    if head != pin["commit"]:
        raise ValueError(
            f"external clone {clone_path} is at {head}, but {pins_path} pins "
            f"{name} at {pin['commit']} — re-clone at the pinned commit")
    return {**fields, "external_pin_ok": True}
