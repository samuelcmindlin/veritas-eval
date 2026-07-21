"""Shared provenance primitives (src/provenance.py) — invariant-2 machinery.

The config_hash recipe is pinned by known-answer: committed reports
(results/stage0b/report.json config_hash be166d7fbc78...) embed hashes under
this exact recipe, so a silent change here would orphan them.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from provenance import PINS_PATH, config_hash, git_dirty, git_provenance, git_sha, pin_provenance


class TestConfigHash:
    def test_known_answer_recipe_pinned(self):
        cfg = {"b": [2, 3], "a": 1}
        expected = hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        assert config_hash(cfg) == expected
        # literal pin: fails if the serialization recipe ever drifts
        assert config_hash({"id": "x", "n": 2}) == (
            hashlib.sha256(b'{"id": "x", "n": 2}').hexdigest())

    def test_key_order_invariant(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


class TestGitState:
    def test_sha_shape(self):
        sha = git_sha(".")
        assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")

    def test_dirty_is_bool_and_in_provenance(self):
        p = git_provenance(".")
        assert p["our_git_sha"] == git_sha(".")
        assert isinstance(p["our_git_dirty"], bool)
        assert isinstance(git_dirty("."), bool)


def _make_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "f.txt").write_text("x")
    env_args = ["-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "-C", str(path), "add", "f.txt"], check=True)
    subprocess.run(["git", *env_args, "-C", str(path), "commit", "-qm", "c"],
                   check=True)
    return git_sha(str(path))


class TestPinProvenance:
    def _pins(self, tmp_path: Path, clone: Path, commit: str) -> Path:
        pins = tmp_path / "pins.yaml"
        pins.write_text(
            f"dep:\n  repo: x\n  commit: {commit}\n  cloned_to: {clone}\n")
        return pins

    def test_match_passes(self, tmp_path):
        clone = tmp_path / "dep"
        sha = _make_repo(clone)
        pins = self._pins(tmp_path, clone, sha)
        fields = pin_provenance("dep", clone, pins_path=pins)
        assert fields == {"external_git_sha": sha, "external_pin_name": "dep",
                          "external_pin_ok": True}

    def test_drifted_clone_raises(self, tmp_path):
        clone = tmp_path / "dep"
        _make_repo(clone)
        pins = self._pins(tmp_path, clone, "0" * 40)
        with pytest.raises(ValueError, match="re-clone at the pinned commit"):
            pin_provenance("dep", clone, pins_path=pins)

    def test_unpinned_path_recorded_not_asserted(self, tmp_path):
        clone = tmp_path / "elsewhere"
        sha = _make_repo(clone)
        pins = self._pins(tmp_path, tmp_path / "dep", sha)
        fields = pin_provenance("dep", clone, pins_path=pins)
        assert fields["external_pin_ok"] is False
        assert fields["external_git_sha"] == sha
        assert "un-pinned" in fields["external_pin_note"]


class TestRealPins:
    """Assert the real clones (when present) sit at their pinned commits —
    catches silent drift of external inputs on the dev machine."""

    @pytest.mark.parametrize("name", ["deception_detection", "llm_liedetector"])
    def test_real_clone_matches_pin(self, name):
        import yaml

        pin = yaml.safe_load(PINS_PATH.read_text())[name]
        clone = Path(pin["cloned_to"])
        if not (clone / ".git").exists():
            pytest.skip(f"{clone} not cloned on this machine")
        assert pin_provenance(name, clone)["external_pin_ok"] is True
