import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_dry_run_emits_bin_and_manifest(tmp_path):
    out = tmp_path / "out"
    repo = Path(__file__).resolve().parents[1]
    target = 50000
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "build_forgetting_slice.py"),
            "--tokenizer",
            "google/gemma-4-31b-it",
            "--output-dir",
            str(out),
            "--target-tokens",
            str(target),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, r.stderr

    bin_path = out / "wikitext_val.bin"
    manifest_path = out / "manifest.json"
    assert bin_path.exists(), "wikitext_val.bin missing"
    assert bin_path.stat().st_size > 0, "wikitext_val.bin empty"
    assert manifest_path.exists(), "manifest.json missing"

    manifest = json.loads(manifest_path.read_text())
    for key in (
        "dataset",
        "config",
        "split",
        "tokenizer",
        "n_tokens",
        "n_examples",
        "sha256_bin",
        "target_tokens",
        "overshoot",
    ):
        assert key in manifest, f"manifest missing field: {key}"

    assert manifest["dataset"] == "dry-run"
    assert manifest["target_tokens"] == target
    assert manifest["n_tokens"] >= target
    assert manifest["overshoot"] == manifest["n_tokens"] - target
    assert manifest["overshoot"] >= 0

    expected_sha = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    assert manifest["sha256_bin"] == expected_sha
