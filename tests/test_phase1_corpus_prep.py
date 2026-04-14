import json
import subprocess
import sys
from pathlib import Path


def test_chunking_and_oversample(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "src.txt").write_text(" ".join(["word"] * 10000))
    tier = {"gap_categories": ["idiom"]}
    (tmp_path / "tier.json").write_text(json.dumps(tier))
    probes = {
        "probes": [
            {"id": "x", "category": "idiom", "question": "q", "expected": "word"}
        ]
    }
    (tmp_path / "probes.json").write_text(json.dumps(probes))
    out = tmp_path / "out"
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "phase1_corpus_prep.py"),
            "--raw-dir",
            str(raw),
            "--tier-json",
            str(tmp_path / "tier.json"),
            "--probes",
            str(tmp_path / "probes.json"),
            "--tokenizer",
            "google/gemma-4-31b-it",
            "--window",
            "2048",
            "--overlap",
            "128",
            "--output-dir",
            str(out),
            "--oversample-factor",
            "3",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_chunks"] > 0
    assert manifest["oversampled_categories"] == ["idiom"]
    assert manifest["window"] == 2048
    assert manifest["overlap"] == 128
    assert "n_tokens" in manifest
    assert "source_files" in manifest
    assert any("src.txt" in s for s in manifest["source_files"])
