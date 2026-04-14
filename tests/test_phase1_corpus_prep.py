import json
import subprocess
import sys
from pathlib import Path


def _run_dry(tmp_path, tier, probes, raw_text=None):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "src.txt").write_text(raw_text if raw_text is not None else " ".join(["word"] * 10000))
    (tmp_path / "tier.json").write_text(json.dumps(tier))
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
    return r, out


def test_dry_run_manifest_and_oversample_detection(tmp_path):
    tier = {"gap_categories": ["idiom"]}
    probes = {
        "probes": [
            {"id": "x", "category": "idiom", "question": "q", "expected": "word"}
        ]
    }
    r, out = _run_dry(tmp_path, tier, probes)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_chunks"] > 0
    assert manifest["oversampled_categories"] == ["idiom"]
    assert manifest["window"] == 2048
    assert manifest["overlap"] == 128
    assert "n_tokens" in manifest
    assert "source_files" in manifest
    assert any("src.txt" in s for s in manifest["source_files"])


def test_oversample_excludes_non_matching_category(tmp_path):
    # Two gap categories: "idiom" (probe expects "word", which IS in the corpus)
    # and "obscure" (probe expects "quetzalcoatl", which is NOT in the corpus).
    # Only "idiom" should be reported as oversampled.
    tier = {"gap_categories": ["idiom", "obscure"]}
    probes = {
        "probes": [
            {"id": "x", "category": "idiom", "question": "q", "expected": "word"},
            {
                "id": "y",
                "category": "obscure",
                "question": "q",
                "expected": "quetzalcoatl",
            },
        ]
    }
    r, out = _run_dry(tmp_path, tier, probes)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["oversampled_categories"] == ["idiom"]
