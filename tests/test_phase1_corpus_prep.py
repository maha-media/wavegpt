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
    raw_text = " ".join(
        ["filler"] * 500
        + ["fredric", "austrian", "pianist", "concert", "conductor"]
        + ["filler"] * 500
    )
    tier = {"gap_categories": ["biographical"]}
    probes = {
        "probes": [
            {
                "id": "x",
                "category": "biographical",
                "question": "q",
                "expected": "Fredric Kurzweil, an Austrian-Jewish concert pianist and conductor.",
            }
        ]
    }
    r, out = _run_dry(tmp_path, tier, probes, raw_text=raw_text)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_chunks"] > 0
    assert manifest["oversampled_categories"] == ["biographical"]
    assert manifest["window"] == 2048
    assert manifest["overlap"] == 128
    assert "n_tokens" in manifest
    assert "source_files" in manifest
    assert any("src.txt" in s for s in manifest["source_files"])


def test_oversample_excludes_non_matching_category(tmp_path):
    # Two gap categories. "biographical" probe has ≥2 distinctive tokens
    # ("fredric", "pianist") present in the corpus → matches. "obscure"
    # expects words that aren't there → no match.
    raw_text = " ".join(
        ["filler"] * 500 + ["fredric", "austrian", "pianist"] + ["filler"] * 500
    )
    tier = {"gap_categories": ["biographical", "obscure"]}
    probes = {
        "probes": [
            {
                "id": "x",
                "category": "biographical",
                "question": "q",
                "expected": "Fredric Kurzweil was a concert pianist.",
            },
            {
                "id": "y",
                "category": "obscure",
                "question": "q",
                "expected": "quetzalcoatl tenochtitlan",
            },
        ]
    }
    r, out = _run_dry(tmp_path, tier, probes, raw_text=raw_text)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["oversampled_categories"] == ["biographical"]


def test_subject_name_alone_does_not_match(tmp_path):
    # A probe whose only candidate tokens are the subject name ("Kurzweil")
    # must NOT match every chunk of a Ray corpus — that was the bug that
    # produced a 16x whole-corpus duplicate instead of a targeted oversample.
    raw_text = " ".join(["kurzweil"] + ["filler"] * 3000)
    tier = {"gap_categories": ["biographical"]}
    probes = {
        "probes": [
            {
                "id": "x",
                "category": "biographical",
                "question": "q",
                "expected": "Ray Kurzweil.",
            }
        ]
    }
    r, out = _run_dry(tmp_path, tier, probes, raw_text=raw_text)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["oversampled_categories"] == []
