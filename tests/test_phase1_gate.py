"""Tests for scripts/phase1_gate.py — offline (dry-run) only.

Heavy deps (torch / transformers / 137GB checkpoint) are deliberately skipped;
the dry-run path consumes a JSON fixture with fabricated gate inputs.
"""
import json
import subprocess
import sys
from pathlib import Path


def _write(p, obj):
    Path(p).write_text(json.dumps(obj))


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "phase1_gate.py"


def _baseline():
    return {
        "subject": "ray",
        "tier": "B",
        "correct_rate": 0.5,
        "gap_categories": ["idiom"],
        "category_rates": {
            "bio":   {"correct_rate": 1.0, "strong_rate": 1.0, "total": 2},
            "idiom": {"correct_rate": 0.25, "strong_rate": 0.0, "total": 4},
        },
    }


def _probes():
    return {"subject": "ray", "probes": [
        {"id": "b1", "category": "bio",   "question": "q", "expected": "e"},
        {"id": "b2", "category": "bio",   "question": "q", "expected": "e"},
        {"id": "i1", "category": "idiom", "question": "q", "expected": "e"},
    ]}


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


def test_all_gates_pass(tmp_path):
    _write(tmp_path / "baseline.json", _baseline())
    _write(tmp_path / "base_alphas.json", {"attn_o": 0.333, "attn_q": 0.40})
    _write(tmp_path / "probes.json", _probes())
    fixture = {
        "probe_scores": [
            {"id": "b1", "category": "bio",   "score": 2},
            {"id": "b2", "category": "bio",   "score": 2},
            {"id": "i1", "category": "idiom", "score": 2},
        ],
        "alphas": {"attn_o": 0.335, "attn_q": 0.41},
        "forget_ppl": 10.5,
    }
    _write(tmp_path / "fixture.json", fixture)

    out = tmp_path / "gate_out"
    r = _run([
        "--dry-run", "--dry-run-fixture", str(tmp_path / "fixture.json"),
        "--checkpoint", str(tmp_path / "fake.pt"),
        "--probe-snapshot", str(tmp_path / "probe_snapshot.json"),  # unused in dry-run
        "--baseline-tier", str(tmp_path / "baseline.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--forget-bin", str(tmp_path / "fake.bin"),
        "--forget-ppl-base", "10.0",
        "--model-config", "stub",
        "--output-dir", str(out),
    ])
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    result = json.loads((out / "phase1_gate.json").read_text())
    assert result["all_pass"] is True
    assert result["gene_recall"]["pass"] is True
    assert result["phi_integrity"]["pass"] is True
    assert result["general_retention"]["pass"] is True
    assert not (out / "knob_suggestion.md").exists()
    md = (out / "phase1_gate.md").read_text()
    assert "gene_recall" in md.lower() or "biographical" in md.lower()


def test_phi_fail_emits_knob(tmp_path):
    _write(tmp_path / "baseline.json", _baseline())
    _write(tmp_path / "base_alphas.json", {"attn_o": 0.333, "attn_q": 0.40})
    _write(tmp_path / "probes.json", _probes())
    fixture = {
        "probe_scores": [
            {"id": "b1", "category": "bio",   "score": 2},
            {"id": "b2", "category": "bio",   "score": 2},
            {"id": "i1", "category": "idiom", "score": 2},
        ],
        "alphas": {"attn_o": 0.29, "attn_q": 0.41},  # attn_o drifted
        "forget_ppl": 10.5,
    }
    _write(tmp_path / "fixture.json", fixture)

    out = tmp_path / "gate_out"
    r = _run([
        "--dry-run", "--dry-run-fixture", str(tmp_path / "fixture.json"),
        "--checkpoint", str(tmp_path / "fake.pt"),
        "--probe-snapshot", str(tmp_path / "probe_snapshot.json"),
        "--baseline-tier", str(tmp_path / "baseline.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--forget-bin", str(tmp_path / "fake.bin"),
        "--forget-ppl-base", "10.0",
        "--model-config", "stub",
        "--output-dir", str(out),
    ])
    assert r.returncode != 0, f"expected nonzero exit; stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    result = json.loads((out / "phase1_gate.json").read_text())
    assert result["all_pass"] is False
    assert result["phi_integrity"]["pass"] is False
    assert result["gene_recall"]["pass"] is True
    assert result["general_retention"]["pass"] is True
    knob = out / "knob_suggestion.md"
    assert knob.exists(), "knob_suggestion.md not emitted on phi fail"
    knob_txt = knob.read_text()
    assert "attn_o" in knob_txt or "attn-o" in knob_txt
    # no promotion file
    assert not (out / "phase1_verified.pt").exists()


def test_retention_fail_emits_knob(tmp_path):
    _write(tmp_path / "baseline.json", _baseline())
    _write(tmp_path / "base_alphas.json", {"attn_o": 0.333, "attn_q": 0.40})
    _write(tmp_path / "probes.json", _probes())
    fixture = {
        "probe_scores": [
            {"id": "b1", "category": "bio",   "score": 2},
            {"id": "b2", "category": "bio",   "score": 2},
            {"id": "i1", "category": "idiom", "score": 2},
        ],
        "alphas": {"attn_o": 0.335, "attn_q": 0.41},
        "forget_ppl": 12.0,  # 12.0 / 10.0 = 1.20 > 1.10
    }
    _write(tmp_path / "fixture.json", fixture)

    out = tmp_path / "gate_out"
    r = _run([
        "--dry-run", "--dry-run-fixture", str(tmp_path / "fixture.json"),
        "--checkpoint", str(tmp_path / "fake.pt"),
        "--probe-snapshot", str(tmp_path / "probe_snapshot.json"),
        "--baseline-tier", str(tmp_path / "baseline.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--forget-bin", str(tmp_path / "fake.bin"),
        "--forget-ppl-base", "10.0",
        "--model-config", "stub",
        "--output-dir", str(out),
    ])
    assert r.returncode != 0
    result = json.loads((out / "phase1_gate.json").read_text())
    assert result["general_retention"]["pass"] is False
    assert result["all_pass"] is False
    knob = (out / "knob_suggestion.md").read_text()
    assert "harmonic" in knob.lower() or "lr" in knob.lower()


def test_recall_fail_gap_category(tmp_path):
    # baseline: idiom correct_rate 0.25; requires 0.50 (2x).
    # Phase-1 probe returns idiom rate 0.33 — fails 2x test.
    baseline = _baseline()
    baseline["category_rates"]["idiom"]["correct_rate"] = 0.25
    _write(tmp_path / "baseline.json", baseline)
    _write(tmp_path / "base_alphas.json", {"attn_o": 0.333, "attn_q": 0.40})
    probes = {"subject": "ray", "probes": [
        {"id": "b1", "category": "bio", "question": "q", "expected": "e"},
        {"id": "b2", "category": "bio", "question": "q", "expected": "e"},
        {"id": "i1", "category": "idiom", "question": "q", "expected": "e"},
        {"id": "i2", "category": "idiom", "question": "q", "expected": "e"},
        {"id": "i3", "category": "idiom", "question": "q", "expected": "e"},
    ]}
    _write(tmp_path / "probes.json", probes)
    fixture = {
        "probe_scores": [
            {"id": "b1", "category": "bio", "score": 2},
            {"id": "b2", "category": "bio", "score": 2},
            {"id": "i1", "category": "idiom", "score": 1},
            {"id": "i2", "category": "idiom", "score": 0},
            {"id": "i3", "category": "idiom", "score": 0},
        ],
        "alphas": {"attn_o": 0.335, "attn_q": 0.41},
        "forget_ppl": 10.5,
    }
    _write(tmp_path / "fixture.json", fixture)

    out = tmp_path / "gate_out"
    r = _run([
        "--dry-run", "--dry-run-fixture", str(tmp_path / "fixture.json"),
        "--checkpoint", str(tmp_path / "fake.pt"),
        "--probe-snapshot", str(tmp_path / "probe_snapshot.json"),
        "--baseline-tier", str(tmp_path / "baseline.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--forget-bin", str(tmp_path / "fake.bin"),
        "--forget-ppl-base", "10.0",
        "--model-config", "stub",
        "--output-dir", str(out),
    ])
    assert r.returncode != 0
    result = json.loads((out / "phase1_gate.json").read_text())
    assert result["gene_recall"]["pass"] is False
    # idiom sub-check should fail
    assert result["gene_recall"]["per_category"]["idiom"]["pass"] is False
    # idiom rate should be 0.333... and required 0.5
    assert abs(result["gene_recall"]["per_category"]["idiom"]["required"] - 0.5) < 1e-6
