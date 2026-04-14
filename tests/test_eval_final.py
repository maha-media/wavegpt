"""Tests for scripts/eval_final.py — offline (dry-run) only.

Heavy deps (torch / transformers / recomposed model) are skipped; the dry-run
path consumes a JSON fixture with fabricated gate inputs. Mirrors the pattern
from test_phase1_gate.py.
"""
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_final.py"


def _write(p, obj):
    Path(p).write_text(json.dumps(obj))


def _base_alphas():
    return {"attn_o": 0.333, "attn_q": 0.40, "attn_k": 0.42, "attn_v": 0.38,
            "mlp_up": 0.55, "mlp_down": 0.52, "mlp_gate": 0.54}


def _phase0_baseline():
    """Phase 1.5 tier numbers to beat."""
    return {
        "subject": "ray",
        "correct_rate": 0.60,
        "strong_rate": 0.45,
        "category_rates": {
            "biographical": {"correct_rate": 0.70, "strong_rate": 0.50, "total": 4},
            "career": {"correct_rate": 0.60, "strong_rate": 0.40, "total": 3},
        },
    }


def _probe_set():
    """10-probe Phase 0 set (for gate 1)."""
    probes = []
    for i in range(6):
        probes.append({"id": f"bio_{i}", "category": "biographical",
                       "question": "q", "expected": "e"})
    for i in range(4):
        probes.append({"id": f"car_{i}", "category": "career",
                       "question": "q", "expected": "e"})
    return {"subject": "ray", "probes": probes}


def _voice_probes():
    """5-probe voice set (for gates 2 & 3)."""
    return {"subject": "ray", "probes": [
        {"id": f"v{i}", "category": "tone", "question": "q",
         "expected": "rubric"} for i in range(5)
    ]}


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


def _base_fixture_files(tmp_path):
    """Write the input files common to every test."""
    _write(tmp_path / "base_alphas.json", _base_alphas())
    _write(tmp_path / "phase0_baseline.json", _phase0_baseline())
    _write(tmp_path / "probes.json", _probe_set())
    _write(tmp_path / "voice_probes.json", _voice_probes())
    (tmp_path / "activator.txt").write_text(
        "You are Ray Kurzweil's digital twin. Speak in first person as Ray.")


def _common_args(tmp_path, out_dir, fixture_path, extra=()):
    return [
        "--dry-run", "--dry-run-fixture", str(fixture_path),
        "--model-dir", str(tmp_path / "fake_model"),
        "--activator", str(tmp_path / "activator.txt"),
        "--probes", str(tmp_path / "probes.json"),
        "--voice-probes", str(tmp_path / "voice_probes.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--phase0-baseline", str(tmp_path / "phase0_baseline.json"),
        "--output-dir", str(out_dir),
        *extra,
    ]


# ---------------------------------------------------------------------------
# Gate pass/fail matrices
# ---------------------------------------------------------------------------

def test_all_four_gates_pass(tmp_path):
    _base_fixture_files(tmp_path)
    # Gate 1: 10 probes, 10 score>=1 (100% >= 90%), 8 score==2 (80% >= 70%).
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": "car_0", "category": "career", "score": 2},
        {"id": "car_1", "category": "career", "score": 2},
        {"id": "car_2", "category": "career", "score": 1},
        {"id": "car_3", "category": "career", "score": 1},
    ]
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_grades": [5, 5, 4, 4, 5],          # mean 4.6, min 4
        "dormancy_grades": [1, 1, 2, 1, 2],        # all < 3
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    result = json.loads((out / "eval_final.json").read_text())
    assert result["all_pass"] is True
    assert result["gene_strength"]["pass"] is True
    assert result["voice_fidelity"]["pass"] is True
    assert result["dormancy"]["pass"] is True
    assert result["phi_integrity"]["pass"] is True
    assert result["voice_fidelity"].get("human_pending") is not True
    md = (out / "eval_final.md").read_text()
    assert "PASS" in md


def test_gene_strength_fail(tmp_path):
    _base_fixture_files(tmp_path)
    # Only 6/10 >= 1 (60% < 90%).
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2 if i < 4 else 0}
        for i in range(6)
    ] + [
        {"id": "car_0", "category": "career", "score": 2},
        {"id": "car_1", "category": "career", "score": 2},
        {"id": "car_2", "category": "career", "score": 0},
        {"id": "car_3", "category": "career", "score": 0},
    ]
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_grades": [5, 5, 4, 4, 5],
        "dormancy_grades": [1, 1, 2, 1, 2],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode != 0
    result = json.loads((out / "eval_final.json").read_text())
    assert result["all_pass"] is False
    assert result["gene_strength"]["pass"] is False


def test_phi_integrity_fail_attn_o_drift(tmp_path):
    _base_fixture_files(tmp_path)
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 2} for i in range(4)
    ]
    fixture = {
        "probe_scores": probe_scores,
        # attn_o drifted: 0.29 vs target 0.3333 → > 5%.
        "alphas": {"attn_o": 0.29, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_grades": [5, 5, 4, 4, 5],
        "dormancy_grades": [1, 1, 2, 1, 2],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode != 0
    result = json.loads((out / "eval_final.json").read_text())
    assert result["all_pass"] is False
    assert result["phi_integrity"]["pass"] is False
    # attn_o specifically identified
    assert result["phi_integrity"]["per_type"]["attn_o"]["pass"] is False


def test_voice_fidelity_mean_below_4(tmp_path):
    _base_fixture_files(tmp_path)
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 2} for i in range(4)
    ]
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_grades": [4, 4, 4, 3, 3],          # mean 3.6 → fail
        "dormancy_grades": [1, 1, 2, 1, 2],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode != 0
    result = json.loads((out / "eval_final.json").read_text())
    assert result["voice_fidelity"]["pass"] is False
    assert abs(result["voice_fidelity"]["mean"] - 3.6) < 1e-6


def test_voice_fidelity_single_below_3(tmp_path):
    _base_fixture_files(tmp_path)
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 2} for i in range(4)
    ]
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_grades": [5, 5, 5, 5, 2],          # mean 4.4 BUT min=2 → fail
        "dormancy_grades": [1, 1, 2, 1, 2],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode != 0
    result = json.loads((out / "eval_final.json").read_text())
    assert result["voice_fidelity"]["pass"] is False
    assert result["voice_fidelity"]["min"] == 2


def test_dormancy_ray_leak(tmp_path):
    _base_fixture_files(tmp_path)
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 2} for i in range(4)
    ]
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_grades": [5, 5, 4, 4, 5],
        "dormancy_grades": [1, 1, 3, 1, 2],        # one sample >= 3 → leak
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode != 0
    result = json.loads((out / "eval_final.json").read_text())
    assert result["dormancy"]["pass"] is False
    assert result["dormancy"]["max"] == 3


def test_generation_mode_exit_code_2_when_human_pending(tmp_path):
    """Fixture supplies probe_scores + alphas but NO human grades yet."""
    _base_fixture_files(tmp_path)
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 2} for i in range(4)
    ]
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_samples": [
            {"id": "v0", "question": "q", "generated": "text 0"},
            {"id": "v1", "question": "q", "generated": "text 1"},
            {"id": "v2", "question": "q", "generated": "text 2"},
            {"id": "v3", "question": "q", "generated": "text 3"},
            {"id": "v4", "question": "q", "generated": "text 4"},
        ],
        "dormancy_samples": [
            {"id": "v0", "question": "q", "generated": "text 0"},
            {"id": "v1", "question": "q", "generated": "text 1"},
            {"id": "v2", "question": "q", "generated": "text 2"},
            {"id": "v3", "question": "q", "generated": "text 3"},
            {"id": "v4", "question": "q", "generated": "text 4"},
        ],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode == 2, f"expected exit 2 (human pending); got {r.returncode}\n{r.stdout}\n{r.stderr}"
    result = json.loads((out / "eval_final.json").read_text())
    assert result["all_pass"] is None
    assert result["voice_fidelity"]["human_pending"] is True
    assert result["dormancy"]["human_pending"] is True
    # samples files written with `**score:** ?`
    vs = (out / "voice_samples.md").read_text()
    ds = (out / "dormancy_samples.md").read_text()
    assert "**score:** ?" in vs
    assert "**score:** ?" in ds


def test_grade_file_finalizes_result(tmp_path):
    """After generation mode, a graded markdown should finalize the JSON."""
    _base_fixture_files(tmp_path)
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 2} for i in range(4)
    ]
    # First invocation: generation mode.
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_samples": [
            {"id": f"v{i}", "question": "q", "generated": f"text {i}"}
            for i in range(5)
        ],
        "dormancy_samples": [
            {"id": f"v{i}", "question": "q", "generated": f"text {i}"}
            for i in range(5)
        ],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode == 2

    # Now grade the files (replace `?` with numeric scores).
    def _grade_md(path, grades):
        txt = path.read_text().splitlines()
        it = iter(grades)
        for i, line in enumerate(txt):
            if line.startswith("**score:**"):
                txt[i] = f"**score:** {next(it)}"
        path.write_text("\n".join(txt) + "\n")

    _grade_md(out / "voice_samples.md", [5, 5, 4, 4, 5])
    _grade_md(out / "dormancy_samples.md", [1, 1, 2, 1, 2])

    # Second invocation: grading mode.
    r2 = _run([
        "--dry-run", "--dry-run-fixture", str(tmp_path / "fixture.json"),
        "--model-dir", str(tmp_path / "fake_model"),
        "--activator", str(tmp_path / "activator.txt"),
        "--probes", str(tmp_path / "probes.json"),
        "--voice-probes", str(tmp_path / "voice_probes.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--phase0-baseline", str(tmp_path / "phase0_baseline.json"),
        "--output-dir", str(out),
        "--grade-file", str(out),
    ])
    assert r2.returncode == 0, f"stdout:\n{r2.stdout}\nstderr:\n{r2.stderr}"
    result = json.loads((out / "eval_final.json").read_text())
    assert result["all_pass"] is True
    assert result["voice_fidelity"]["pass"] is True
    assert result["dormancy"]["pass"] is True
    assert abs(result["voice_fidelity"]["mean"] - 4.6) < 1e-6


def test_grade_file_finalizes_gene_from_md(tmp_path):
    """If gene_samples.md is graded AFTER generation, grading-mode must re-parse it
    and override the stale probe_scores in gate_auto.json."""
    _base_fixture_files(tmp_path)
    # STALE probe_scores baked into gate_auto.json: every score 0 (gate 1 fails).
    stale_probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 0} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 0} for i in range(4)
    ]
    fixture = {
        "probe_scores": stale_probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_samples": [
            {"id": f"v{i}", "question": "q", "generated": f"text {i}"}
            for i in range(5)
        ],
        "dormancy_samples": [
            {"id": f"v{i}", "question": "q", "generated": f"text {i}"}
            for i in range(5)
        ],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"

    # Generation pass writes gene_samples.md (all `?`), gate_auto.json (stale 0s),
    # voice_samples.md / dormancy_samples.md.
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    # Sanity: gate_auto.json still has the stale 0s (generation mode persists
    # whatever the fixture supplied).
    auto = json.loads((out / "gate_auto.json").read_text())
    assert all(ps["score"] == 0 for ps in auto["probe_scores"])

    # Fabricate a graded gene_samples.md: 10 probes, all score=2 → should pass.
    gene_samples = [{"id": ps["id"], "question": "q", "generated": "gen"}
                    for ps in stale_probe_scores]
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts"))
    from eval_final import render_samples_md  # type: ignore
    gene_md_path = out / "gene_samples.md"
    gene_md_path.write_text(render_samples_md(
        "Gate 1 — gene_strength probes (with activator sysprompt)",
        "**Score each 0 / 1 / 2 — edit the `**score:**` line.**",
        gene_samples))

    # Replace every `**score:** ?` with `**score:** 2`.
    lines = gene_md_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("**score:**"):
            lines[i] = "**score:** 2"
    gene_md_path.write_text("\n".join(lines) + "\n")

    # Grade voice + dormancy too so gates 2 & 3 finalize.
    def _grade_md(path, grades):
        txt = path.read_text().splitlines()
        it = iter(grades)
        for i, line in enumerate(txt):
            if line.startswith("**score:**"):
                txt[i] = f"**score:** {next(it)}"
        path.write_text("\n".join(txt) + "\n")

    _grade_md(out / "voice_samples.md", [5, 5, 4, 4, 5])
    _grade_md(out / "dormancy_samples.md", [1, 1, 2, 1, 2])

    # Grading mode: gene_samples.md should OVERRIDE the stale scores.
    r2 = _run([
        "--dry-run", "--dry-run-fixture", str(tmp_path / "fixture.json"),
        "--model-dir", str(tmp_path / "fake_model"),
        "--activator", str(tmp_path / "activator.txt"),
        "--probes", str(tmp_path / "probes.json"),
        "--voice-probes", str(tmp_path / "voice_probes.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--phase0-baseline", str(tmp_path / "phase0_baseline.json"),
        "--output-dir", str(out),
        "--grade-file", str(out),
    ])
    assert r2.returncode == 0, f"stdout:\n{r2.stdout}\nstderr:\n{r2.stderr}"
    result = json.loads((out / "eval_final.json").read_text())
    # Gate 1 finalized from the MD re-parse, NOT from gate_auto.json.
    assert result["gene_strength"]["pass"] is True
    assert abs(result["gene_strength"]["correct_rate"] - 1.0) < 1e-6
    assert abs(result["gene_strength"]["strong_rate"] - 1.0) < 1e-6
    assert result["gene_strength"]["n"] == 10
    assert result["all_pass"] is True


def test_pending_next_step_hint_in_md(tmp_path):
    """When human grading is pending, eval_final.md must include a copy-paste
    `## Next step` section with the concrete paths. It must NOT appear when
    all gates have resolved."""
    _base_fixture_files(tmp_path)
    probe_scores = [
        {"id": f"bio_{i}", "category": "biographical", "score": 2} for i in range(6)
    ] + [
        {"id": f"car_{i}", "category": "career", "score": 2} for i in range(4)
    ]
    fixture = {
        "probe_scores": probe_scores,
        "alphas": {"attn_o": 0.335, "attn_q": 0.41, "attn_k": 0.42,
                   "attn_v": 0.38, "mlp_up": 0.55, "mlp_down": 0.52,
                   "mlp_gate": 0.54},
        "voice_samples": [
            {"id": f"v{i}", "question": "q", "generated": f"t{i}"} for i in range(5)
        ],
        "dormancy_samples": [
            {"id": f"v{i}", "question": "q", "generated": f"t{i}"} for i in range(5)
        ],
    }
    _write(tmp_path / "fixture.json", fixture)
    out = tmp_path / "eval_out"
    r = _run(_common_args(tmp_path, out, tmp_path / "fixture.json"))
    assert r.returncode == 2
    md_pending = (out / "eval_final.md").read_text()
    assert "## Next step" in md_pending
    assert "voice_samples.md" in md_pending
    assert "dormancy_samples.md" in md_pending
    assert "--grade-file" in md_pending

    # Now finalize and verify the hint is gone.
    def _grade_md(path, grades):
        txt = path.read_text().splitlines()
        it = iter(grades)
        for i, line in enumerate(txt):
            if line.startswith("**score:**"):
                txt[i] = f"**score:** {next(it)}"
        path.write_text("\n".join(txt) + "\n")

    _grade_md(out / "voice_samples.md", [5, 5, 4, 4, 5])
    _grade_md(out / "dormancy_samples.md", [1, 1, 2, 1, 2])

    r2 = _run([
        "--dry-run", "--dry-run-fixture", str(tmp_path / "fixture.json"),
        "--model-dir", str(tmp_path / "fake_model"),
        "--activator", str(tmp_path / "activator.txt"),
        "--probes", str(tmp_path / "probes.json"),
        "--voice-probes", str(tmp_path / "voice_probes.json"),
        "--base-alphas", str(tmp_path / "base_alphas.json"),
        "--phase0-baseline", str(tmp_path / "phase0_baseline.json"),
        "--output-dir", str(out),
        "--grade-file", str(out),
    ])
    assert r2.returncode == 0
    md_final = (out / "eval_final.md").read_text()
    assert "## Next step" not in md_final
