"""Smoke test for scripts/phase1_cpt.py — toy model, 2 training steps, CPU."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from wavegpt.data_io import write_datafile


def test_cpt_one_step(tmp_path):
    tokens = np.arange(1024, dtype=np.uint16).tolist()
    write_datafile(str(tmp_path / "train.bin"), tokens)
    write_datafile(str(tmp_path / "val.bin"), tokens)
    write_datafile(str(tmp_path / "forget.bin"), tokens)

    out_dir = tmp_path / "out"
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "phase1_cpt.py"),
            "--smoke",
            "--train-bin", str(tmp_path / "train.bin"),
            "--val-bin", str(tmp_path / "val.bin"),
            "--forget-bin", str(tmp_path / "forget.bin"),
            "--output-dir", str(out_dir),
            "--max-steps", "2",
            "--eval-every", "1",
            "--harmonic-lambda", "0.01",
            "--attn-o-weight", "10.0",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    log_path = out_dir / "training_log.json"
    assert log_path.exists(), "training_log.json missing"
    log = json.loads(log_path.read_text())
    assert isinstance(log, list)
    assert len(log) >= 2, f"expected >= 2 log entries, got {len(log)}"
    last = log[-1]
    assert "harmonic_loss" in last
    assert "forget_ppl" in last
    assert "val_ppl" in last
    assert "train_loss" in last
    assert "step" in last


def test_cpt_eval_only(tmp_path):
    tokens = np.arange(1024, dtype=np.uint16).tolist()
    write_datafile(str(tmp_path / "train.bin"), tokens)
    write_datafile(str(tmp_path / "val.bin"), tokens)
    write_datafile(str(tmp_path / "forget.bin"), tokens)

    out_dir = tmp_path / "out_eval"
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "phase1_cpt.py"),
            "--smoke",
            "--eval-only",
            "--train-bin", str(tmp_path / "train.bin"),
            "--val-bin", str(tmp_path / "val.bin"),
            "--forget-bin", str(tmp_path / "forget.bin"),
            "--output-dir", str(out_dir),
            "--max-steps", "1",
            "--harmonic-lambda", "0.0",
            "--attn-o-weight", "10.0",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    log = json.loads((out_dir / "training_log.json").read_text())
    assert len(log) == 1
    assert "forget_ppl" in log[0]
    assert "val_ppl" in log[0]
