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
            "--lr", "1e-4",
            "--attn-o-lr-mult", "0.1",
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
    for k in ("step", "train_loss", "val_ppl", "forget_ppl", "lr_other", "lr_attn_o"):
        assert k in last, f"missing key {k} in log entry: {last}"
    # Tier LR must actually be smaller than the base LR. That's the whole point.
    assert last["lr_attn_o"] < last["lr_other"], \
        f"attn_o LR ({last['lr_attn_o']}) should be smaller than other LR ({last['lr_other']})"
    assert last["lr_attn_o"] == last["lr_other"] * 0.1

    # Startup log should announce the two parameter groups — confirms attn_o
    # params were actually found and routed (not silently zero). This guards
    # against a rename that breaks _is_attn_o_param's substring match.
    assert "attn_o=" in r.stdout, \
        f"expected 'attn_o=' in startup stdout; got:\n{r.stdout}"
    # Count must be nonzero — the SmokeModel has 2 blocks × o_proj (U,S,V) so
    # expect >0 params routed to the attn_o group.
    import re
    m = re.search(r"attn_o=([\d,]+) params", r.stdout)
    assert m, f"could not find attn_o param count in stdout:\n{r.stdout}"
    assert int(m.group(1).replace(",", "")) > 0, \
        "attn_o param group is empty — substring match broke"


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
    assert log[0]["train_loss"] is None
