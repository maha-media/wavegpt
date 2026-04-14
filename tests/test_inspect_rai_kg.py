"""Test for scripts/inspect_rai_kg.py — runs in dry-run mode against a fixture."""
import json
import subprocess
import sys
from pathlib import Path


def test_schema_fields_required(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fixture = repo / "tests" / "fixtures" / "fake_mongo.json"
    out = tmp_path / "kg_schema.json"

    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "inspect_rai_kg.py"),
            "--dry-run",
            "--fixture", str(fixture),
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert out.exists(), "kg_schema.json missing"

    schema = json.loads(out.read_text())
    for coll in ("entities", "relationships", "temporal_facts", "chunks"):
        assert coll in schema, f"missing collection {coll}"
        assert "fields" in schema[coll], f"{coll}.fields missing"
        fields = schema[coll]["fields"]
        assert isinstance(fields, list), f"{coll}.fields not a list"
        assert len(fields) >= 3, \
            f"{coll}.fields has <3 entries: {fields}"
