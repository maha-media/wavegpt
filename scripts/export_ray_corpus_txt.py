#!/usr/bin/env python3
"""Export Ray-authored + Ray-biographical chunks from the digital_rai MongoDB
to plain-text files under ``--output-dir``.

Downstream consumer: ``scripts/phase1_corpus_prep.py`` reads a directory of
``.txt``/``.md`` and handles tokenization, windowing, and gap-category
oversampling. This script's only job is to pull clean text from MongoDB
(``digital_rai``) with an explicit source allowlist, so the corpus
composition is auditable.

Allowlist is split into three groups so the emitted files reveal
provenance at a glance:
  books/       — Ray-authored books
  voice/       — Ray's speeches, essays, interviews, op-eds
  bio_facts/   — third-party biographical material about Ray (Wikipedia, bio blurb)

Usage:
    python3 scripts/export_ray_corpus_txt.py \\
        --mongo-uri mongodb://172.23.208.1:27017 \\
        --output-dir data/ray/raw
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


BOOKS = [
    "The Age of Intelligent Machines",
    "The Age of Spiritual Machines",
    "How to Create a Mind",
    "The Singularity Is Near",
    "The Singularity Is Nearer",
    "Transcend",
    "Fantastic Voyage",
    "A Chronicle of Ideas",
    "Where Does the Time Go",
    "Danielle Chronicles of a Superheroine",
]

VOICE = [
    "TED Talk Official Written Script",
    "TED Talk 2024 Transcript",
    "Next Big Idea Club Bookbite",
    "Next Big Idea Book Club Speech Script",
    "Lex Fridman Podcast #321 Transcript",
    "SERICEO Speech Script",
    "Robert A. Muh Award Speech Script",
    "National Federation of the Blind Speech 2024",
    "National Federation of the Blind Speech 2025",
    "Washington Post Interview Prep Notes",
    "2025 Writing Compilation",
    "Google Statement — Leaving Google",
    "Academy for Teachers Speech Script",
    "AI Stories",
    "27 Quotes by Ray Kurzweil",
    "RAY-RAI Key Phrases and Concepts",
    "The Law of Accelerating Returns — Economics Essay",
    "Ray Kurzweil Reader",
    "Ray Kurzweil Interview — Academy of Achievement",
    "TIME — 2045: The Year Man Becomes Immortal",
    "TIME — At TIME 100 Impact Dinner",
    "TIME 100 Impact Dinner Panel Transcript",
    "TIME — Inside the Kurzweil SXSW Keynote",
    "TIME — Can We Talk",
    "TIME — Will My PC Be Smarter Than I Am",
    "TIME — Interview with Ray Kurzweil 2010",
    "TIME — Future Proofing",
    "TIME — An Interview With Ray Kurzweil",
    "TIME — Dont Fear Artificial Intelligence",
    "TIME — Robots Will Demand Rights",
    "TIME — Inside the Mind of Futurist Ray Kurzweil",
    "TIME — The Promise and Peril of AI",
]

BIO_FACTS = [
    "Wikipedia — Ray Kurzweil Biography",
    "Kurzweil Bio Short",
]


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mongo-uri", default="mongodb://172.23.208.1:27017")
    p.add_argument("--db", default="digital_rai")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="skip file writes, print what would be emitted")
    return p.parse_args()


def export_source(db, source_name: str, out_path: Path, dry_run: bool) -> dict:
    source = db.sources.find_one({"source_name": source_name})
    if source is None:
        return {"source_name": source_name, "status": "missing",
                "chunks": 0, "chars": 0, "path": None}
    sid = source["source_id"]
    chunks = list(db.chunks.find(
        {"source_id": sid},
        {"text": 1, "context_prefix": 1, "chunk_index": 1, "chunk_type": 1},
    ).sort("chunk_index", 1))
    parts = []
    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
    body = "\n\n".join(parts)
    char_count = len(body)
    if not dry_run and body:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
    return {
        "source_name": source_name,
        "status": "ok" if body else "empty",
        "chunks": len(chunks),
        "chars": char_count,
        "path": str(out_path),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    try:
        from pymongo import MongoClient
    except ImportError:
        sys.exit("pymongo required: pip install pymongo")

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[args.db]

    groups = [("books", BOOKS), ("voice", VOICE), ("bio_facts", BIO_FACTS)]
    manifest = {"mongo_uri": args.mongo_uri, "db": args.db, "groups": {}}

    total_chars = 0
    total_chunks = 0
    for group_name, names in groups:
        group_dir = out_dir / group_name
        results = []
        for nm in names:
            out_path = group_dir / f"{safe_filename(nm)}.txt"
            r = export_source(db, nm, out_path, args.dry_run)
            results.append(r)
            total_chars += r["chars"]
            total_chunks += r["chunks"]
            status_tag = "MISSING" if r["status"] == "missing" else (
                "empty" if r["status"] == "empty" else f"{r['chars']:>9} chars")
            print(f"  [{group_name:>9}] {status_tag:>15}  {nm}", flush=True)
        manifest["groups"][group_name] = results

    manifest["total_chars"] = total_chars
    manifest["total_chunks"] = total_chunks
    print(f"\n[done] {total_chunks} chunks, {total_chars:,} chars across "
          f"{sum(len(g[1]) for g in groups)} sources", flush=True)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2))
        print(f"       wrote {out_dir/'export_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
