"""Inspect digital_rai MongoDB collections, emit field-shape summary JSON."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def inspect_collection(docs, sample=50):
    fields = Counter()
    types = {}
    for d in docs[:sample]:
        for k, v in d.items():
            fields[k] += 1
            t = type(v).__name__
            if k not in types or types[k] == "NoneType":
                types[k] = t
    return {
        "fields": sorted(fields.keys()),
        "types": types,
        "n_sampled": min(sample, len(docs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo-uri", default="mongodb://172.23.208.1:27017")
    ap.add_argument("--db", default="digital_rai")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixture", help="read JSON fixture instead of MongoDB (test mode)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no-op flag; fixture mode is inherently dry (no Mongo IO)")
    args = ap.parse_args()

    if args.fixture:
        collections = json.loads(Path(args.fixture).read_text())
    else:
        from pymongo import MongoClient
        c = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        db = c[args.db]
        collections = {
            name: list(db[name].find().limit(50))
            for name in ["entities", "relationships", "temporal_facts", "chunks"]
        }
        for name, docs in collections.items():
            if not docs:
                raise SystemExit(f"ERROR: collection {name!r} is empty or missing — aborting")

    schema = {name: inspect_collection(docs) for name, docs in collections.items()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(schema, indent=2, default=str))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
