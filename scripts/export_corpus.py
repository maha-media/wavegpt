#!/usr/bin/env python3
"""
Export enriched training corpus from a knowledge graph.

Reads entities, relationships, and chunks as JSON files, generates
harmonically-ordered augmented training data, and writes .bin files.

Usage:
    wavegpt export --entities data/entities.json \\
                   --relationships data/rels.json \\
                   --chunks data/chunks.json \\
                   --output data/corpus/
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.data_io import write_datafile, tokenize_text, GPT2_EOT
from wavegpt.narratives import (
    generate_entity_context_narratives,
    generate_relationship_chain_narratives,
    generate_cross_source_narratives,
    generate_entity_type_summaries,
    generate_contrastive_narratives,
    generate_counterpoint_narratives,
)


def narratives_to_tokens(narratives: list[dict]) -> list[int]:
    """Convert narrative dicts to token stream."""
    all_tokens = []
    for n in narratives:
        text = n.get("text", "")
        if text.strip():
            all_tokens.append(GPT2_EOT)
            all_tokens.extend(tokenize_text(text))
    return all_tokens


def chunks_to_tokens(chunks: list[dict]) -> list[int]:
    """Convert chunk dicts to token stream."""
    all_tokens = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if text:
            all_tokens.append(GPT2_EOT)
            all_tokens.extend(tokenize_text(text))
    return all_tokens


def entity_narratives_to_tokens(entities: list[dict]) -> list[int]:
    """Simple entity narratives: 'X is a technology.'"""
    all_tokens = []
    for ent in entities:
        name = ent.get("name", "")
        etype = ent.get("type", "concept")
        desc = ent.get("description", "")
        if name:
            text = f"{name} is a {etype}."
            if desc:
                text += f" {desc}"
            all_tokens.append(GPT2_EOT)
            all_tokens.extend(tokenize_text(text))
    return all_tokens


def relationship_narratives_to_tokens(relationships: list[dict], entity_names: dict) -> list[int]:
    """Relationship narratives: 'X enables Y.'"""
    all_tokens = []
    for rel in relationships:
        src = entity_names.get(rel.get("source_entity", ""), "")
        tgt = entity_names.get(rel.get("target_entity", ""), "")
        rtype = rel.get("type", "related to")
        if src and tgt:
            text = f"{src} {rtype} {tgt}."
            desc = rel.get("description", "")
            if desc:
                text += f" {desc}"
            all_tokens.append(GPT2_EOT)
            all_tokens.extend(tokenize_text(text))
    return all_tokens


def main():
    parser = argparse.ArgumentParser(description="Export enriched training corpus")
    parser.add_argument("--entities", required=True, help="Path to entities.json")
    parser.add_argument("--relationships", required=True, help="Path to relationships.json")
    parser.add_argument("--chunks", required=True, help="Path to chunks.json")
    parser.add_argument("--output", default="data/corpus")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--chain-max", type=int, default=5000)
    parser.add_argument("--context-min-chunks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading data...")
    entities = json.loads(Path(args.entities).read_text())
    relationships = json.loads(Path(args.relationships).read_text())
    chunks_list = json.loads(Path(args.chunks).read_text())
    chunks_map = {c["chunk_id"]: c for c in chunks_list}
    entity_names = {e["entity_id"]: e.get("name", "") for e in entities}
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)}")
    print(f"  Chunks: {len(chunks_list)}")

    # Source names (optional)
    source_names = {}
    sources_seen = set()
    for c in chunks_list:
        sid = c.get("source_id", "")
        if sid and sid not in sources_seen:
            sources_seen.add(sid)
            source_names[sid] = sid  # default: use source_id as name

    # ── Tokenize raw text ──
    print("\nTokenizing...")
    raw_tokens = chunks_to_tokens(chunks_list)
    print(f"  Raw text: {len(raw_tokens):,} tokens")

    ent_tokens = entity_narratives_to_tokens(entities)
    print(f"  Entity narratives: {len(ent_tokens):,} tokens")

    rel_tokens = relationship_narratives_to_tokens(relationships, entity_names)
    print(f"  Relationship narratives: {len(rel_tokens):,} tokens")

    # ── Rich narratives ──
    print("\nGenerating rich narratives...")

    ctx_narr = generate_entity_context_narratives(
        entities=entities, chunks=chunks_map,
        min_chunks=args.context_min_chunks, seed=args.seed,
    )
    ctx_tokens = narratives_to_tokens(ctx_narr)
    print(f"  Entity context: {len(ctx_narr)} → {len(ctx_tokens):,} tokens")

    chain_narr = generate_relationship_chain_narratives(
        relationships=relationships, entity_names=entity_names,
        max_narratives=args.chain_max, seed=args.seed,
    )
    chain_tokens = narratives_to_tokens(chain_narr)
    print(f"  Chains: {len(chain_narr)} → {len(chain_tokens):,} tokens")

    cross_narr = generate_cross_source_narratives(
        entities=entities, chunks=chunks_map, source_names=source_names,
        seed=args.seed,
    )
    cross_tokens = narratives_to_tokens(cross_narr)
    print(f"  Cross-source: {len(cross_narr)} → {len(cross_tokens):,} tokens")

    type_narr = generate_entity_type_summaries(entities=entities, seed=args.seed)
    type_tokens = narratives_to_tokens(type_narr)
    print(f"  Type summaries: {len(type_narr)} → {len(type_tokens):,} tokens")

    contrast_narr = generate_contrastive_narratives(
        entities=entities, chunks=chunks_map,
        min_chunks=args.context_min_chunks, max_narratives=5000, seed=args.seed,
    )
    contrast_tokens = narratives_to_tokens(contrast_narr)
    print(f"  Contrastive: {len(contrast_narr)} → {len(contrast_tokens):,} tokens")

    cp_narr = generate_counterpoint_narratives(
        entities=entities, chunks=chunks_map,
        relationships=relationships, entity_names=entity_names,
        min_chunks=args.context_min_chunks, max_narratives=5000, seed=args.seed,
    )
    cp_tokens = narratives_to_tokens(cp_narr)
    print(f"  Counterpoint: {len(cp_narr)} → {len(cp_tokens):,} tokens")

    # ── Assemble (harmonic order: C→G→D→A→♫) ──
    all_aug = (ent_tokens + type_tokens +                  # C: identity
               ctx_tokens +                                 # G: function
               rel_tokens + chain_tokens + cross_tokens +  # D: connection
               contrast_tokens +                            # A: nuance
               cp_tokens)                                   # ♫: counterpoint

    all_full = raw_tokens + all_aug
    total = len(all_full)
    aug_pct = 100 * len(all_aug) / total

    print(f"\n{'='*60}")
    print(f"  Total: {total:,} tokens ({aug_pct:.1f}% augmented)")
    print(f"{'='*60}")

    # ── Write files ──
    print(f"\nWriting to {output_dir}/...")

    write_datafile(str(output_dir / "rai_aug_000.bin"), all_aug)
    print(f"  rai_aug_000.bin:   {len(all_aug):,} tokens")

    # Harmonic layer files
    layer_c = ent_tokens + type_tokens
    layer_g = ctx_tokens
    layer_d = rel_tokens + chain_tokens + cross_tokens
    layer_a = contrast_tokens + cp_tokens

    for name, tokens in [("C", layer_c), ("G", layer_g), ("D", layer_d), ("A", layer_a)]:
        write_datafile(str(output_dir / f"rai_layer_{name}.bin"), tokens)
        print(f"  rai_layer_{name}.bin: {len(tokens):>10,} tokens")

    # Train/val split
    val_count = max(1024, int(total * args.val_fraction))
    train_tokens = all_full[val_count:]
    val_tokens = all_full[:val_count]

    write_datafile(str(output_dir / "rai_train_000.bin"), train_tokens)
    write_datafile(str(output_dir / "rai_val_000.bin"), val_tokens)
    print(f"  rai_train_000.bin: {len(train_tokens):,} tokens")
    print(f"  rai_val_000.bin:   {len(val_tokens):,} tokens")

    # Stats
    stats = {
        "total_tokens": total,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "aug_tokens": len(all_aug),
        "aug_pct": round(aug_pct, 1),
        "raw_tokens": len(raw_tokens),
    }
    (output_dir / "export_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n  Stats: {output_dir}/export_stats.json")


if __name__ == "__main__":
    main()
