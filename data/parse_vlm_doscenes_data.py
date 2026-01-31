#!/usr/bin/env python3
"""
VLM (OpenEMMA DoScenes) Results Aggregator.

This script traverses the dataset directory, parses metadata and ADE results,
matches them against an annotation CSV, and aggregates the results into a single CSV.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd

# ==== Configuration ====
# Using a config dict or constants allows for easy updates or future argparse integration.
CONFIG = {
    "BASE_DIR": Path("./VLM_doScenes_Data/"),
    "ANN_PATH": Path("./doScenes/annotated_doscenes.csv"),
    "OUT_PATH": Path("./_doscenes_results.csv"),
    "LOG_LEVEL": logging.INFO
}

# ==== Logging Setup ====
logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==== Text & parsing Helpers ====

def normalize_quotes(s: Any) -> Optional[str]:
    """
    Strip exactly one pair of surrounding single/double quotes.
    Returns None if the input is None or NaN.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    
    s = str(s).strip()
    # Check for matching quotes at start and end
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s


def normalize_key(s: Any) -> Optional[str]:
    """
    Create a case-insensitive normalized key with collapsed spaces.
    Useful for fuzzy string matching.
    """
    s = normalize_quotes(s)
    if s is None:
        return None
    # Collapse internal whitespace to single spaces and lowercase
    return re.sub(r"\s+", " ", s).strip().casefold()


def extract_numeric_name(name_value: Any) -> Optional[int]:
    """
    Best-effort extraction of a numeric scene ID.
    E.g., 'scene-1110' -> 1110, '1110' -> 1110.
    """
    if name_value is None:
        return None
    
    if isinstance(name_value, (int, float)):
        try:
            return int(name_value)
        except ValueError:
            return None
            
    s = str(name_value)
    match = re.search(r"(\d+)", s)
    return int(match.group(1)) if match else None


def parse_meta(meta_path: Path) -> Dict[str, Any]:
    """
    Parse _meta.txt for scene number, timestamp, and free-text instruction.
    Handles potential multiline instructions.
    """
    out = {"scene_number": None, "created_at": None, "instruction": None}
    
    if not meta_path or not meta_path.exists():
        return out

    try:
        content = meta_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        logger.warning(f"Failed to read {meta_path}: {e}")
        return out

    lines = content.splitlines()
    instr_lines = []
    in_instr = False

    for line in lines:
        lo = line.lower()
        if not in_instr:
            if lo.startswith("scene_number:"):
                val = line.split(":", 1)[1].strip()
                try:
                    out["scene_number"] = int(val)
                except ValueError:
                    pass
            elif lo.startswith("created_at:"):
                out["created_at"] = line.split(":", 1)[1].strip()
            elif lo.startswith("instruction:"):
                in_instr = True
        else:
            instr_lines.append(line)

    if in_instr:
        instr = "\n".join(instr_lines).strip()
        out["instruction"] = instr if instr else None

    return out


# ==== Core Logic ====

def load_annotation_maps(ann_path: Path) -> Tuple[Dict, Dict]:
    """
    Load the CSV annotations and build lookup maps.
    Returns:
        map_primary: (scene_num, normalized_instruction) -> instruction_type
        map_fallback: normalized_instruction -> instruction_type
    """
    if not ann_path.exists():
        logger.error(f"Annotation file not found at {ann_path}")
        sys.exit(1)

    ann = pd.read_csv(ann_path)
    required_cols = {"Scene Number", "Instruction Type", "Instruction"}
    
    if not required_cols.issubset(ann.columns):
        logger.error(f"CSV missing columns. Required: {required_cols}")
        sys.exit(1)

    # Pre-process columns
    ann["Scene Number int"] = pd.to_numeric(ann["Scene Number"], errors="coerce").astype("Int64")
    ann["Instruction_norm"] = ann["Instruction"].apply(normalize_key)

    map_primary = {}
    map_fallback = {}

    for _, row in ann.iterrows():
        sc = row["Scene Number int"]
        itype = row["Instruction Type"]
        ikey = row["Instruction_norm"]

        if pd.isna(sc) or ikey is None:
            continue
        
        # Primary: Exact match on Scene ID AND Instruction text
        map_primary[(int(sc), ikey)] = itype
        
        # Fallback: Match on Instruction text only (first wins)
        map_fallback.setdefault(ikey, itype)

    logger.info(f"Loaded {len(ann)} annotations.")
    return map_primary, map_fallback


def process_dataset(base_dir: Path, map_primary: Dict, map_fallback: Dict) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Walks the directory, parses files, matches annotations, and builds the results DataFrame.
    """
    rows = []
    stats = {"matched_primary": 0, "matched_fallback": 0, "unmatched": 0}

    # Walk directory
    for root, _, files in os.walk(base_dir):
        if "ade_results.jsonl" not in files:
            continue

        root_path = Path(root)
        ade_path = root_path / "ade_results.jsonl"
        
        # Get metadata
        meta = parse_meta(root_path / "_meta.txt")
        scene_from_meta = meta.get("scene_number")
        instr_from_meta = meta.get("instruction")
        instr_key = normalize_key(instr_from_meta)

        # Process the JSONL file
        with ade_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s: 
                    continue
                
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue

                # Determine Name and Scene Number
                name_raw = rec.get("name")
                scene_num = extract_numeric_name(name_raw)
                
                # If JSON name isn't numeric, fallback to meta.txt scene number
                if scene_num is None and scene_from_meta is not None:
                    scene_num = scene_from_meta

                # Match Instruction Type
                itype = None
                if instr_key is not None:
                    if scene_num is not None and (scene_num, instr_key) in map_primary:
                        itype = map_primary[(scene_num, instr_key)]
                        stats["matched_primary"] += 1
                    elif instr_key in map_fallback:
                        itype = map_fallback[instr_key]
                        stats["matched_fallback"] += 1
                    else:
                        stats["unmatched"] += 1
                else:
                    stats["unmatched"] += 1

                rows.append({
                    "name": scene_num if scene_num is not None else name_raw,
                    "ade1s": rec.get("ade1s"),
                    "ade2s": rec.get("ade2s"),
                    "ade3s": rec.get("ade3s"),
                    "ade": rec.get("avgade"),
                    "instruction": instr_from_meta,
                    "instruction_type": itype
                })

    df = pd.DataFrame(rows, columns=[
        "name", "ade1s", "ade2s", "ade3s", "ade", "instruction", "instruction_type"
    ])
    
    return df, stats


def save_results(df: pd.DataFrame, path: Path):
    """Sorts and saves the DataFrame to CSV."""
    if df.empty:
        logger.warning("No data found to save.")
        return

    # Sort by numeric scene ID if possible
    sort_key = df["name"].apply(extract_numeric_name)
    df = df.assign(_k=sort_key).sort_values(by="_k", kind="mergesort").drop(columns=["_k"])
    
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Successfully wrote {len(df)} rows to {path}")
    except OSError as e:
        logger.error(f"Failed to save file: {e}")


# ==== Main Entry Point ====

def main():
    logger.info("Starting OpenEMMA processing...")

    # 1. Load Annotations
    map_primary, map_fallback = load_annotation_maps(CONFIG["ANN_PATH"])

    # 2. Process Dataset
    df, stats = process_dataset(CONFIG["BASE_DIR"], map_primary, map_fallback)

    # 3. Log Statistics
    logger.info(f"Matching Results:")
    logger.info(f"  Exact Match (Scene+Text): {stats['matched_primary']}")
    logger.info(f"  Text Only Match:          {stats['matched_fallback']}")
    logger.info(f"  Unmatched:                {stats['unmatched']}")

    # 4. Save
    save_results(df, CONFIG["OUT_PATH"])


if __name__ == "__main__":
    main()