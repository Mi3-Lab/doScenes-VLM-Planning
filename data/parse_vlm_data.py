#!/usr/bin/env python3
"""
VLM (OpenEMMA Without DoScenes) Results Aggregator.

This script traverses the dataset directory, collects metrics from scattered
ADE results files, extracts numeric scene IDs, and aggregates them into a sorted CSV.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# ==== Configuration ====
CONFIG = {
    "BASE_DIR": Path("./VLM_Data/"),
    "OUT_PATH": Path("./vlmresults.csv"),
    "LOG_LEVEL": logging.INFO
}

# ==== Logging Setup ====
logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==== Helper Functions ====

def extract_scene_id(name_str: Any) -> Optional[int]:
    """
    Extracts the numeric part from a scene name string.
    E.g., "scene-1110" -> 1110. Returns None if no number is found.
    """
    if not name_str:
        return None
    
    # Handle cases where input might already be int/float
    if isinstance(name_str, (int, float)):
        return int(name_str)

    s = str(name_str)
    match = re.search(r"(\d+)", s)
    return int(match.group(1)) if match else None


# ==== Core Logic ====

def collect_results(base_dir: Path) -> pd.DataFrame:
    """
    Walks the directory structure and aggregates ADE results.
    """
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        sys.exit(1)

    logger.info(f"Scanning directory: {base_dir} ...")
    
    results: List[Dict[str, Any]] = []
    file_count = 0

    for root, _, files in os.walk(base_dir):
        if "ade_results.jsonl" not in files:
            continue
            
        file_path = Path(root) / "ade_results.jsonl"
        file_count += 1
        
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON in {file_path} line {line_num}")
                        continue

                    name_raw = data.get("name", "")
                    scene_num = extract_scene_id(name_raw)

                    if scene_num is not None:
                        results.append({
                            "name": scene_num,
                            "ade1s": data.get("ade1s"),
                            "ade2s": data.get("ade2s"),
                            "ade3s": data.get("ade3s"),
                            "avgade": data.get("avgade")
                        })
        except OSError as e:
            logger.warning(f"Could not read file {file_path}: {e}")

    logger.info(f"Processed {file_count} 'ade_results.jsonl' files.")
    
    df = pd.DataFrame(results, columns=["name", "ade1s", "ade2s", "ade3s", "avgade"])
    return df


def save_summary(df: pd.DataFrame, output_path: Path):
    """
    Sorts the DataFrame and saves it to CSV.
    """
    if df.empty:
        logger.warning("No results found. CSV will not be created.")
        return

    # Sort by the numeric scene number
    df = df.sort_values(by="name", ascending=True)

    try:
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved sorted summary to {output_path} with {len(df)} entries.")
    except OSError as e:
        logger.error(f"Failed to save CSV to {output_path}: {e}")


# ==== Main Entry Point ====

def main():
    # 1. Collect Data
    df = collect_results(CONFIG["BASE_DIR"])

    # 2. Save Results
    save_summary(df, CONFIG["OUT_PATH"])


if __name__ == "__main__":
    main()