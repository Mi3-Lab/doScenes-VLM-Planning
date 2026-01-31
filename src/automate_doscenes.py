import pandas as pd
import subprocess
import shlex
import os
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime

# === Paths ===
csv_path = "sorted_combined_annotations.csv"
results_root = Path("llava-v1.6-mistral-7b_results/openemma")
manifest_path = results_root / "results_manifest.csv"

# === Ensure results directory exists ===
results_root.mkdir(parents=True, exist_ok=True)

# === Load and clean CSV ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=["Instruction"])
df = df[df["Instruction"].astype(str).str.strip() != ""]
df = df.sort_values(by="Scene Number", ascending=True).reset_index(drop=True)

timestamp_pat = re.compile(r"^\d{8}-\d{6}$")

def slugify_instruction(instr: str, max_len: int = 48) -> str:
    """Make a short, filesystem-safe slug + an 8-char hash for uniqueness."""
    base = (
        instr.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("'", "")
        .replace('"', "")
        .replace(":", "_")
        .replace("|", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace("<", "_")
        .replace(">", "_")
    )
    base = base[:max_len].strip("_")
    h = hashlib.sha1(instr.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}" if base else h

def list_dirs(path: Path) -> set[str]:
    return {p.name for p in path.iterdir() if p.is_dir()}

def newest_timestamp_folder(path: Path) -> Path | None:
    """Pick the newest directory that looks like YYYYMMDD-HHMMSS."""
    candidates = [p for p in path.iterdir() if p.is_dir() and timestamp_pat.match(p.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

# === Process each scene ===
for _, row in df.iterrows():
    scene_num = int(row["Scene Number"])
    instruction = str(row["Instruction"]).strip()

    print(f"\n========= Running Scene {scene_num} =========")
    print(f"Prompt: {instruction}\n")

    # Snapshot existing folders before run
    before = list_dirs(results_root)

    # === Build your command ===
    cmd = (
        f'python main.py '
        f'--model-path llava-v1.6-mistral-7b '
        f'--method openemma '
        f'--dataroot /data/nuscenes '
        f'--doScenes "{instruction}"'
    )

    # === Run main.py and send scene number via stdin ===
    process = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    # Send scene number followed by two newlines (as you had)
    out, _ = process.communicate(input=f"{scene_num}\n\n")

    # Ensure filesystem has flushed; communicate() already waits for process exit.
    time.sleep(1)

    # Detect new directories created by this run
    after = list_dirs(results_root)
    delta = list(after - before)

    # Decide which directory is the real result folder
    new_folder_path: Path | None = None
    if len(delta) == 1:
        new_folder_path = results_root / delta[0]
    elif len(delta) > 1:
        # Prefer a timestamp-looking, newest folder
        ts_candidates = [d for d in delta if timestamp_pat.match(d)]
        if ts_candidates:
            newest = max(ts_candidates, key=lambda name: (results_root / name).stat().st_mtime)
            new_folder_path = results_root / newest
        else:
            # Fallback: newest by mtime among all new dirs
            new_folder_path = max(
                (results_root / d for d in delta),
                key=lambda p: p.stat().st_mtime
            )
    else:
        # No obvious delta—fallback to the newest timestamp folder overall
        new_folder_path = newest_timestamp_folder(results_root)

    if new_folder_path is None or not new_folder_path.exists():
        print(f"❌ No new folder detected for Scene {scene_num}")
        print(out)  # surface logs to help debugging
        continue

    # Build a descriptive, collision-resistant name
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify_instruction(instruction)
    final_name = f"scene_{scene_num:04d}_{slug}_{ts}"
    final_path = results_root / final_name

    # Rename
    try:
        new_folder_path.rename(final_path)
    except OSError:
        # If something already exists with same name, append a counter
        i = 2
        while True:
            alt = results_root / f"{final_name}__{i}"
            if not alt.exists():
                new_folder_path.rename(alt)
                final_path = alt
                break
            i += 1

    # Write a small meta file with full instruction for quick reference
    meta = final_path / "_meta.txt"
    meta.write_text(
        f"scene_number: {scene_num}\n"
        f"created_at: {ts}\n"
        f"instruction:\n{instruction}\n",
        encoding="utf-8"
    )

    # Append to manifest (create if missing)
    # Columns: created_at, scene_number, folder, instruction
    import csv
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["created_at", "scene_number", "folder", "instruction"])
        w.writerow([ts, f"{scene_num:04d}", final_path.name, instruction])

    print(f"✅ Renamed result folder → {final_path}")
    print(f"✅ Logged to manifest     → {manifest_path}")
    print(f"✅ Finished Scene {scene_num}")
