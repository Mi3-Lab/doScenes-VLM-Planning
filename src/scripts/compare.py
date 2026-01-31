import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Hardcoded path of OpenEMMA results
base_path = "/home/mi3/parthib_openemma_to_nuplan/OpenEMMA/llava_results/openemma"

# Save to compare folder
compare_folder = os.path.join(base_path, "compare")
os.makedirs(compare_folder, exist_ok=True)

def get_scene_folder(scene_number, is_doScenes):
    scene_str = str(scene_number).zfill(3)
    suffix = "_doScenes" if is_doScenes else "_no_doScenes"
    return os.path.join(base_path, f"scene_{scene_str}{suffix}")

def plot_trajectories_side_by_side(doScemes_folder_path, no_doScenes_folder_path, scene_number):
    """
    Loads existing trajectory plots from both folders and merges them side by side.
    Assumes same image size.
    """

    doScenes_traj_plot_path = os.path.join(doScemes_folder_path, "[PLOT]_trajectories_plot.png")
    no_doScenes_traj_plot_path = os.path.join(no_doScenes_folder_path, "[PLOT]_trajectories_plot.png")

    if not os.path.exists(doScenes_traj_plot_path) or not os.path.exists(no_doScenes_traj_plot_path):
        print("[PLOT] One or both plot images are missing.")
        return

    doScenes_img = Image.open(doScenes_traj_plot_path)
    no_doScenes_img = Image.open(no_doScenes_traj_plot_path)

    # Assume same size
    width, height = no_doScenes_img.size
    combined_width = width * 2

    merged_img = Image.new("RGB", (combined_width, height))
    merged_img.paste(doScenes_img, (0, 0))         # Left: no doScenes
    merged_img.paste(no_doScenes_img, (width, 0))     # Right: doScenes

    save_path = os.path.join(compare_folder, f"[PLOT]_scene_{scene_number:03d}_comparison.png")
    merged_img.save(save_path)
    print(f"[PLOT] Saved side-by-side plot")

def read_avgade(folder):
    """
    Reads the avgADE from the ade_results.jsonl file in the specified folder.
    """

    ade_path = os.path.join(folder, "ade_results.jsonl")

    with open(ade_path, 'r') as file:
        for line in file:
            return json.loads(line).get("avgade")
    return None

def compare_ade(scene_number, do_path, no_path):
    """
    Compares the avgADE from both folders (doScenes and non-doScenes annotated scenes)
    and prints the results.
    """

    ade_doScenes = read_avgade(do_path)
    ade_no_doScenes = read_avgade(no_path)

    print("\n[ADE COMPARISON]")
    print(f"Scene {str(scene_number).zfill(3)}:")
    print(f"  Default OpenEMMA avgADE:     {ade_no_doScenes:.3f}")
    print(f"  doScenes-Prompted avgADE:    {ade_doScenes:.3f}")

    delta = ade_doScenes - ade_no_doScenes
    direction = "higher" if delta > 0 else "lower"
    print(f"  doScenes avgADE is {abs(delta):.3f} {direction} than default.\n")

if __name__ == "__main__":
    scene_input = input("Enter scene number (e.g., 4): ").strip()
    while not scene_input.isdigit():
        scene_input = input("Invalid input. Enter a numeric scene number: ").strip()

    scene_number = int(scene_input)
    doScenes_path = get_scene_folder(scene_number, is_doScenes=True)
    no_doScenes_path = get_scene_folder(scene_number, is_doScenes=False)

    if not os.path.exists(doScenes_path) or not os.path.exists(no_doScenes_path):
        print("[ERROR] One or both scene folders do not exist:")
    else:
        plot_trajectories_side_by_side(doScenes_path, no_doScenes_path, scene_number)
        compare_ade(scene_number, doScenes_path, no_doScenes_path)
