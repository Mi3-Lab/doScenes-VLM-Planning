import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded path of OpenEMMA results
base_path = "/home/mi3/parthib_openemma_to_nuplan/OpenEMMA/llava_results/openemma"

def get_scene_folder_path(scene_number, is_doScenes_annotation_used):
    """
    Constructs and returns the full path to the scene folder.
    """

    scene_str = str(scene_number).zfill(3)
    suffix = "_doScenes" if is_doScenes_annotation_used else "_no_doScenes"
    scene_folder = f"scene_{scene_str}{suffix}"
    return os.path.join(base_path, scene_folder)


def move_files_to_subfolder(folder_path, filename_suffix, subfolder_name):
    """
    Moves files that end with a given suffix into a specified subfolder
    within the provided folder and prints a summary.
    """

    destination_path = os.path.join(folder_path, subfolder_name)
    os.makedirs(destination_path, exist_ok=True)

    moved_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(filename_suffix):
            source_file = os.path.join(folder_path, filename)
            destination_file = os.path.join(destination_path, filename)
            shutil.move(source_file, destination_file)
            moved_count += 1

    print(f"Moved {moved_count} file(s) to '{subfolder_name}' folder")

def plot_npy_files(folder_path, title, output_filename):
    """
    Loads and plots all .npy files from the given folder and saves the plot.
    """

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    
    plt.figure(figsize=(8, 6))

    for file in files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path, allow_pickle=True)
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y, marker='o', label=file)

    plt.title(title)
    plt.legend(fontsize='small', loc='best', ncol=2)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True)
    plt.tight_layout()

    # Save to parent folder
    save_path = os.path.join(os.path.dirname(folder_path), output_filename)
    plt.savefig(save_path)
    print(f"[PLOT] Saved plot to: {save_path}")

if __name__ == "__main__":
    scene_input = input("Enter scene number (e.g., 4): ").strip()
    doscenes_input = input("Was doScenes annotation used? (y/n): ").strip().lower()

    while not scene_input.isdigit():
        scene_input = input("Invalid input. Enter a numeric scene number: ").strip()

    while doscenes_input not in ["y", "n"]:
        doscenes_input = input("Invalid input. Please enter 'y' or 'n': ").strip().lower()

    scene_number = int(scene_input)
    is_doScenes_annotation_used = doscenes_input == "y"
    folder_path = get_scene_folder_path(scene_number, is_doScenes_annotation_used)

    if not os.path.exists(folder_path):
        print(f"Error: Folder does not exist → {folder_path}")
    else:
        move_files_to_subfolder(folder_path, "_pred_traj.npy", "pred_trajectories")
        move_files_to_subfolder(folder_path, "pred_speeds.npy", "pred_speeds")
        move_files_to_subfolder(folder_path, "pred_curvatures.npy", "pred_curvatures")
        move_files_to_subfolder(folder_path, "front_cam.jpg", "fromt_cam_images")
        move_files_to_subfolder(folder_path, "logs.txt", "logs")
        move_files_to_subfolder(folder_path, "traj.jpg", "trajectory_plots")

        traj_folder_path = os.path.join(folder_path, "pred_trajectories")
        plot_title = "doScenes-prompted OpenEMMA Trajectories" if is_doScenes_annotation_used else "Default OpenEMMA Trajectories"
        plot_npy_files(traj_folder_path, plot_title, "[PLOT]_trajectories_plot.png")
        # plot_npy_files(os.path.join(folder_path, "pred_speeds"), "Predicted Speeds", "[PLOT]_speeds_plot.png")
        # plot_npy_files(os.path.join(folder_path, "pred_curvatures"), "Predicted Curvatures", "[PLOT]_curvatures_plot.png")
