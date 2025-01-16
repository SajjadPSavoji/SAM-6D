import os
import sys
import math
import time
import subprocess

#############################
# 1) Configuration
#############################

# Path to your ShapeNet directory
RENDER_DIR = os.path.dirname(os.path.abspath(__file__))

SHAPENET_PATH = os.path.join(
    RENDER_DIR,
    '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/shapenetcorev2'
)
SHAPENET_ORIG_PATH = os.path.join(SHAPENET_PATH, 'models_orig')
OUTPUT_DIR = os.path.join(
    RENDER_DIR,
    '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/templates'
)

# The BlenderProc script that actually does the rendering
RENDER_SCRIPT = "render_shapenet_templates.py"

# Number of parallel processes to launch (can be > number of GPUs)
NUM_PROCESSES = 80

# GPU IDs you have available (e.g., 8 GPUs).
# We'll assign processes in a round-robin manner across these GPUs.
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

#############################
# 2) Helper Functions
#############################

def chunkify(lst, num_chunks):
    """
    Splits the list `lst` into `num_chunks` parts (as evenly as possible).
    Returns a list of sublists.
    """
    chunk_size = math.ceil(len(lst) / num_chunks)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def custom_progress_bar(current, total, start_time):
    """
    Display a custom progress bar with ETA using `print`.
    """
    progress = current / total
    bar_length = 40
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    elapsed_time = time.time() - start_time
    eta = (elapsed_time / (current + 1)) * (total - current - 1) if current > 0 else 0

    elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta))

    print(f'\r[{bar}] {current}/{total} - Elapsed: {elapsed_formatted} - ETA: {eta_formatted}', end='')
    if current == total - 1:
        print()

#############################
# 3) Main driver code
#############################

if __name__ == "__main__":

    # -----------------------------------------------------
    # A) Gather all (synset_id, source_id) that are not yet rendered
    # -----------------------------------------------------
    all_items = []

    # Count total objects for progress bar
    total_objects = 0
    for synset_id in os.listdir(SHAPENET_ORIG_PATH):
        synset_fpath = os.path.join(SHAPENET_ORIG_PATH, synset_id)
        if not os.path.isdir(synset_fpath) or '.' in synset_id:
            continue
        total_objects += len(os.listdir(synset_fpath))

    start_time = time.time()
    cur_iter = 0

    for synset_id in os.listdir(SHAPENET_ORIG_PATH):
        synset_fpath = os.path.join(SHAPENET_ORIG_PATH, synset_id)
        if not os.path.isdir(synset_fpath) or '.' in synset_id:
            continue

        for source_id in os.listdir(synset_fpath):
            cur_iter += 1
            custom_progress_bar(cur_iter, total_objects, start_time)

            save_synset_folder = os.path.join(OUTPUT_DIR, synset_id)
            if not os.path.exists(save_synset_folder):
                os.makedirs(save_synset_folder)

            save_fpath = os.path.join(save_synset_folder, source_id)
            if not os.path.exists(save_fpath):
                os.mkdir(save_fpath)

            cad_path = os.path.join(SHAPENET_ORIG_PATH, synset_id, source_id)
            obj_fpath = os.path.join(cad_path, 'models', 'model_normalized.obj')
            if not os.path.exists(obj_fpath):
                # No valid model found
                continue

            # Check if already rendered (rgb_41.png as your "completion marker")
            rgb42path = os.path.join(save_fpath, 'rgb_41.png')
            if os.path.exists(rgb42path):
                # Already rendered
                continue

            # If we get here, we want to render this item
            all_items.append((synset_id, source_id))

    print(f"\nTotal items to render: {len(all_items)}")

    # -----------------------------------------------------
    # B) Split the items into N chunks (NUM_PROCESSES)
    # -----------------------------------------------------
    if len(all_items) == 0:
        print("No new items to render. Exiting.")
        sys.exit(0)

    items_chunks = chunkify(all_items, NUM_PROCESSES)

    # -----------------------------------------------------
    # C) Launch BlenderProc processes
    # -----------------------------------------------------
    processes = []
    for rank, chunk in enumerate(items_chunks):
        # Create a file that contains this chunk's (synset_id, source_id) pairs
        chunk_file = f"chunk_{rank}.txt"
        with open(chunk_file, 'w') as f:
            for (synset_id, source_id) in chunk:
                f.write(f"{synset_id},{source_id}\n")

        # Round-robin GPU assignment: pick GPU based on rank
        gpu_index = GPU_IDS[rank % len(GPU_IDS)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        # Compose the command to call BlenderProc with the render script
        cmd = [
            "blenderproc",
            "run",
            RENDER_SCRIPT,
            "--chunk_file", chunk_file
        ]

        print(f"\nLaunching process {rank} on GPU {gpu_index} with {len(chunk)} items...")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # -----------------------------------------------------
    # D) Wait for all processes
    # -----------------------------------------------------
    for p in processes:
        p.wait()

    print("\nAll rendering processes finished.")

    # -----------------------------------------------------
    # E) Clean up temp chunk files
    # -----------------------------------------------------
    for rank in range(len(items_chunks)):
        chunk_file = f"chunk_{rank}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

    print("\nAll temp files cleaned up.")