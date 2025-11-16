#!/usr/bin/env python3
import os
import signal
import multiprocessing as mp
import threading
import shutil
from src.worker import worker
from src.GPU_monitor import monitor_and_restart_workers

# --- Configuration (replace these values as needed) ---
map_name = "250312_mobility_map"
RT_name = "250704_RT_map"
gpu_ids = [0, 1, 2, 3, 4, 5]
enable_monitoring = True
threshold = 25
interval = 60
# -----------------------------------------------------

if __name__ == "__main__":
    monitoring_active = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    map_list_path = os.path.join(project_root, "map", "map_lists")
    splits_dir = os.path.join(map_list_path, map_name, "splits")
    BS_dir = os.path.join(map_list_path, map_name, f"{map_name}_BS_coord.xlsx")
    RT_path = os.path.join(map_list_path, RT_name, f"{RT_name}.xml")
    output_dir = os.path.join(script_dir, f"{RT_name}_paths")
    if not os.path.isdir(splits_dir):
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    if not os.path.isfile(RT_path):
        raise FileNotFoundError(f"RT XML not found: {RT_path}")
    json_files = sorted([
        fn for fn in os.listdir(splits_dir)
        if fn.startswith("split_") and fn.endswith(".json")
    ])
    if not json_files:
        raise RuntimeError(f"No split_*.json files in {splits_dir}")
    if len(gpu_ids) < len(json_files):
        raise ValueError(
            f"You provided {len(gpu_ids)} GPUs but need at least {len(json_files)}."
        )
    gpu_ids = gpu_ids[:len(json_files)]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(os.path.join(output_dir,"log")):
        os.makedirs(os.path.join(output_dir,"log"))
    if not os.path.isdir(os.path.join(output_dir,"paths")):
        os.makedirs(os.path.join(output_dir,"paths"))
    def signal_handler(sig, frame):
        global monitoring_active
        print("\nReceived signal to terminate. Finishing jobs...")
        monitoring_active = False
    signal.signal(signal.SIGINT, signal_handler)
    mp.set_start_method("spawn", force=True)
    processes = []
    worker_args_list = []
    for i, jf in enumerate(json_files):
        gpu_id = gpu_ids[i]
        args = (gpu_id, splits_dir, BS_dir, RT_path, output_dir, jf)
        worker_args_list.append(args)
        p = mp.Process(
            target=worker,
            args=args,
            name=f"worker_{map_name}_GPU{gpu_id}"
        )
        p.start()
        processes.append(p)
    if enable_monitoring:
        print(f"Starting GPU monitoring")
        monitor_thread = threading.Thread(
            target=monitor_and_restart_workers,
            args=(monitoring_active, processes, worker_args_list, gpu_ids, threshold, interval, output_dir),
            name="gpu_monitor",
            daemon=True
        )
        monitor_thread.start()
    for p in processes:
        p.join()
    monitoring_active = False
    print("All jobs completed.")
    zip_filename = f"{RT_name}_paths.zip"
    zip_path = os.path.join(script_dir, zip_filename)
    print(f"Compressing output directory to {zip_filename}...")
    try:
        shutil.make_archive(
            os.path.splitext(zip_path)[0],
            'zip',
            output_dir
        )
        print(f"Compression completed. Archive saved to: {zip_path}")
        zip_size_bytes = os.path.getsize(zip_path)
        zip_size_mb = zip_size_bytes / (1024 * 1024)
        print(f"Archive size: {zip_size_mb:.2f} MB")
    except Exception as e:
        print(f"Error compressing output directory: {e}")
