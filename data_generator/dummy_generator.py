#!/usr/bin/env python3
import os
import signal
import multiprocessing as mp
import threading
import shutil
import json
import pickle
import numpy as np
from datetime import datetime

# --- Configuration (replace these values as needed) ---
map_name = "250312_mobility_map"
RT_name = "250704_RT_map"
gpu_ids = [0, 1, 2, 3, 4, 5]
enable_monitoring = False
# -----------------------------------------------------

def dummy_worker(
    gpu_id: int,
    splits_dir: str,
    BS_path: str,
    RT_path: str,
    output_dir: str,
    json_file: str
) -> None:
    """
    Dummy worker function to generate mock path data without actual RT calculations.
    """
    print(f"[GPU {gpu_id}] Starting dummy worker for {json_file}")
    
    # Load config for path attributes
    config_path = os.path.join(os.path.dirname(__file__), "src", "worker_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load split data
    split_json_path = os.path.join(splits_dir, json_file)
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
    
    # Create log file
    worker_log_filename = os.path.join(output_dir, "log", f"{os.path.splitext(json_file)[0]}_dummy_log.txt")
    split_log_filename = os.path.join(output_dir, "log", f"{os.path.splitext(json_file)[0]}_log.txt")
    
    # Load BS coordinates for reference
    import pandas as pd
    bs_coords = pd.read_excel(BS_path).to_dict(orient='records')
    num_bs = len(bs_coords)
    
    # Get total number of roads and UEs to process
    total_roads = len(split_data)
    total_ues = sum(len(ue_list) for ue_list in split_data.values())
    total_directions = total_ues * 4
    
    # Log total number of roads and UEs to process
    with open(split_log_filename, 'a') as log_file:
        log_file.write(f"Total roads to process: {total_roads}, Total UEs: {total_ues}, Total directions: {total_directions}\n")
        log_file.write(f"Roads: {', '.join(sorted(split_data.keys()))}\n")
        log_file.flush()
    
    processed_roads = 0
    processed_ues = 0
    
    # Process each road in sorted order
    for road_name in sorted(split_data.keys()):
        print(f"[GPU {gpu_id}] Processing road: {road_name} with {len(split_data[road_name])} UEs")
        
        # Create base name for the road (remove .ply extension)
        base = road_name.replace('.ply', '')
        
        # Create road directory
        road_path = os.path.join(output_dir, "paths", base)
        if not os.path.exists(road_path):
            os.makedirs(road_path)
        
        # Initialize base coordinates file
        base_coords_path = os.path.join(road_path, f"{base}_coordinates.json")
        base_coordinates = []
        
        # Process each UE
        ue_list = split_data[road_name]
        for j, ue in enumerate(ue_list):
            # Process each direction (1-4)
            for direction in range(1, 5):
                rx_name = f"rx_{base}_{j}_{direction}"
                
                # Generate dummy path data
                dummy_path_data = generate_dummy_path_data(config, num_bs, rx_name, ue, direction)
                
                # Save as pickle file (same format as original)
                out_name = f"rx_{j}_{direction}.pkl"
                out_path = os.path.join(road_path, out_name)
                
                with open(out_path, 'wb') as pf:
                    pickle.dump(dummy_path_data, pf)
                
                # Add coordinates for this rx to base coordinates file
                base_coordinates.append({
                    'rx_name': rx_name,
                    'position': [ue['x'], ue['y'], ue['z']],
                    'index': j,
                    'direction': direction
                })
                
                # Log UE position to split log
                with open(split_log_filename, 'a') as log_file:
                    log_file.write(f"{rx_name}: position=[{ue['x']}, {ue['y']}, {ue['z']}], direction={direction}\n")
                    log_file.flush()
                
                print(f"[GPU {gpu_id}] Saved dummy paths → {out_path}")
                processed_ues += 1
        
        # Save coordinates file for this road
        with open(base_coords_path, 'w') as f:
            json.dump(base_coordinates, f, indent=2)
        
        processed_roads += 1
        with open(split_log_filename, 'a') as log_file:
            log_file.write(f"Completed road {road_name}: {processed_roads}/{total_roads} roads, {processed_ues}/{total_directions} directions\n")
            log_file.flush()
        
        # Log progress
        with open(worker_log_filename, 'a') as log_file:
            log_file.write(f"Completed road {road_name}\n")
            log_file.flush()
    
    # Log final progress
    with open(split_log_filename, 'a') as log_file:
        log_file.write(f"All UEs processed: {processed_roads}/{total_roads} roads, {processed_ues}/{total_directions} directions\n")
        log_file.flush()
    
    print(f"[GPU {gpu_id}] Completed dummy worker for {json_file}")

def generate_dummy_path_data(config, num_bs, rx_name, ue, direction):
    """
    Generate realistic dummy path data that matches the expected format.
    """
    num_paths = np.random.randint(1, 16)
    
    # Create dummy data structure matching the original format
    path_attributes = {}
    
    for attr in config["path_attributes"]:
        if attr == "a_re":
            dummy_data = []
            for _ in range(8):
                path_data = []
                for _ in range(26):
                    ant_data = []
                    for _ in range(16):
                        ant_data.append(np.random.normal(0, 1e-6, num_paths).tolist())
                    path_data.append(ant_data)
                dummy_data.append(path_data)
            path_attributes[attr] = [dummy_data]
            
        elif attr == "a_im":
            dummy_data = []
            for _ in range(8):
                path_data = []
                for _ in range(26):
                    ant_data = []
                    for _ in range(16):
                        ant_data.append(np.random.normal(0, 1e-6, num_paths).tolist())
                    path_data.append(ant_data)
                dummy_data.append(path_data)
            path_attributes[attr] = [dummy_data]
            
        elif attr == "phi_r":
            path_attributes[attr] = [
                [np.random.uniform(-np.pi, np.pi, num_paths).tolist() for _ in range(26)]
            ]
        elif attr == "phi_t":
            path_attributes[attr] = [
                [np.random.uniform(-np.pi, np.pi, num_paths).tolist() for _ in range(26)]
            ]
        elif attr == "tau":
            path_attributes[attr] = [
                [np.random.exponential(100e-9, num_paths).tolist() for _ in range(26)]
            ]
        elif attr == "theta_r":
            path_attributes[attr] = [
                [np.random.uniform(-np.pi/2, np.pi/2, num_paths).tolist() for _ in range(26)]
            ]
        elif attr == "theta_t":
            path_attributes[attr] = [
                [np.random.uniform(-np.pi/2, np.pi/2, num_paths).tolist() for _ in range(26)]
            ]
        elif attr == "interactions":
            interactions = []
            for _ in range(7):  # max_depth = 7
                depth_interactions = [
                    [np.random.randint(0, 5, num_paths).tolist() for _ in range(26)]
                ]
                interactions.append(depth_interactions)
            path_attributes[attr] = interactions
        elif attr == "valid":
            path_attributes[attr] = [
                [np.random.choice([True, False], num_paths, p=[0.8, 0.2]).tolist() for _ in range(26)]
            ]
    
    # Create the complete data structure (same format as path_generator)
    result = [path_attributes]
    
    # Add metadata (same as path_generator)
    metadata = {
        "rx_name": rx_name,
        "rx_position": [ue['x'], ue['y'], ue['z']],
        "direction": direction
    }
    result.append(metadata)
    
    return result

if __name__ == "__main__":
    monitoring_active = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    map_list_path = os.path.join(project_root, "map", "map_lists")
    splits_dir = os.path.join(map_list_path, map_name, "splits")
    BS_dir = os.path.join(map_list_path, map_name, f"{map_name}_BS_coord.xlsx")
    RT_path = os.path.join(map_list_path, RT_name, f"{RT_name}.xml")
    output_dir = os.path.join(script_dir, f"{RT_name}_paths_dummy")
    
    # Check required files and directories
    if not os.path.isdir(splits_dir):
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    if not os.path.isfile(BS_dir):
        raise FileNotFoundError(f"BS coord file not found: {BS_dir}")
    
    json_files = sorted([
        fn for fn in os.listdir(splits_dir)
        if fn.startswith("split_") and fn.endswith(".json")
    ])
    if not json_files:
        raise RuntimeError(f"No split_*.json files in {splits_dir}")
    
    # Create output directories
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(os.path.join(output_dir, "log")):
        os.makedirs(os.path.join(output_dir, "log"))
    if not os.path.isdir(os.path.join(output_dir, "paths")):
        os.makedirs(os.path.join(output_dir, "paths"))
    
    def signal_handler(sig, frame):
        global monitoring_active
        print("\nReceived signal to terminate. Finishing jobs...")
        monitoring_active = False
    
    signal.signal(signal.SIGINT, signal_handler)
    mp.set_start_method("spawn", force=True)
    
    processes = []
    worker_args_list = []
    
    for i, jf in enumerate(json_files):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # Cycle through available GPUs
        args = (gpu_id, splits_dir, BS_dir, RT_path, output_dir, jf)
        worker_args_list.append(args)
        
        p = mp.Process(
            target=dummy_worker,
            args=args,
            name=f"dummy_worker_{RT_name}_GPU{gpu_id}"
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    monitoring_active = False
    print("All dummy data generation completed.")