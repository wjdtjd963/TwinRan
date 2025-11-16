import os
import json
from src.worker_utils import (
    load_bs_coords, path_attr_serializer, positions_match,
    log_worker, load_last_processed_ue, track_processed_roads, process_each_road,
    get_orientation_for_direction
)

# --- Main worker function ---
def worker(
    gpu_id: int,
    splits_dir: str,
    BS_path: str,
    RT_path: str,
    output_dir: str,
    json_file: str
) -> None:
    """
    Main worker function to process a single split file and generate path/scene data.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import tensorflow as tf
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver, Camera
    import pickle

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "worker_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    tx_cfg = config["antenna"]["tx"]
    rx_cfg = config["antenna"]["rx"]
    scene_cfg = config["scene"]
    solver_cfg = config["path_solver"]

    worker_log_filename = os.path.join(output_dir, "log", f"{os.path.splitext(json_file)[0]}_worker_log.txt")
    log_worker(f"Starting worker for {json_file}", gpu_id, worker_log_filename)

    split_json_path = os.path.join(splits_dir, json_file)
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
    split_log_filename = os.path.join(output_dir, "log", f"{os.path.splitext(json_file)[0]}_log.txt")

    # Use helper to load last processed UE and progress
    last_processed_ue, processed_roads, processed_ues = load_last_processed_ue(split_log_filename, gpu_id, worker_log_filename)
    
    # Process each road in the split file
    with tf.device("/GPU:0"):
        log_worker(f"Processing {json_file}", gpu_id, worker_log_filename)
        scene = load_scene(RT_path)
        p_solver = PathSolver()

        # Antenna setup (from config)
        scene.tx_array = PlanarArray(
            num_rows=tx_cfg["num_rows"],
            num_cols=tx_cfg["num_cols"],
            vertical_spacing=tx_cfg["vertical_spacing"],
            horizontal_spacing=tx_cfg["horizontal_spacing"],
            pattern=tx_cfg["pattern"],
            polarization=tx_cfg["polarization"]
        )
        scene.rx_array = PlanarArray(
            num_rows=rx_cfg["num_rows"],
            num_cols=rx_cfg["num_cols"],
            vertical_spacing=rx_cfg["vertical_spacing"],
            horizontal_spacing=rx_cfg["horizontal_spacing"],
            pattern=rx_cfg["pattern"],
            polarization=rx_cfg["polarization"]
        )
        scene.frequency = scene_cfg["frequency"]
        scene.synthetic_array = scene_cfg["synthetic_array"]

        # Load BS coordinates and add transmitters to the scene
        bs_coords = load_bs_coords(BS_path)
        log_worker(f"Loaded {len(bs_coords)} BS coords", gpu_id, worker_log_filename)
        for idx, c in enumerate(bs_coords):
            tx = Transmitter(
                name=f"tx_{idx}", 
                position=[c['X'], c['Y'], c['Z']], 
                look_at=[c['Ori_X'], c['Ori_Y'], c['Ori_Z']]
            )
            scene.add(tx)
            log_worker(f"Added tx{idx}: pos={tx.position}, look_at={tx.look_at}", gpu_id, worker_log_filename)

        # Set up directory for rendered images
        images_dir = os.path.join(output_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        existing_images = [f for f in os.listdir(images_dir) 
                          if f.startswith(os.path.splitext(json_file)[0]) and f.endswith("_scene.png")]
        render_count = len(existing_images)
        log_worker(f"Found {render_count} existing rendered images for this worker", gpu_id, worker_log_filename)
        # Get total number of roads and UEs to process
        total_roads = len(split_data)
        total_ues = sum(len(ue_list) for ue_list in split_data.values())  # Actual UE count
        total_directions = total_ues * 4  # 4 directions per UE
        # Log total number of roads and UEs to process
        with open(split_log_filename, 'a') as log_file:
            log_file.write(f"Total roads to process: {total_roads}, Total UEs: {total_ues}, Total directions: {total_directions}\n")
            log_file.write(f"Roads: {', '.join(sorted(split_data.keys()))}\n")
            log_file.flush()
            
        # Use helper to track processed roads
        processed_road_names = track_processed_roads(split_data, processed_roads, last_processed_ue, gpu_id, worker_log_filename)
        processed_roads = len(processed_road_names)
        log_worker(f"After adjustment: {processed_roads}/{total_roads} roads marked as processed", gpu_id, worker_log_filename)

        # Use helper to process each road (returns info for UE-level processing)
        road_iter_info, render_count = process_each_road(
            split_data, processed_road_names, last_processed_ue, output_dir, json_file, gpu_id, worker_log_filename
        )
        # Now process UEs for each road in the main worker
        for road_info in road_iter_info:
            road_name = road_info['road_name']
            start_idx = road_info['start_idx']
            base = road_info['base']
            ue_list = split_data[road_name]
            
            # Initialize base coordinates file if it doesn't exist
            base_coords_path = os.path.join(output_dir, "paths", base, f"{base}_coordinates.json")
            base_coords_dir = os.path.dirname(base_coords_path)
            
            # Create directory if it doesn't exist
            if not os.path.exists(base_coords_dir):
                os.makedirs(base_coords_dir)
                log_worker(f"Created directory: {base_coords_dir}", gpu_id, worker_log_filename)
            
            if not os.path.exists(base_coords_path):
                base_coordinates = []
                with open(base_coords_path, 'w') as f:
                    json.dump(base_coordinates, f)
                log_worker(f"Created new coordinates file: {base_coords_path}", gpu_id, worker_log_filename)
            else:
                # Load existing coordinates
                with open(base_coords_path, 'r') as f:
                    base_coordinates = json.load(f)
                log_worker(f"Loaded existing coordinates from: {base_coords_path}", gpu_id, worker_log_filename)
            if start_idx > 0 and start_idx - 1 < len(ue_list):
                last_ue = ue_list[start_idx - 1]
                last_ue_pos = [last_ue['x'], last_ue['y'], last_ue['z']]
                if positions_match(last_ue_pos, last_processed_ue['position']):
                    log_worker(f"Position verification successful for last processed UE at index {start_idx - 1}", gpu_id, worker_log_filename)
                    # Check if we need to resume from a specific direction
                    if 'direction' in last_processed_ue:
                        start_direction = last_processed_ue['direction'] + 1
                        if start_direction > 4:
                            start_direction = 1
                            start_idx += 1
                        log_worker(f"Resuming from direction {start_direction} for UE at index {start_idx - 1}", gpu_id, worker_log_filename)
                else:
                    log_worker(f"Warning: Last processed UE position doesn't match between log and current data", gpu_id, worker_log_filename)
                    log_worker(f"  Log: {last_processed_ue['position']}", gpu_id, worker_log_filename)
                    log_worker(f"  Current data: {last_ue_pos}", gpu_id, worker_log_filename)
                    log_worker(f"  This might indicate data inconsistency or log corruption", gpu_id, worker_log_filename)
            elif last_processed_ue != None:
                log_worker(f"Warning: Last processed UE index {start_idx - 1} is out of range for road {last_processed_ue['road_name']}", gpu_id, worker_log_filename)

            # Process each UE in the road
            for j, ue in enumerate(ue_list):
                if j < start_idx:
                    continue
                
                # Process each direction for this UE
                start_direction = 1
                if last_processed_ue and 'direction' in last_processed_ue and positions_match([ue['x'], ue['y'], ue['z']], last_processed_ue['position']):
                    start_direction = last_processed_ue['direction'] + 1
                    if start_direction > 4:
                        continue  # Move to next UE
                
                for direction in range(start_direction, 5):  # 1, 2, 3, 4
                    rx_name = f"rx_{base}_{j}_{direction}"
                    rx = Receiver(
                        name=rx_name,
                        position=[ue['x'], ue['y'], ue['z']],
                        look_at=[ue['x'], ue['y'], ue['z']] + get_orientation_for_direction(direction)
                    )
                    scene.add(rx)
                    log_worker(f"Added receiver {rx_name} with direction {direction}", gpu_id, worker_log_filename)

                    # P_solver settings (from config) - Run for each direction individually
                    paths = p_solver(
                        scene=scene,
                        max_depth=solver_cfg["max_depth"],
                        max_num_paths_per_src=solver_cfg["max_num_paths_per_src"],
                        samples_per_src=solver_cfg["samples_per_src"],
                        synthetic_array=solver_cfg["synthetic_array"],
                        los=solver_cfg["los"],
                        specular_reflection=solver_cfg["specular_reflection"],
                        diffuse_reflection=solver_cfg["diffuse_reflection"],
                        refraction=solver_cfg["refraction"]
                    )
                    log_worker(f"{rx_name} → paths found", gpu_id, worker_log_filename)

                    # Save paths
                    out_name = f"rx_{j}_{direction}.pkl"
                    base_path = os.path.join(output_dir, "paths", base)
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)
                    out_path = os.path.join(base_path, out_name)
                    path_data = path_attr_serializer(paths, config)
                    path_data.append({
                        "rx_name": rx_name,
                        "rx_position": [ue['x'], ue['y'], ue['z']],
                        "direction": direction
                    })
                    with open(out_path, 'wb') as pf:
                        import pickle
                        pickle.dump(path_data, pf)
                    log_worker(f"Saved paths → {out_path}", gpu_id, worker_log_filename)

                    # Add coordinates for this rx to base coordinates file
                    base_coordinates.append({
                        'rx_name': rx_name,
                        'position': [ue['x'], ue['y'], ue['z']],
                        'index': j,
                        'direction': direction
                    })
                    with open(base_coords_path, 'w') as f:
                        json.dump(base_coordinates, f, indent=2)
                    log_worker(f"Added coordinates for {rx_name} (direction {direction}) to {base_coords_path}", gpu_id, worker_log_filename)
                    
                    # Log UE position to split log
                    with open(split_log_filename, 'a') as log_file:
                        log_file.write(f"{rx_name}: position=[{ue['x']}, {ue['y']}, {ue['z']}], direction={direction}\n")
                        log_file.flush()
                    log_worker(f"Logged UE position to split log", gpu_id, worker_log_filename)

                    # Render scene image
                    if render_count < 5:
                        image_filename = f"{os.path.splitext(json_file)[0]}_{rx_name}_scene.png"
                        image_path = os.path.join(images_dir, image_filename)
                        if os.path.exists(image_path):
                            log_worker(f"Image for {rx_name} already exists, skipping rendering", gpu_id, worker_log_filename)
                        else:
                            my_cam = Camera(position=[0,0,1500], look_at=[0,0,0])
                            scene.render_to_file(camera=my_cam,
                                    paths=paths,
                                    filename=image_path,
                                    resolution=[2500,2500])
                            log_worker(f"Rendered scene image for {rx_name} (image {render_count+1} of 5)", gpu_id, worker_log_filename)

                            import numpy as np
                            attr_dims = {}
                            for attr in config["path_attributes"]:
                                if attr in ["a_re", "a_im"]:
                                    if "a" not in attr_dims:
                                        if hasattr(paths, "a"):
                                            arr = getattr(paths, "a")
                                            arr_np = arr.numpy() if hasattr(arr, "numpy") else np.array(arr)
                                            attr_dims["a"] = arr_np.shape
                                        else:
                                            attr_dims["a"] = None
                                else:
                                    if hasattr(paths, attr):
                                        arr = getattr(paths, attr)
                                        arr_np = arr.numpy() if hasattr(arr, "numpy") else np.array(arr)
                                        attr_dims[attr] = arr_np.shape
                                    else:
                                        attr_dims[attr] = None
                            log_message = f"{rx_name} path attribute dimensions:\n"
                            for k, v in attr_dims.items():
                                log_message += f"    {k}: {v}\n"
                            log_worker(log_message.strip(), gpu_id, worker_log_filename)
                            render_count += 1
                    
                    scene.remove(rx_name)
                    log_worker(f"Removed receiver {rx_name}", gpu_id, worker_log_filename)
                    processed_ues += 1  # Count per direction
            
            processed_roads += 1
            with open(split_log_filename, 'a') as log_file:
                log_file.write(f"Completed road {road_name}: {processed_roads}/{total_roads} roads, {processed_ues}/{total_directions} directions\n")
                log_file.flush()

    # Log final progress
    with open(split_log_filename, 'a') as log_file:
        log_file.write(f"All UEs processed: {processed_roads}/{total_roads} roads, {processed_ues}/{total_directions} directions\n")
        log_file.flush()
    log_worker(f"All UEs processed for {json_file}: {processed_roads}/{total_roads} roads, {processed_ues}/{total_directions} directions", gpu_id, worker_log_filename)
