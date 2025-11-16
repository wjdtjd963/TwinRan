import os
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Set, Tuple, Any

def load_bs_coords(BS_path: str) -> List[Dict[str, Any]]:
    """
    Load base station coordinates from an Excel file.
    Args:
        BS_path (str): Path to the Excel file.
    Returns:
        list: List of dictionaries with BS coordinates.
    """
    if not os.path.isfile(BS_path):
        raise FileNotFoundError(f"BS coord file not found: {BS_path}")
    df = pd.read_excel(BS_path)
    expected_cols = ['X', 'Y', 'Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {BS_path}")
    df = df[expected_cols]
    coords = df.to_dict(orient='records')
    return coords

def path_attr_serializer(paths, config: dict = None) -> List[Dict[str, Any]]:
    """
    Serialize path attributes to a list of dictionaries.
    Args:
        paths: Path object with attributes to serialize.
        config: Configuration dictionary (should contain 'path_attributes' list).
    Returns:
        list: List of dictionaries with path attributes.
    """
    serializable = []
    attr_list = config["path_attributes"]
    attr_dict = {}
    
    for attr in attr_list:
        if attr == "a_re":
            arr = paths.a[0].numpy()
            if arr.ndim == 6:
                arr = arr[0]
            attr_dict["a_re"] = arr.tolist()
            
        elif attr == "a_im":
            arr = paths.a[1].numpy()
            if arr.ndim == 6:
                arr = arr[0]
            attr_dict["a_im"] = arr.tolist()
            
        elif hasattr(paths, attr):
            arr = getattr(paths, attr).numpy()
            attr_dict[attr] = arr.tolist()
        else:
            pass
    
    serializable.append(attr_dict)
    return serializable

def road_name_comes_after(road_a: str, road_b: str) -> bool:
    """
    Check if road_a comes after road_b in alphabetical order.
    """
    return road_a > road_b

def positions_match(pos1: List[float], pos2: List[float], tolerance: float = 1e-6) -> bool:
    """
    Check if two positions match within a given tolerance.
    """
    return all(abs(a - b) < tolerance for a, b in zip(pos1, pos2))

def log_worker(message: str, gpu_id: int, worker_log_filename: str) -> None:
    """
    Log a message to both console and worker log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}][GPU {gpu_id}] {message}"
    print(full_message)
    try:
        with open(worker_log_filename, 'a') as log_file:
            log_file.write(full_message + "\n")
            log_file.flush()
    except Exception as e:
        print(f"Error writing to worker log: {e}")

def load_last_processed_ue(split_log_filename: str, gpu_id: int, worker_log_filename: str) -> Tuple[Dict[str, Any], int, int]:
    """
    Load last processed UE and progress from split log file.
    Returns:
        tuple: (last_processed_ue, processed_roads, processed_ues)
    """
    last_processed_ue = None
    processed_roads = 0
    processed_ues = 0
    if os.path.exists(split_log_filename):
        with open(split_log_filename, 'r') as log_file:
            log_lines = log_file.readlines()
            for line in reversed(log_lines):
                if "Completed road" in line and "roads" in line and "UEs" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        counts_part = parts[1].strip()
                        road_count_match = re.search(r'(\d+)/\d+ roads', counts_part)
                        ue_count_match = re.search(r'(\d+)/\d+ UEs', counts_part)
                        if road_count_match:
                            processed_roads = int(road_count_match.group(1))
                            log_worker(f"Found {processed_roads} processed roads from log", gpu_id, worker_log_filename)
                        if ue_count_match:
                            processed_ues = int(ue_count_match.group(1))
                        break
            if log_lines:
                for line in reversed(log_lines):
                    if ': position=' in line:
                        try:
                            parts = line.split(': position=')
                            rx_name = parts[0]
                            position_str = parts[1].strip()
                            
                            # Remove direction part if present
                            if ', direction=' in position_str:
                                position_str = position_str.split(', direction=')[0]
                            
                            if position_str.startswith('['):
                                position_str = position_str[1:]
                            if position_str.endswith(']'):
                                position_str = position_str[:-1]
                            position_values = [val.strip() for val in position_str.split(',')]
                            last_position = [float(pos) for pos in position_values]
                            road_name_parts = rx_name.split('_')
                            if len(road_name_parts) >= 4:
                                road_name_part = '_'.join(road_name_parts[1:-2])
                                last_idx = int(road_name_parts[-2])
                                direction = int(road_name_parts[-1])
                                last_road_name = f"{road_name_part}.ply"
                                last_processed_ue = {
                                    'road_name': last_road_name,
                                    'index': last_idx,
                                    'direction': direction,
                                    'position': last_position
                                }
                                log_worker(f"Last processed UE: {rx_name} at {last_position}", gpu_id, worker_log_filename)
                                break
                        except ValueError as e:
                            log_worker(f"Error parsing position from log: {e}, line: {line}", gpu_id, worker_log_filename)
                            continue
                        except Exception as e:
                            log_worker(f"Unexpected error parsing log: {e}, line: {line}", gpu_id, worker_log_filename)
                            continue
    return last_processed_ue, processed_roads, processed_ues

def track_processed_roads(split_data: Dict[str, Any], processed_roads: int, last_processed_ue: Dict[str, Any], gpu_id: int, worker_log_filename: str) -> Set[str]:
    """
    Track and return the set of already processed road names.
    """
    processed_road_names = set()
    if processed_roads > 0:
        sorted_roads = sorted(split_data.keys())
        if last_processed_ue:
            for road_name in sorted_roads:
                if road_name_comes_after(last_processed_ue['road_name'], road_name):
                    processed_road_names.add(road_name)
                    log_worker(f"Marking road {road_name} as already processed based on last UE", gpu_id, worker_log_filename)
        if len(processed_road_names) < processed_roads:
            remaining = processed_roads - len(processed_road_names)
            for road_name in sorted_roads:
                if road_name not in processed_road_names and remaining > 0:
                    processed_road_names.add(road_name)
                    remaining -= 1
                    log_worker(f"Marking road {road_name} as already processed based on count", gpu_id, worker_log_filename)
    return processed_road_names

def get_orientation_for_direction(direction: int) -> list:
    """
    Get orientation vector for a specific direction (1-4).
    Direction 1: [0, 1, 0] (North), Direction 2: [1, 0, 0] (East), 
    Direction 3: [0, -1, 0] (South), Direction 4: [-1, 0, 0] (West)
    """
    if direction == 1:  # North
        return [0.0, 1.0, 0.0]
    elif direction == 2:  # East
        return [1.0, 0.0, 0.0]
    elif direction == 3:  # South
        return [0.0, -1.0, 0.0]
    elif direction == 4:  # West
        return [-1.0, 0.0, 0.0]
    else:
        return [0.0, 1.0, 0.0]  # Default to North

def process_each_road(
    split_data: Dict[str, Any],
    processed_road_names: Set[str],
    last_processed_ue: Dict[str, Any],
    output_dir: str,
    json_file: str,
    gpu_id: int,
    worker_log_filename: str
) -> Tuple[list, int]:
    """
    Prepare road iteration info for UE-level processing.
    Returns:
        tuple: (road_iter_info, render_count)
    """
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    existing_images = [f for f in os.listdir(images_dir) 
                      if f.startswith(os.path.splitext(json_file)[0]) and f.endswith("_scene.png")]
    render_count = len(existing_images)
    log_worker(f"Found {render_count} existing rendered images for this worker", gpu_id, worker_log_filename)
    road_iter_info = []
    for road_name in sorted(split_data.keys()):
        if road_name in processed_road_names:
            log_worker(f"Skipping already processed road {road_name}", gpu_id, worker_log_filename)
            continue
        start_idx = 0
        if last_processed_ue and road_name == last_processed_ue['road_name']:
            # If the last direction was 4, move to next UE, otherwise stay on same UE
            if last_processed_ue['direction'] >= 4:
                start_idx = last_processed_ue['index'] + 1
            else:
                start_idx = last_processed_ue['index']
            log_worker(f"Resuming from UE index {start_idx} for road {road_name}", gpu_id, worker_log_filename)
        elif last_processed_ue and road_name_comes_after(last_processed_ue['road_name'],road_name):
            log_worker(f"Skipping road {road_name} as it was already processed", gpu_id, worker_log_filename)
            continue
        base = road_name.replace('.ply', '')
        road_iter_info.append({
            'road_name': road_name,
            'start_idx': start_idx,
            'base': base
        })
    return road_iter_info, render_count 