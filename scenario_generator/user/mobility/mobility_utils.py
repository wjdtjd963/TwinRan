import numpy as np
import random

# Handle imports for both direct execution and module import
try:
    # When imported as a module
    from ...scenario_config import MATERIAL_USER_TYPE
    from ...scenario_constant import RANDOM_OFFSET_RANGE_XY, RANDOM_OFFSET_RANGE_Z
except ImportError:
    # When executed directly
    from scenario_config import MATERIAL_USER_TYPE
    from scenario_constant import RANDOM_OFFSET_RANGE_XY, RANDOM_OFFSET_RANGE_Z

def calculate_distance(coord1, coord2):
    """
    Calculate Euclidean distance between two coordinates.
    
    Args:
        coord1 (np.ndarray or list): First coordinate
        coord2 (np.ndarray or list): Second coordinate
    
    Returns:
        float: Distance between two coordinates
    """
    # --- Ensure both coordinates are numpy arrays and column vectors ---
    coord1 = np.array(coord1).reshape(-1, 1) if not isinstance(coord1, np.ndarray) else coord1
    coord2 = np.array(coord2).reshape(-1, 1) if not isinstance(coord2, np.ndarray) else coord2
    return np.linalg.norm(coord1 - coord2)

def calculate_vector_magnitude(vector):
    """
    Calculate the magnitude of a vector.
    
    Args:
        vector (np.ndarray or list): Vector
    
    Returns:
        float: Magnitude of the vector
    """
    # --- Ensure input is a numpy array ---
    vector = np.array(vector) if not isinstance(vector, np.ndarray) else vector
    return np.linalg.norm(vector)

def find_closest_point_index(coordinate, path_points):
    """
    Find the index of the closest path point to the given coordinate.
    
    Args:
        coordinate (np.ndarray): Current coordinate (3x1 vector)
        path_points (list): List of path points
    
    Returns:
        int: Index of the closest point
    """
    # --- Handle empty path case ---
    if not path_points:
        return 0
    current_point_idx = 0
    min_distance = float('inf')
    # --- Find the path point with the minimum distance ---
    for i, point in enumerate(path_points):
        dist = calculate_distance(coordinate, point)
        if dist < min_distance:
            min_distance = dist
            current_point_idx = i
    return current_point_idx

def get_road_types(map_data):
    """
    Classify roads by allowed user types based on material properties.
    
    Args:
        map_data (dict): Map data containing road information
    
    Returns:
        tuple: (pedestrian_roads, vehicle_roads) - Lists of road data tuples
    """
    pedestrian_roads = []
    vehicle_roads = []
    # --- Iterate over all roads and classify by allowed user type ---
    for road_name, road_list in map_data.items():
        for road_data in road_list:
            material = road_data.get("material", "")
            allowed_types = MATERIAL_USER_TYPE.get(material, [])
            if "pedestrian" in allowed_types:
                pedestrian_roads.append((road_name, road_data))
            if "vehicle" in allowed_types:
                vehicle_roads.append((road_name, road_data))
    return pedestrian_roads, vehicle_roads

def determine_destination(road_list, user):
    """
    Determine the next destination point for a user based on their current position and direction.
    
    Args:
        road_list (list): List of road data tuples (road_name, road_data)
        user (User): User object with current position and path information
    
    Raises:
        ValueError: If current path is not found or has no valid points
    """
    # --- Find the current road and path info for the user ---
    road_name = user.mobility.current_path[0]
    road_info = user.mobility.current_path[1]
    road_data = None
    for name, data in road_list:
        if name == road_name:
            road_data = data
            break
    if not road_data:
        # If the current path is not found, raise an error
        raise ValueError(f"Current path '{road_name}' not found in road list")
    path_points = road_data.get(road_info, [])
    if not path_points or len(path_points) <= 1:
        # If there are not enough path points, raise an error
        raise ValueError(f"No valid path points found for road info '{road_info}' on road '{road_name}'")
    current_point_idx = find_closest_point_index(user.mobility.coordinate, path_points)
    # --- Decide next destination based on direction ---
    if user.mobility.destination_direction:
        # If moving forward along the path
        if current_point_idx == len(path_points) - 1:
            # If at the end, try to find the next road
            find_next_road(road_data, road_list, user)
        else:
            # Otherwise, move to the next point
            user.mobility.destination = apply_height_randomness(path_points[current_point_idx + 1], user)
    else:
        # If moving backward along the path
        if current_point_idx == 0:
            # If at the start, try to find the previous road
            find_next_road(road_data, road_list, user)
        else:
            # Otherwise, move to the previous point
            user.mobility.destination = apply_height_randomness(path_points[current_point_idx - 1], user)

def _find_valid_neighbor_road(current_road_data, road_list, user_direction):
    """
    Find a valid neighboring road based on user direction.
    
    Args:
        current_road_data (dict): Current road data
        road_list (list): List of all available roads
        user_direction (bool): User movement direction
    
    Returns:
        tuple: (next_road_name, next_road_data) or (None, None) if no valid neighbor found
    """
    # --- Get the list of neighboring roads in the direction of movement ---
    neighbors = current_road_data.get("adjacent1" if user_direction else "adjacent0", [])
    random.shuffle(neighbors)
    # --- Search for a valid neighbor in the road list ---
    for next_road_name in neighbors:
        for name, data in road_list:
            if name == next_road_name:
                return next_road_name, data
    # --- If no valid neighbor is found, return None ---
    return None, None

def _handle_uturn(current_road_data, user):
    """
    Handle U-turn logic when no valid next road is found.
    
    Args:
        current_road_data (dict): Current road data
        user (User): User object to perform U-turn
    """
    road_name = user.mobility.current_path[0]
    road_info = user.mobility.current_path[1]
    path_points = current_road_data.get(road_info, [])
    # --- Reverse the user's direction ---
    user.mobility.destination_direction = not user.mobility.destination_direction
    if user.mobility.destination_direction:
        # If now moving forward
        if user.user_type == "pedestrian":
            # Pedestrian: set destination to the start of the path
            user.mobility.destination = apply_height_randomness(path_points[0], user)
        else:
            # Vehicle: switch to the opposite lane and set destination to the start
            new_lane = road_info[:-1] + ("0" if road_info.endswith("1") else "1")
            user.mobility.current_path = [road_name, new_lane]
            path_points_new = current_road_data.get(new_lane, [])
            user.mobility.destination = apply_height_randomness(path_points_new[0], user)
    else:
        # If now moving backward
        if user.user_type == "pedestrian":
            # Pedestrian: set destination to the end of the path
            user.mobility.destination = apply_height_randomness(path_points[-1], user)
        else:
            # Vehicle: switch to the opposite lane and set destination to the end
            new_lane = road_info[:-1] + ("1" if road_info.endswith("0") else "0")
            user.mobility.current_path = [road_name, new_lane]
            path_points_new = current_road_data.get(new_lane, [])
            user.mobility.destination = apply_height_randomness(path_points_new[-1], user)

def _set_next_road_destination(current_road_name, next_road_name, next_road_data, user):
    """
    Set destination on a newly found next road.
    
    Args:
        current_road_name (str): Name of current road
        next_road_name (str): Name of next road
        next_road_data (dict): Next road data
        user (User): User object
    
    Returns:
        np.ndarray: Destination coordinate or None if error
    """
    # --- Determine which direction to take on the new road ---
    if current_road_name in next_road_data.get("adjacent0", []):
        user.mobility.destination_direction = True
        road_info, path_points = choose_lane(next_road_data, user)
        # If there are at least two points, move to the second; otherwise, use the first
        user.destination = apply_height_randomness(
            path_points[1] if len(path_points) > 1 else path_points[0], user
        )
    elif current_road_name in next_road_data.get("adjacent1", []):
        user.mobility.destination_direction = False
        road_info, path_points = choose_lane(next_road_data, user)
        # If there are at least two points, move to the second-to-last; otherwise, use the last
        user.destination = apply_height_randomness(
            path_points[-2] if len(path_points) > 1 else path_points[-1], user
        )
    else:
        # If the current road is not adjacent, return None
        return None
    # --- Update the user's current path ---
    user.mobility.current_path = [next_road_name, road_info]
    return user.mobility.destination

def find_next_road(road_data, road_list, user):
    """
    Determine the next road and set the user's destination.
    
    Args:
        road_data (dict): Current road data dictionary
        road_list (list): List of all road data tuples
        user (User): User object
    """
    current_road_name = user.mobility.current_path[0]
    # --- Try to find a valid neighboring road ---
    next_road_name, next_road_data = _find_valid_neighbor_road(road_data, road_list, user.mobility.destination_direction)
    if next_road_data is None:
        # If no valid neighbor, perform a U-turn
        _handle_uturn(road_data, user)
    else:
        # Otherwise, set the destination on the next road
        _set_next_road_destination(current_road_name, next_road_name, next_road_data, user)

def choose_lane(next_road_data, user):
    """
    Select an appropriate lane for the user on the next road based on user type and road material.
    
    Args:
        next_road_data (dict): Next road data dictionary
        user (User): User object with type and direction information
    
    Returns:
        tuple: (road_info, path_points) - Selected lane name and its path points
    """
    user_type = user.user_type
    destination_direction = user.mobility.destination_direction
    road_info = ""
    path_points = []
    material = next_road_data.get("material")
    allowed_types = MATERIAL_USER_TYPE.get(material, [])
    # --- Lane selection logic based on user type and allowed types ---
    if user_type == "pedestrian":
        if allowed_types == ["pedestrian"] or "pedestrian" in allowed_types:
            # Prefer outer lanes for pedestrians if available
            outer_lanes = [info for info in ["road_outer_0", "road_outer_1"] if next_road_data.get(info) is not None]
            if outer_lanes:
                road_info = random.choice(outer_lanes)
                path_points = next_road_data.get(road_info, [])
            # If no outer lanes, try inner lanes
            if not path_points and (allowed_types == ["pedestrian"] or allowed_types == ["pedestrian", "vehicle"]):
                inner_lanes = [info for info in ["road_inner_0", "road_inner_1"] if next_road_data.get(info) is not None]
                if inner_lanes:
                    road_info = random.choice(inner_lanes)
                    path_points = next_road_data.get(road_info, [])
    elif user_type == "vehicle":
        if allowed_types == ["vehicle"] or "vehicle" in allowed_types:
            # Prefer inner lanes for vehicles if available
            inner_lanes = [info for info in ["road_inner_0", "road_inner_1"] if next_road_data.get(info) is not None]
            if inner_lanes:
                target_lane_info = "road_inner_0" if destination_direction else "road_inner_1"
                if target_lane_info in inner_lanes:
                    road_info = target_lane_info
                    path_points = next_road_data.get(road_info, [])
                else:
                    road_info = random.choice(inner_lanes)
                    path_points = next_road_data.get(road_info, [])
            # If no inner lanes, try outer lanes
            if not path_points and (allowed_types == ["vehicle"] or allowed_types == ["pedestrian", "vehicle"]):
                outer_lanes = [info for info in ["road_outer_0", "road_outer_1"] if next_road_data.get(info) is not None]
                if outer_lanes:
                    road_info = random.choice(outer_lanes)
                    path_points = next_road_data.get(road_info, [])
    return road_info, path_points

def apply_height_randomness(point, user):
    """
    Apply random spatial offset and user height to a 3D point.
    
    Args:
        point (list or np.ndarray): 3D point coordinates
        user (User): User object with height information
    
    Returns:
        np.ndarray: Modified 3D point with randomness and height applied
    """
    point_np = np.array(point).reshape(3, 1)
    # --- Add random offset in XY and Z directions ---
    random_offset = np.array([
        random.uniform(*RANDOM_OFFSET_RANGE_XY),
        random.uniform(*RANDOM_OFFSET_RANGE_XY),
        random.uniform(*RANDOM_OFFSET_RANGE_Z)
    ]).reshape(3, 1)
    # --- Add user height to the Z coordinate ---
    height_offset = np.array([0, 0, user.mobility.height]).reshape(3, 1)
    return point_np + random_offset + height_offset