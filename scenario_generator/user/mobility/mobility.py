import numpy as np
import random
from .mobility_utils import get_road_types, determine_destination, calculate_distance, calculate_vector_magnitude
from ..UE_initializer import initialize_user_mobility

# Handle imports for both direct execution and module import
try:
    # When imported as a module
    from ...scenario_constant import INITIAL_POPULATION_RATIO, PEDESTRIAN_VEHICLE_RATIO, USER_GENERATION_RETRY_COUNT, MOVE_THRESHOLD
except ImportError:
    # When executed directly
    from scenario_constant import INITIAL_POPULATION_RATIO, PEDESTRIAN_VEHICLE_RATIO, USER_GENERATION_RETRY_COUNT, MOVE_THRESHOLD

def _calculate_movement_distance(user, time):
    """
    Calculate the total distance a user should move based on time and velocity.
    
    Args:
        user (User): User object
        time (float): Time in milliseconds
    
    Returns:
        float: Total distance to move
    """
    # --- Accumulate time and calculate movement distance ---
    if not hasattr(user.mobility, 'time_stack') or user.mobility.time_stack is None:
        user.mobility.time_stack = 0
    user.mobility.time_stack += time
    velocity_m_per_ms = user.mobility.velocity / 3.6 / 1000  # Convert km/h to m/ms
    total_distance = calculate_vector_magnitude(velocity_m_per_ms * user.mobility.time_stack)
    return total_distance

def _process_movement_segments(user, total_distance, road_list):
    """
    Process movement by breaking it into segments and updating user position.
    
    Args:
        user (User): User object
        total_distance (float): Total distance to move
        road_list (list): List of road data
    """
    # --- Divide movement into segments to avoid large jumps ---
    num_segments = int(np.ceil(total_distance / MOVE_THRESHOLD))
    remaining_distance = total_distance - ((num_segments - 1) * MOVE_THRESHOLD)

    for i in range(num_segments):
        # For the last segment, use the remaining distance
        if i == num_segments - 1 and remaining_distance == 0:
            continue
        segment_distance = MOVE_THRESHOLD if i < num_segments - 1 else remaining_distance
        movement_direction = user.mobility.velocity / calculate_vector_magnitude(user.mobility.velocity)
        movement = movement_direction * segment_distance
        _update_with_movement(user, movement, road_list)

def _calculate_initial_population_counts(max_population):
    """
    Calculate initial population counts for pedestrians and vehicles.
    
    Args:
        max_population (int): Maximum population size
    
    Returns:
        tuple: (initial_population, pedestrian_count, vehicle_count)
    """
    # --- Determine how many users to generate for each type ---
    initial_population = int(max_population * INITIAL_POPULATION_RATIO)
    pedestrian_count = int(initial_population * PEDESTRIAN_VEHICLE_RATIO)
    vehicle_count = initial_population - pedestrian_count
    return initial_population, pedestrian_count, vehicle_count

def _generate_users_of_type(user_type, count, road_list, user_positions, start_idx, RT_name):
    """
    Generate users of a specific type.
    
    Args:
        user_type (str): Type of user ("pedestrian" or "vehicle")
        count (int): Number of users to generate
        road_list (list): List of road data
        user_positions (list): List of existing user positions
        start_idx (int): Starting index for user IDs
    
    Returns:
        tuple: (generated_users, user_positions, next_idx)
    """
    generated_users = []
    user_idx = start_idx
    for _ in range(count):
        # Try multiple times to generate a valid user (avoid collisions, etc.)
        for attempt in range(USER_GENERATION_RETRY_COUNT):
            if not road_list:
                # If there are no available roads, skip
                continue
            user, coordinate = initialize_user_mobility(user_type, road_list, user_positions, RT_name)
            if user:
                # If user is successfully generated, assign ID and add to list
                user.user_id = f"{user_type}_{user_idx}"
                generated_users.append(user)
                user_positions.append(coordinate)
                user_idx += 1
                break  # Stop retrying for this user
    return generated_users

def _update_with_movement(user, movement, road_list):
    """
    Update user position by receiving movement vector.
    
    Args:
        user (User): User object
        movement (np.ndarray): Movement vector (3x1)
        road_list (list): Road list
    """
    # --- Move the user by the given movement vector ---
    new_coordinate = user.mobility.coordinate + movement
    current_to_dest = user.mobility.destination - user.mobility.coordinate
    current_to_new = new_coordinate - user.mobility.coordinate
    user.mobility.coordinate = new_coordinate
    # --- If the user has reached or passed the destination, assign a new destination ---
    if calculate_vector_magnitude(current_to_new) >= calculate_vector_magnitude(current_to_dest):
        determine_destination(road_list, user)
        direction = user.mobility.destination - user.mobility.coordinate
        orientation = direction / calculate_vector_magnitude(direction)
        velocity = orientation * calculate_vector_magnitude(user.mobility.velocity)
        user.mobility.orientation = orientation
        user.mobility.velocity = velocity

def initialize(map_data, max_population, user_positions=[], pedestrian_idx=0, vehicle_idx=0, RT_name=None):
    """
    Initialize initial users based on map data and max population.
    
    Args:
        map_data (dict): Map data
        max_population (int): Maximum population
    
    Returns:
        tuple: (users, pedestrian_roads, vehicle_roads)
    """
    # --- Calculate how many users to generate for each type ---
    initial_population, pedestrian_count, vehicle_count = _calculate_initial_population_counts(max_population)
    users = []
    
    # --- Get available roads for each user type ---
    pedestrian_roads, vehicle_roads = get_road_types(map_data)
    # --- Generate pedestrian users ---
    generated_pedestrians = _generate_users_of_type("pedestrian", pedestrian_count, pedestrian_roads, user_positions, pedestrian_idx, RT_name)
    users.extend(generated_pedestrians)
    # --- Generate vehicle users ---
    generated_vehicles = _generate_users_of_type("vehicle", vehicle_count, vehicle_roads, user_positions, vehicle_idx, RT_name)
    users.extend(generated_vehicles)
    print(f"Generated {len(users)} users: {sum(1 for u in users if u.user_type=='pedestrian')} pedestrians, {sum(1 for u in users if u.user_type=='vehicle')} vehicles")
    return users, pedestrian_roads, vehicle_roads
    
def update_user(users, time, pedestrian_roads, vehicle_roads, RT_name):
    """
    Update user position by receiving time.
    
    Args:
        user (User): User object
        time (float): Time in milliseconds
        pedestrian_roads (list): Pedestrian roads
        vehicle_roads (list): Vehicle roads
    """
    # --- Update each user's position ---
    for user in users:
        # --- Calculate how far the user should move in this update ---
        total_distance = _calculate_movement_distance(user, time)
        if total_distance < MOVE_THRESHOLD:
            # If the movement is too small, skip updating
            continue
        # --- Select the appropriate road list based on user type ---
        if user.user_type == "pedestrian":
            road_list = pedestrian_roads
        else:
            road_list = vehicle_roads
        # --- Move the user in segments ---
        _process_movement_segments(user, total_distance, road_list)
        # --- Update PKL file after movement ---
        user.pkl_parser(RT_name)
        # --- Reset the time stack after movement ---
        user.mobility.time_stack = 0