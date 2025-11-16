import numpy as np
import random
from .user import User
from .mobility.mobility_utils import determine_destination, choose_lane, apply_height_randomness, calculate_distance, calculate_vector_magnitude

# Handle imports for both direct execution and module import
try:
    # When imported as a module
    from ..scenario_config import (PEDESTRIAN_HEIGHT_RANGE, PEDESTRIAN_SPEED_RANGE, VEHICLE_HEIGHT_RANGE, 
                          VEHICLE_SPEED_RANGE, MIN_DISTANCE_BETWEEN_USERS,
                          POISSON_LAMBDA_RATE_RANGE, EXPONENTIAL_RATE_RANGE, 
                          UNIFORM_MIN_RATE_RANGE, UNIFORM_MAX_RATE_RANGE,
                          TOTAL_TRAFFIC_BITS_MIN, TOTAL_TRAFFIC_BITS_MAX,
                          TRAFFIC_TIMEOUT_MIN, TRAFFIC_TIMEOUT_MAX)
    from ..scenario_constant import BIT_GENERATING
except ImportError:
    # When executed directly
    from scenario_config import (PEDESTRIAN_HEIGHT_RANGE, PEDESTRIAN_SPEED_RANGE, VEHICLE_HEIGHT_RANGE, 
                          VEHICLE_SPEED_RANGE, MIN_DISTANCE_BETWEEN_USERS,
                          POISSON_LAMBDA_RATE_RANGE, EXPONENTIAL_RATE_RANGE, 
                          UNIFORM_MIN_RATE_RANGE, UNIFORM_MAX_RATE_RANGE,
                          TOTAL_TRAFFIC_BITS_MIN, TOTAL_TRAFFIC_BITS_MAX,
                          TRAFFIC_TIMEOUT_MIN, TRAFFIC_TIMEOUT_MAX)
    from scenario_constant import BIT_GENERATING

def initialize_user_mobility(user_type, road_list, user_positions, rt_name):
    """
    Initialize a user (pedestrian or vehicle).
    
    Args:
        user_type (str): "pedestrian" or "vehicle"
        road_list (list): List of road data
        user_positions (list): List of existing user positions
    
    Returns:
        tuple: (User, np.ndarray) generated user and coordinate or (None, None)
    """
    # --- Create user object and set type ---
    user = User()
    user.user_type = user_type

    # --- Randomly assign initial direction ---
    # (True - index ascending, False - index descending)
    user.mobility.destination_direction = random.choice([True, False])

    # --- Select a random road and lane for the user ---
    road_name, road_data = random.choice(road_list)
    road_info, path_points = choose_lane(road_data, user)
    user.mobility.current_path = [road_name, road_info]

    # --- Set height and speed based on user type ---
    if user_type == "pedestrian":
        # Pedestrians get height and speed from pedestrian range
        height = random.uniform(*PEDESTRIAN_HEIGHT_RANGE)
        speed = random.uniform(*PEDESTRIAN_SPEED_RANGE)
    elif user_type == "vehicle":
        # Vehicles get height and speed from vehicle range
        height = random.uniform(*VEHICLE_HEIGHT_RANGE)
        speed = random.uniform(*VEHICLE_SPEED_RANGE)
    else:
        # If user_type is invalid, return None
        return None, None
    user.mobility.height = height

    # --- If no path points are available, fail to generate user ---
    if not path_points:
        return None, None

    # --- Randomly select a location along the path ---
    idx = random.randint(0, len(path_points)-2)
    point1 = np.array(path_points[idx])
    point2 = np.array(path_points[idx+1])
    ratio = random.uniform(0.3, 0.7)
    point = point1 + ratio * (point2 - point1)
    coordinate = apply_height_randomness(point, user)

    # --- Check if the new user is too close to existing users ---
    for pos in user_positions:
        if calculate_distance(coordinate, pos) < MIN_DISTANCE_BETWEEN_USERS:
            # If too close, fail to generate user
            return None, None
    user.mobility.coordinate = coordinate

    # --- Determine the user's destination ---
    determine_destination(road_list, user)
    if user.mobility.destination is None:
        # If no valid destination, fail to generate user
        return None, None

    # --- Calculate direction and velocity ---
    direction = user.mobility.destination - user.mobility.coordinate
    if calculate_vector_magnitude(direction) == 0:
        # If the user is already at the destination, set orientation and velocity to zero
        orientation = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        velocity = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    else:
        # Otherwise, normalize direction and set velocity
        orientation = direction / calculate_vector_magnitude(direction)
        velocity = orientation * speed
    user.mobility.orientation = orientation
    user.mobility.velocity = velocity
    user.mobility.time_stack = 0

    ## PKL update method call
    user.pkl_parser(rt_name)

    return user, coordinate

def generate_traffic_parameters(traffic_type):
    """
    Generate random parameters for the specified traffic type
    
    Args:
        traffic_type (str): traffic type ('poisson', 'exponential', 'uniform')
    
    Returns:
        dict: dictionary containing the generated parameters as keyword arguments
    """
    if traffic_type == 'poisson':
        # Generate random lambda_rate for Poisson process
        lambda_rate = random.uniform(*POISSON_LAMBDA_RATE_RANGE)  # bits/sec
        return {'lambda_rate': lambda_rate}
    
    elif traffic_type == 'exponential':
        # Generate random rate for Exponential process
        rate = random.uniform(*EXPONENTIAL_RATE_RANGE)  # bits/sec
        return {'rate': rate}
    
    elif traffic_type == 'uniform':
        # Generate random min_rate and max_rate for Uniform process
        min_rate = random.uniform(*UNIFORM_MIN_RATE_RANGE)  # bits/sec
        max_rate = random.uniform(*UNIFORM_MAX_RATE_RANGE)  # bits/sec
        return {'min_rate': min_rate, 'max_rate': max_rate}
    
    else:
        # Default to Poisson if unknown traffic type
        lambda_rate = random.uniform(*POISSON_LAMBDA_RATE_RANGE)  # bits/sec
        return {'lambda_rate': lambda_rate}

def initialize_user_traffic(user):
    """
    Initialize traffic state for a single user.
    Args:
        user (User): user object
    Returns:
        User: initialized user object
    """
    # Import here to avoid circular dependency
    from .traffic.arrival_process import ArrivalProcessSetter
    
    # 1. Set traffic_state first
    user.traffic.traffic_state = BIT_GENERATING
    user.communication_success = True

    # 2. Randomly determine traffic timeout (100ms ~ 1s)
    user.traffic.traffic_timeout = random.uniform(TRAFFIC_TIMEOUT_MIN, TRAFFIC_TIMEOUT_MAX)

    # 3. Randomly determine total bit amount for user (100 kbits ~ 1 Mbit)
    user.traffic.total_traffic_bits = random.randint(TOTAL_TRAFFIC_BITS_MIN, TOTAL_TRAFFIC_BITS_MAX)

    # 4. Randomly choose arrival process type and generate parameters
    traffic_type = random.choice(['poisson', 'exponential', 'uniform'])
    traffic_params = generate_traffic_parameters(traffic_type)
    
    # 5. Create arrival process with generated parameters
    ArrivalProcessSetter.create_user_arrival_process(user, traffic_type, **traffic_params)

    # 6. Generate initial traffic bits through arrival process to fill bucket (for 10ms)
    initial_bits = user.traffic.arrival_process.generate_packets(10)  # Generate bits for 10ms
    user.traffic.traffic_bit_bucket = initial_bits
    user.traffic.traffic_accumulated = initial_bits

    return user
