import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Handle imports for both direct execution and module import
try:
    # When imported as a module
    from .user import mobility, traffic
    from .server_utils import load_map_data, get_user_attributes
    from .scenario_config import DEFAULT_MAX_POPULATION, DEFAULT_ATTRIBUTES, POPULATION_UPDATE_INTERVAL, DEFAULT_MAP_NAME, DEFAULT_RT_NAME
    from .scenario_constant import BUCKET_EMPTY
    from .BS import BS_initializer
except ImportError:
    # When executed directly
    from user import mobility, traffic
    from server_utils import load_map_data, get_user_attributes
    from scenario_config import DEFAULT_MAX_POPULATION, DEFAULT_ATTRIBUTES, POPULATION_UPDATE_INTERVAL, DEFAULT_MAP_NAME, DEFAULT_RT_NAME
    from scenario_constant import BUCKET_EMPTY
    from BS import BS_initializer

@dataclass
class ScenarioState:
    """
    Scenario state class to hold simulation data
    
    Attributes:
        users: List of User objects in the simulation
        map_data: Map data dictionary
        map_name: Name of the current map
        pedestrian_roads: List of pedestrian road data
        vehicle_roads: List of vehicle road data
    """
    users: List[Any] = None
    inactive_users: List[Any] = None
    map_data: Optional[Dict[str, Any]] = None
    map_name: Optional[str] = None
    RT_name: Optional[str] = None
    pedestrian_roads: List[Any] = None
    vehicle_roads: List[Any] = None
    elapsed_time: float = None
    population_timer: float = None
    bs_list: List[Any] = None
    max_population: int = DEFAULT_MAX_POPULATION
    
    def __post_init__(self):
        # Initialize default values for lists if not provided
        if self.users is None:
            self.users = []
        if self.pedestrian_roads is None:
            self.pedestrian_roads = []
        if self.vehicle_roads is None:
            self.vehicle_roads = []
        if self.bs_list is None:
            self.bs_list = []

def filter_users(users, inactive_users, elapsed_time):
    """
    Filter users based on traffic state and communication success
    
    Args:
        users (list): List of current users (each user is stored as a dictionary)
        inactive_users (list): List of inactive users (each user is stored as a dictionary)
        elapsed_time (float): Elapsed time of the simulation
    """
    i = 0
    while i < len(users):
        if elapsed_time > users[i].traffic.traffic_timeout:
            if users[i].communication_success:
                users[i].communication_success = False

        if users[i].traffic.traffic_state == BUCKET_EMPTY:
            inactive_users.append(users.pop(i))
        else: i += 1

def add_users(state):
    """
    Update the population of the scenario.
    
    Args:
        users (list): List of current users (each user is stored as a dictionary)
        map_data
    
    Returns:
        list: List of new users

    """
    original_user_count = len(state.users)

    if original_user_count >= state.max_population: return []
    else:
        new_user_num = (state.max_population - original_user_count) * random.uniform(0.5, 1.0)
        if new_user_num == 0: return []
        # --- Initialize users and road lists using mobility module ---
        user_positions = []
        pedestrian_idx = 0
        vehicle_idx = 0
        for user in state.users:
            # Get user coordinate, Find last idx of user
            user_positions.append(user.mobility.coordinate)
            if user.user_type == 'pedestrian': pedestrian_idx += 1
            elif user.user_type == 'vehicle': vehicle_idx += 1
        for inactive_user in state.inactive_users:
            if inactive_user.user_type == 'pedestrian': pedestrian_idx += 1
            elif inactive_user.user_type == 'vehicle': vehicle_idx += 1
        users, _, _ = mobility.initialize(state.map_data, new_user_num, user_positions, pedestrian_idx, vehicle_idx, RT_name=state.RT_name)
        # --- Initialize user traffic states ---
        users = traffic.initialize(users)
        for user in users:
            user.traffic.traffic_timeout += state.elapsed_time

        return users


def initialize_scenario_standalone(
    map_name: str,
    RT_name: str,
    max_population: int = DEFAULT_MAX_POPULATION,
    requested_attributes: list[str] = DEFAULT_ATTRIBUTES
):
    """
    Initialize the scenario state based on map data and population.
    
    Args:
        map_name (str): Name of the map to load
        max_population (int): Maximum number of users to generate
        requested_attributes (list[str]): List of user attributes to return
    
    Returns:
        tuple: (response_dict, scenario_state) or (None, None) if failed
    """
    # --- Create scenario state object and set map name ---
    scenario_state = ScenarioState()
    scenario_state.map_name = map_name
    scenario_state.RT_name = RT_name
    scenario_state.elapsed_time = 0.0
    scenario_state.population_timer = POPULATION_UPDATE_INTERVAL
    scenario_state.max_population = max_population  
    # --- Load map data from file ---
    map_data = load_map_data(scenario_state.map_name)
    if not map_data:
        # If map loading fails, return None
        print(f"Error: Failed to load map: {scenario_state.map_name}")
        return None, None
    scenario_state.map_data = map_data

    # --- Initialize users and road lists using mobility module ---
    scenario_state.users, scenario_state.pedestrian_roads, scenario_state.vehicle_roads = mobility.initialize(scenario_state.map_data, scenario_state.max_population, RT_name=scenario_state.RT_name)
    scenario_state.inactive_users = []
    # --- Initialize user traffic states ---
    scenario_state.users = traffic.initialize(scenario_state.users)

    # --- Initialize BS list using ---
    scenario_state.bs_list = BS_initializer(scenario_state.map_name)
    
    # --- Prepare response data with requested user attributes ---
    response_data = []
    for user in scenario_state.users:
        user_data = get_user_attributes(user, requested_attributes)
        response_data.append(user_data)

    return {
        "status": "initialized",
        "map_name": scenario_state.map_name,
        "user_count": len(scenario_state.users),
        "bs_count": len(scenario_state.bs_list),
        "users": response_data
    }, scenario_state

def update_scenario_standalone(
    scenario_state: ScenarioState,
    time: float,
    requested_attributes: list[str] = DEFAULT_ATTRIBUTES,
    throughput: list[int] = None
):
    """
    Update the scenario state based on time and return updated user attributes.
    
    Args:
        scenario_state (ScenarioState): Current scenario state
        time (float): Time to advance the simulation
        requested_attributes (list[str]): List of user attributes to return
        throughput (list): List of throughput values for each user (int)
    
    Returns:
        list: List of updated user data dictionaries or None if failed
    """
    # --- Check if scenario state and map data are valid ---
    if not scenario_state or not scenario_state.map_data:
        # If state is not initialized, return None
        print("Error: Scenario state not initialized or map data missing")
        return None
    if time is None:
        # If time is not provided, return None
        print("Error: Time parameter is required")
        return None

    # --- Update each user's position using the mobility module ---
    scenario_state.elapsed_time += time
    mobility.update_user(scenario_state.users, time, scenario_state.pedestrian_roads, scenario_state.vehicle_roads, scenario_state.RT_name)

    # traffic update
    traffic.update_users(scenario_state.users, throughput, time)

    # move inactive users to inactive_users list
    filter_users(scenario_state.users, scenario_state.inactive_users, scenario_state.elapsed_time)

    if scenario_state.elapsed_time < scenario_state.population_timer: pass
    else:
        scenario_state.population_timer += POPULATION_UPDATE_INTERVAL
        # Generate new users
        scenario_state.users.extend(add_users(scenario_state))

    # --- Prepare response data with requested user attributes ---
    response_data = []
    for user in scenario_state.users:
        user_data = get_user_attributes(user, requested_attributes)
        response_data.append(user_data)

    return response_data, scenario_state


def monitoring_scenario(state):
    users = state.users
    inactive_users = state.inactive_users

    print("--------------------------------")
    print("Monitoring Scenario")
    print("-----")

    print("Users whose communication has ended:")
    for user in inactive_users:
        if user.communication_success:
            print(f"{user.user_id}: O")
        else:
            print(f"{user.user_id}: X")

    print("-----")
    print("Users whose communication is ongoing:")
    for user in users:
        if user.communication_success:
            print(f"{user.user_id}: {user.traffic.traffic_state}, {user.traffic.traffic_accumulated}, {user.traffic.traffic_bit_bucket}")
        else:
            print(f"{user.user_id}: {user.traffic.traffic_state}, {user.traffic.traffic_accumulated}, {user.traffic.traffic_bit_bucket}, X")
    print("--------------------------------")

    print("pkl test")
    for user in users:
        # Extract position and direction information from PKL file
        pkl_info = "None"
        if user.pkl:
            try:
                import pickle
                with open(user.pkl, 'rb') as f:
                    pkl_data = pickle.load(f)
                
                # Find rx_position and direction information from the last item in PKL file
                if isinstance(pkl_data, list) and len(pkl_data) > 0:
                    last_item = pkl_data[-1]
                    if isinstance(last_item, dict):
                        rx_position = last_item.get('rx_position', 'N/A')
                        direction = last_item.get('direction', 'N/A')
                        pkl_info = f"pos{rx_position}_dir{direction}"
                    else:
                        pkl_info = "no_metadata"
                else:
                    pkl_info = "empty_or_invalid"
            except Exception as e:
                pkl_info = f"error: {str(e)[:20]}"
        
        print(f"{user.user_id}: {user.pkl} ({pkl_info})")
        print(f"{user.user_id}: {user.mobility.coordinate} , {user.mobility.orientation}, {user.mobility.current_path[0]}")



if __name__ == "__main__":
    # --- Example usage: initialize and update scenario ---
    population_timer = 0.0
    map_name = DEFAULT_MAP_NAME
    RT_name = DEFAULT_RT_NAME
    initial_result, current_state = initialize_scenario_standalone(map_name, RT_name)

    if initial_result:
        print("Scenario initialized successfully:")
        print(f"Map Name: {initial_result['map_name']}")
        print(f"User Count: {initial_result['user_count']}")
        print(f"BS Count: {initial_result['bs_count']}")
        print("--------------------------------")
        # print("Initial Users data:", initial_result['users'])  # Uncomment to print initial user data

        for _ in range(1000):
            time_to_update = 5
            population_timer += time_to_update
            print(f"Current time: {population_timer} [ms]")

            throughput = [random.randint(4000000, 8000000)*time_to_update for _ in range(len(current_state.users))]

            updated_users_data, current_state = update_scenario_standalone(current_state, time_to_update, DEFAULT_ATTRIBUTES, throughput)

            if updated_users_data:
                print(f"\nScenario updated to time {time_to_update}:")
                print(f"Successfully updated positions for {len(updated_users_data)} users.")
            else:
                print("Scenario update failed.")
            
            monitoring_scenario(current_state)
            a = input("Press Enter to continue...")
    else:
        print("Scenario initialization failed.") 