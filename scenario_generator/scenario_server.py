from flask import Flask, request, jsonify, current_app
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Handle imports for both direct execution and module import
try:
    # When imported as a module
    from .user import mobility, traffic
    from .server_utils import load_map_data, get_user_attributes
    from .scenario_config import DEFAULT_MAX_POPULATION, DEFAULT_ATTRIBUTES
    from .BS import BS_initializer
except ImportError:
    # When executed directly
    from user import mobility, traffic
    from server_utils import load_map_data, get_user_attributes
    from scenario_config import DEFAULT_MAX_POPULATION, DEFAULT_ATTRIBUTES
    from BS import BS_initializer

app = Flask(__name__)

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
        bs_list: List of BS objects
    """
    users: List[Any] = None
    map_data: Optional[Dict[str, Any]] = None
    map_name: Optional[str] = None
    pedestrian_roads: List[Any] = None
    vehicle_roads: List[Any] = None
    bs_list: List[Any] = None
    
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

@app.route('/initialize', methods=['POST'])
def initialize():
    """
    Initialize scenario generator by receiving map name.
    
    Expected JSON payload:
        map_name (str): Name of the map to load
        max_population (int, optional): Maximum number of users (default: 500)
        attributes (list[str], optional): List of user attributes to return
    
    Returns:
        JSON response with initialization status and user data
    """
    # --- Create a new scenario state object ---
    current_app.scenario_state = ScenarioState()
    state = current_app.scenario_state

    # --- Parse input data from request ---
    data = request.json
    state.map_name = data.get('map_name')
    max_population = data.get('max_population', DEFAULT_MAX_POPULATION)
    requested_attributes = data.get('attributes', DEFAULT_ATTRIBUTES)

    # --- Check if map name is provided ---
    if not state.map_name:
        # If map name is missing, return error
        return jsonify({"error": "Map name is required"}), 400

    # --- Load map data from file ---
    state.map_data = load_map_data(state.map_name)
    if not state.map_data:
        # If map loading fails, clean up and return error
        del current_app.scenario_state
        return jsonify({"error": f"Failed to load map: {state.map_name}"}), 404

    # --- Initialize users and road lists using mobility module ---
    state.users, state.pedestrian_roads, state.vehicle_roads = mobility.initialize(state.map_data, max_population)
    # --- Initialize user traffic states ---
    state.users = traffic.initialize(state.users)
    
    # --- Initialize BS list using BS_initializer ---
    state.bs_list = BS_initializer(state.map_name)

    # --- Prepare response data with requested user attributes ---
    response_data = []
    for user in state.users:
        user_data = get_user_attributes(user, requested_attributes)
        response_data.append(user_data)

    return jsonify({
        "status": "initialized",
        "map_name": state.map_name,
        "user_count": len(state.users),
        "bs_count": len(state.bs_list),
        "users": response_data
    }), 200

@app.route('/update', methods=['POST'])
def update():
    """
    Update user positions by receiving time.
    
    Expected JSON payload:
        time (float): Time to advance the simulation
        attributes (list[str], optional): List of user attributes to return
    
    Returns:
        JSON response with updated user data
    """
    # --- Retrieve the scenario state from the app context ---
    state = getattr(current_app, 'scenario_state', None)

    # --- Check if scenario state and map data are valid ---
    if not state or not state.map_data:
        # If state is not initialized, return error
        return jsonify({"error": "Server not initialized or map data missing"}), 400

    # --- Parse input data from request ---
    data = request.json
    time = data.get('time')
    requested_attributes = data.get('attributes', DEFAULT_ATTRIBUTES)

    # --- Check if time parameter is provided ---
    if time is None:
        # If time is missing, return error
        return jsonify({"error": "Time parameter is required"}), 400

    # --- Update each user's position using the mobility module ---
    for user in state.users:
        mobility.update_user(user, time, state.pedestrian_roads, state.vehicle_roads)

    # --- Prepare response data with requested user attributes ---
    response_data = []
    for user in state.users:
        user_data = get_user_attributes(user, requested_attributes)
        response_data.append(user_data)

    return jsonify(response_data), 200

if __name__ == '__main__':
    # --- Run the Flask app ---
    app.run(debug=True, host='0.0.0.0', port=5000)