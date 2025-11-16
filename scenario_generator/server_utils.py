import json
import os
import numpy as np

def get_user_class_attributes():
    """
    Get all valid user attributes from the User class using introspection.
    
    Returns:
        list: List of attribute names that are dataclass fields in the User class
    """
    from dataclasses import fields
    try:
        from .user import User
    except ImportError:
        try:
            from user import User
        except ImportError:
            from scenario_generator.user import User
    
    dataclass_fields = [field.name for field in fields(User)]
    # --- Exclude private fields (those starting with '_') ---
    return [field for field in dataclass_fields if not field.startswith('_')]

def load_map_data(map_name):
    """
    Load JSON file of specified map name.
    
    Args:
        map_name (str): Map name
    
    Returns:
        dict: Map data or None if file not found
    """
    # --- Build the absolute path to the map file ---
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_path = os.path.join(workspace_root, "map", "map_lists", map_name, f"{map_name}_extracted.json")
    try:
        # --- Try to open and load the map file ---
        with open(map_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # --- If the file does not exist, print error and return None ---
        print(f"Error: Map file {map_path} not found")
        return None

def get_user_attributes(user, requested_attributes):
    """
    Get requested attributes from a user object.
    
    Args:
        user (User): User object
        requested_attributes (list): List of attribute names to get
    
    Returns:
        dict: Dictionary containing requested attributes
    """
    # --- Get the list of valid attributes from the User class ---
    valid_attributes = get_user_class_attributes()
    result = {}
    entered_loop = False
    try:
        for attr in requested_attributes:
            entered_loop = True
            if attr not in valid_attributes:
                result[attr] = None
                continue
            try:
                # Handle nested attribute access (e.g., "mobility.coordinate", "traffic.traffic_state")
                if '.' in attr:
                    # Split the attribute path (e.g., "mobility.coordinate" -> ["mobility", "coordinate"])
                    attr_parts = attr.split('.')
                    current_obj = user
                    
                    # Navigate through the nested attributes
                    for part in attr_parts:
                        if hasattr(current_obj, part):
                            current_obj = getattr(current_obj, part)
                        else:
                            current_obj = None
                            break
                    
                    value = current_obj
                else:
                    # Direct attribute access
                    value = getattr(user, attr)
                
                if isinstance(value, np.ndarray):
                    value = value.flatten().tolist()
                result[attr] = value
            except AttributeError:
                result[attr] = None
    except Exception as e:
        print('Exception during for loop over requested_attributes:', e)
    if not entered_loop:
        print('The for loop over requested_attributes did not execute.')
    return result

def check_invalid_attributes(requested_attributes):
    """
    Check for invalid attributes in the request.
    
    Args:
        requested_attributes (list): List of requested attributes
    
    Returns:
        list: List of invalid attributes
    """
    # --- Get the list of valid attributes from the User class ---
    valid_attributes = get_user_class_attributes() 
    # --- Return attributes that are not valid ---
    return [attr for attr in requested_attributes if attr not in valid_attributes] 
