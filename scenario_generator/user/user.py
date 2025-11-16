import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:  # Avoid circular imports at runtime
    from ..BS.BS import BS  # noqa: F401

import os
import json

@dataclass
class Mobility:
    """
    Mobility attributes for user movement and positioning
    """
    destination: Optional[np.ndarray] = None
    destination_direction: Optional[bool] = None
    current_path: Optional[List[str]] = None
    height: Optional[float] = None
    coordinate: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    time_stack: Optional[float] = None

    def __post_init__(self):
        """Initialize numpy arrays after object creation"""
        if self.coordinate is not None and not isinstance(self.coordinate, np.ndarray):
            self.coordinate = np.array(self.coordinate).reshape(3, 1)
        if self.velocity is not None and not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity).reshape(3, 1)
        if self.orientation is not None and not isinstance(self.orientation, np.ndarray):
            self.orientation = np.array(self.orientation).reshape(3, 1)
        if self.destination is not None and not isinstance(self.destination, np.ndarray):
            self.destination = np.array(self.destination).reshape(3, 1)
    
    # Custom setter for coordinate to ensure numpy array format
    def set_coordinate(self, value):
        """
        Set coordinate with automatic numpy array conversion
        
        Args:
            value (list, tuple, or np.ndarray): Coordinate values (x, y, z)
        """
        self.coordinate = np.array(value).reshape(3, 1)
    
    # Custom setter for velocity to ensure numpy array format
    def set_velocity(self, value):
        """
        Set velocity with automatic numpy array conversion
        
        Args:
            value (list, tuple, or np.ndarray): Velocity values (x, y, z)
        """
        self.velocity = np.array(value).reshape(3, 1)
    
    # Custom setter for orientation to ensure numpy array format
    def set_orientation(self, value):
        """
        Set orientation with automatic numpy array conversion
        
        Args:
            value (list, tuple, or np.ndarray): Orientation values (x, y, z)
        """
        self.orientation = np.array(value).reshape(3, 1)
    
    # Custom setter for destination to ensure numpy array format
    def set_destination(self, value):
        """
        Set destination with automatic numpy array conversion
        
        Args:
            value (list, tuple, or np.ndarray): Destination values (x, y, z)
        """
        self.destination = np.array(value).reshape(3, 1)

@dataclass
class Traffic:
    """
    Traffic attributes for user communication and traffic management
    """
    total_traffic_bits: Optional[int] = None
    traffic_bit_bucket: Optional[int] = None
    traffic_accumulated: Optional[int] = None
    traffic_state: Optional[int] = None
    arrival_process: Optional[object] = None
    traffic_timeout: Optional[float] = None

@dataclass
class User:
    """
    User class representing a pedestrian or vehicle in the simulation
    
    Attributes:
        user_id: Unique identifier for the user
        user_type: User type ("pedestrian" or "vehicle")
        mobility: Mobility attributes (destination, coordinate, velocity, etc.)
        traffic: Traffic attributes (traffic bits, state, etc.)
    """
    # User attributes
    user_id: Optional[str] = None
    user_type: Optional[str] = None
    pkl: Optional[str] = None
    connected_bs: Optional["BS"] = None
    
    # Monitoring attributes
    communication_success: Optional[bool] = None

    # Component attributes
    mobility: Optional[Mobility] = None
    traffic: Optional[Traffic] = None

    def __post_init__(self):
        """Initialize component objects if not provided"""
        if self.mobility is None:
            self.mobility = Mobility()
        if self.traffic is None:
            self.traffic = Traffic()

    def connect_to_bs(self, bs: "BS"):
        """Connects the UE to a BS."""
        if self.connected_bs:
            self.disconnect_from_bs()
        self.connected_bs = bs
        bs.add_ue(self)

    def disconnect_from_bs(self):
        """Disconnects the UE from its BS."""
        if self.connected_bs:
            self.connected_bs.remove_ue(self)
            self.connected_bs = None

    #Slot-level association without snapshot index
    def cell_association_from_h_slot(self, ue_idx: int, h_slot: np.ndarray, bs_list: List["BS"]) -> Optional["BS"]:
        """Associate this UE to the BS with the largest average channel power.

        Parameters
        ----------
        ue_idx : int
            Index of this UE in the h_slot tensor first dimension
        h_slot : np.ndarray
            CFR for the current slot with shape
            [num_ut, num_ut_ant, num_bs, num_bs_ant, num_ofdm_symbols, num_subcarriers]
        bs_list : list[BS]
            List of candidate base stations

        Returns
        -------
        BS | None
            Selected BS (and connects), or None if bs_list is empty
        """
        if h_slot is None or len(bs_list) == 0:
            return None
        num_bs = h_slot.shape[2]
        gains = []
        for bs_idx in range(num_bs):
            # [num_ut_ant, num_bs_ant, Nsym, Nsc]
            h_slice = h_slot[ue_idx, :, bs_idx, :, :, :]
            gain = np.mean(np.abs(h_slice) ** 2)
            gains.append(gain)
        best_idx = int(np.argmax(gains)) if len(gains) > 0 else None
        best_bs = bs_list[best_idx] if best_idx is not None else None
        if best_bs is not None:
            self.connect_to_bs(best_bs)
        return best_bs
    
    def pkl_parser(self, RT_name: str) -> bool:
        """
        Find PKL file matching user's current state and save to user.pkl
        
        Args:
            RT_name (str): Base path where PKL files are stored
            
        Returns:
            bool: True if PKL file found, False otherwise
        """
        
        # Check required attributes
        if (self.mobility is None or 
            self.mobility.current_path is None or 
            len(self.mobility.current_path) < 1 or
            self.mobility.coordinate is None or
            self.mobility.orientation is None):
            return False
        
        # Step 1: Find road folder
        road_name = self.mobility.current_path[0]  # e.g., "sinchonlo"
        # Remove .ply extension
        if road_name.endswith('.ply'):
            road_name = road_name[:-4]  # Remove .ply
        home_dir = os.path.expanduser("~")
        pkl_path = os.path.join("/home/mimox/250704_RT_map_paths", "paths")
        road_folder = os.path.join(pkl_path, road_name)
        
        # Check if folder exists
        if not os.path.exists(road_folder):
            return False
        
        # Step 2-1: Find idx matching user position from coordinates.json
        coords_file = os.path.join(road_folder, f"{road_name}_coordinates.json")
        if not os.path.exists(coords_file):
            return False
        
        try:
            with open(coords_file, 'r') as f:
                coordinates_data = json.load(f)
        except Exception as e:
            return False
        
        # Find closest idx based on user position (compare x, y only)
        user_position = self.mobility.coordinate.flatten()
        user_position_2d = user_position[:2]  # Extract x, y only
        closest_idx = None
        min_distance = float('inf')
        
        for coord_entry in coordinates_data:
            entry_position = np.array(coord_entry['position'])
            entry_position_2d = entry_position[:2]  # Extract x, y only
            distance = np.linalg.norm(user_position_2d - entry_position_2d)
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = coord_entry['index']
        
        # If position matching fails
        if closest_idx is None:
            return False
        
        # Step 2-2: Find closest direction based on orientation
        # Define cardinal direction vectors (same as get_orientation_for_direction in worker_utils.py)
        cardinal_directions = {
            1: np.array([0.0, 1.0, 0.0]),   # North
            2: np.array([1.0, 0.0, 0.0]),   # East  
            3: np.array([0.0, -1.0, 0.0]),  # South
            4: np.array([-1.0, 0.0, 0.0])   # West
        }
        
        # Find direction most similar to user orientation
        user_orientation = self.mobility.orientation.flatten()
        closest_direction = 1  # Default value
        max_similarity = -1
        
        for direction, direction_vector in cardinal_directions.items():
            # Calculate similarity using dot product (dot product of two unit vectors = cos(angle))
            similarity = np.dot(user_orientation, direction_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                closest_direction = direction
        
        # Step 3: Generate and save PKL file path
        pkl_filename = f"rx_{closest_idx}_{closest_direction}.pkl"
        pkl_path = os.path.join(road_folder, pkl_filename)
        
        # Check if PKL file exists
        if os.path.exists(pkl_path):
            self.pkl = pkl_path
            return True
        else:
            return False 