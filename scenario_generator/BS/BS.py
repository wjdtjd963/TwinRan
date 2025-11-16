from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING, Any, Dict
import numpy as np
import os
import pandas as pd

if TYPE_CHECKING:  # Avoid hard dependency at import time
    from ..user.user import User  # noqa: F401
    
def BS_initializer(map_name: str) -> List[BS]:
    """
    Initialize multiple BS objects from Excel file based on map_name.
    
    Args:
        map_name (str): Name of the map to load BS coordinates for
        
    Returns:
        List of initialized BS objects
    """
    # The location of BS.py is Y-Twin/scenario_generator/BS/BS.py.
    # Construct the path to the BS coordinates Excel file.
    # Target file: Y-Twin/map/map_lists/<map_name>/<map_name>_BS_coord.xlsx
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Y-Twin/
    bs_file_path = os.path.join(project_root, "map", "map_lists", map_name, f"{map_name}_BS_coord.xlsx")
    
    try:
        # Load BS coordinates from Excel file
        if not os.path.isfile(bs_file_path):
            print(f"Warning: BS coord file not found: {bs_file_path}")
            # Return default BS at origin with height 20m
            return [BS(bs_id="bs0", coordinate=[0.0, 0.0, 20.0], orientation=[0.0, 0.0, 0.0])]
        
        df = pd.read_excel(bs_file_path)
        expected_cols = ['X', 'Y', 'Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
        
        # Check if all expected columns exist
        for col in expected_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in {bs_file_path}")
                # Return default BS at origin with height 20m
                return [BS(bs_id="bs0", coordinate=[0.0, 0.0, 20.0], orientation=[0.0, 0.0, 0.0])]
        
        # Filter to only expected columns and convert to records
        df = df[expected_cols]
        coords = df.to_dict(orient='records')
        
        # Create BS objects from coordinates
        bs_list = []
        for i, coord in enumerate(coords):
            pos = [coord.get('X', 0.0), coord.get('Y', 0.0), coord.get('Z', 0.0)]
            ori = [coord.get('Ori_X', 0.0), coord.get('Ori_Y', 0.0), coord.get('Ori_Z', 0.0)]
            bs_list.append(BS(bs_id=f"bs{i}", coordinate=pos, orientation=ori))
        
        print(f"Successfully loaded {len(bs_list)} BS objects from {bs_file_path}")
        return bs_list
        
    except Exception as e:
        print(f"Error loading BS coordinates from {bs_file_path}: {e}")
        # Return default BS at origin with height 20m
        return [BS(bs_id="bs0", coordinate=[0.0, 0.0, 20.0], orientation=[0.0, 0.0, 0.0])]

@dataclass
class BS:
    """Base Station description used by the scenario generator.

    Attributes
    ----------
    bs_id : str | None
        Unique identifier for the base station
    coordinate : np.ndarray | None
        3D coordinate (x, y, z) in meters, shaped [3, 1]
    orientation : np.ndarray | None
        3D orientation (roll, pitch, yaw) in radians, shaped [3, 1]
    connected_ues : list[User]
        Users currently associated with this BS
    """

    bs_id: Optional[str] = None
    coordinate: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    connected_ues: List["User"] = field(default_factory=list)

    def __post_init__(self):
        if self.coordinate is not None and not isinstance(self.coordinate, np.ndarray):
            self.coordinate = np.array(self.coordinate, dtype=float).reshape(3, 1)
        if self.orientation is not None and not isinstance(self.orientation, np.ndarray):
            self.orientation = np.array(self.orientation, dtype=float).reshape(3, 1)




