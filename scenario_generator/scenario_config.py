# Map Configuration
DEFAULT_MAP_NAME = "250312_mobility_map"
DEFAULT_RT_NAME = "250704_RT_map"

# Population Configuration
POPULATION_UPDATE_INTERVAL = 1.0 * 1000 # milli seconds
DEFAULT_MAX_POPULATION = 200 # 500 

# Default attributes for user data
DEFAULT_ATTRIBUTES = ['user_type', 'mobility.coordinate', 'mobility.velocity', 'mobility.orientation']

# User Type Configuration
PEDESTRIAN_HEIGHT_RANGE = (0.9, 1.1) # meters
PEDESTRIAN_SPEED_RANGE = (1.1, 1.7) # m/s (approx. 4-6 km/h)
VEHICLE_HEIGHT_RANGE = (0.7, 0.8) # meters
VEHICLE_SPEED_RANGE = (13.9, 19.4) # m/s (approx. 50-70 km/h)
MIN_DISTANCE_BETWEEN_USERS = 1.0 # meters

# Traffic Configuration
POISSON_LAMBDA_RATE_RANGE = (1000, 10000)   # bits/sec (1-10 kbps)
EXPONENTIAL_RATE_RANGE = (500, 5000)        # bits/sec (0.5-5 kbps)
UNIFORM_MIN_RATE_RANGE = (100, 1000)        # bits/sec (0.1-1 kbps)
UNIFORM_MAX_RATE_RANGE = (2000, 10000)      # bits/sec (2-10 kbps)
TOTAL_TRAFFIC_BITS_MIN = 100_000_000   # 100_000   # 100 kbits
TOTAL_TRAFFIC_BITS_MAX = 1_000_000_000 # 1_000_000 # 1 Mbit 
TRAFFIC_TIMEOUT_MIN = 100   # 100ms
TRAFFIC_TIMEOUT_MAX = 1000  # 1s

# Mapping from material to allowed user types
MATERIAL_USER_TYPE = {
    # Only pedestrian
    "mat-paths_footway": ["pedestrian"],
    "mat-paths_steps": ["pedestrian"],
    "mat-roads_pedestrian": ["pedestrian"],
    # Only vehicle
    "mat-roads_primary": ["vehicle"],
    "mat-roads_secondary": ["vehicle"],
    # Pedestrian + vehicle
    "mat-roads_tertiary": ["pedestrian", "vehicle"],
    "mat-roads_residential": ["pedestrian", "vehicle"],
    "mat-roads_service": ["pedestrian", "vehicle"],
}
