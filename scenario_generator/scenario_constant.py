# State Constants
BIT_GENERATING = 1  # User is generating bits
BUCKET_REMAINING = 2  # User has remaining bits in bucket
BUCKET_EMPTY = 0  # User's bucket is empty

# Physical Constants
PACKET_SIZE_MIN = 64 * 8   # 64 bytes = 512 bits
PACKET_SIZE_MAX = 1500 * 8 # 1500 bytes = 12000 bits
MIN_DISTANCE_BETWEEN_USERS = 1.0 # meters
MOVE_THRESHOLD = 1.5 # meters

# System Constants
USER_GENERATION_RETRY_COUNT = 10
INITIAL_POPULATION_RATIO = 0.8
PEDESTRIAN_VEHICLE_RATIO = 0.7

# Mobility Utils Constants
RANDOM_OFFSET_RANGE_XY = (-0.2, 0.2) # meters
RANDOM_OFFSET_RANGE_Z = (-0.05, 0.05) # meters
