import numpy as np
import random
from ..UE_initializer import initialize_user_traffic

# Handle imports for both direct execution and module import
try:
    # When imported as a module
    from ...scenario_constant import BIT_GENERATING, BUCKET_REMAINING, BUCKET_EMPTY
except ImportError:
    # When executed directly
    from scenario_constant import BIT_GENERATING, BUCKET_REMAINING, BUCKET_EMPTY

def initialize(users):
    """
    Initialize traffic for users.
    
    Args:
        users (list): List of user objects
    
    Returns:
        list: List of initialized user objects
    """
    # Initialize traffic individually for each user
    for user in users:
        initialize_user_traffic(user)
    return users

def update_users(users, throughput, time):
    """
    Update traffic for users.
    
    Args:
        users (list): List of user objects
        throughput (list): List of throughput values for each user (int)
    
    Returns:
        list: List of updated user objects
    """
    for i, user in enumerate(users):
        # Check user state
        if user.traffic.traffic_state == BIT_GENERATING:
            # When state is 1 (generating)
            # 1. Fill the bit bucket using the stored arrival process
            if user.traffic.arrival_process is not None:
                try:
                    generated_bits = user.traffic.arrival_process.generate_packets(time)
                    user.traffic.traffic_bit_bucket += generated_bits
                    user.traffic.traffic_accumulated += generated_bits
                except Exception as e:
                    print(f"Error generating packets for user {i}: {e}")
                    user.traffic.traffic_bit_bucket = 0
            
            # 2. Consume from bucket using throughput
            user_throughput = int(throughput[i])
            _consume_bucket(user, user_throughput)
            
            # 3. Compare accumulated with bucket and change state to 2 if bucket is smaller
            if user.traffic.total_traffic_bits < user.traffic.traffic_accumulated:
                user.traffic.traffic_state = BUCKET_REMAINING
        
        elif user.traffic.traffic_state == BUCKET_REMAINING:
            # When state is 2 (queued)
            # Only execute bucket consumption
            user_throughput = int(throughput[i])
            _consume_bucket(user, user_throughput)
            
            # If bucket becomes 0, change state to 0
            if user.traffic.traffic_bit_bucket == 0:
                user.traffic.traffic_state = BUCKET_EMPTY
        
        else:
            # If state is 0 or other value, do nothing
            pass


def _consume_bucket(user, throughput):
    """
    Internal function to consume throughput amount from user's bucket
    
    Args:
        user (User): user object
        throughput (int): amount of bits to consume
    
    Returns:
        None: user object is modified directly
    """
    if user.traffic.traffic_bit_bucket is None:
        user.traffic.traffic_bit_bucket = 0
    
    # Consume throughput amount from bucket (set to 0 if negative)
    user.traffic.traffic_bit_bucket = max(0, user.traffic.traffic_bit_bucket - throughput)