import numpy as np
import random

# Handle imports for both direct execution and module import
try:
    # When imported as a module
    from ...scenario_config import (POISSON_LAMBDA_RATE_RANGE, EXPONENTIAL_RATE_RANGE, 
                          UNIFORM_MIN_RATE_RANGE, UNIFORM_MAX_RATE_RANGE)
    from ...scenario_constant import PACKET_SIZE_MIN, PACKET_SIZE_MAX
except ImportError:
    # When executed directly
    from scenario_config import (POISSON_LAMBDA_RATE_RANGE, EXPONENTIAL_RATE_RANGE, 
                          UNIFORM_MIN_RATE_RANGE, UNIFORM_MAX_RATE_RANGE)
    from scenario_constant import PACKET_SIZE_MIN, PACKET_SIZE_MAX

class ArrivalProcessSetter:
    """
    Class that creates arrival process objects with randomly generated parameters based on traffic type
    """
    @staticmethod
    def create_user_arrival_process(user, traffic_type, **kwargs):
        """
        Create arrival process object based on user and traffic_type and store it in user
        
        Args:
            user (User): user object
            traffic_type (str): arrival process type ('poisson', 'exponential', 'uniform', etc.)
            **kwargs: additional parameters for each arrival process type (optional)
        
        Returns:
            None: user object is modified directly
        """
        if traffic_type == 'poisson':
            # Set Poisson-specific parameters
            lambda_rate = kwargs.get('lambda_rate', random.uniform(*POISSON_LAMBDA_RATE_RANGE))  # bits/sec
            user.traffic.arrival_process = PoissonProcess(lambda_rate)
        
        elif traffic_type == 'exponential':
            # Set Exponential-specific parameters
            rate = kwargs.get('rate', random.uniform(*EXPONENTIAL_RATE_RANGE))  # bits/sec
            user.traffic.arrival_process = ExponentialProcess(rate)
        
        elif traffic_type == 'uniform':
            # Set Uniform-specific parameters
            min_rate = kwargs.get('min_rate', random.uniform(*UNIFORM_MIN_RATE_RANGE))  # bits/sec
            max_rate = kwargs.get('min_rate', random.uniform(*UNIFORM_MAX_RATE_RANGE))  # bits/sec
            user.traffic.arrival_process = UniformProcess(min_rate, max_rate)
        
        else:
            # Use Poisson as default
            lambda_rate = kwargs.get('lambda_rate', random.uniform(*POISSON_LAMBDA_RATE_RANGE))
            user.traffic.arrival_process = PoissonProcess(lambda_rate)

class BaseArrivalProcess:
    """
    Base class for all arrival process classes
    """
    def __init__(self):
        self.base_duration = 5  # base time unit (milliseconds)
    
    def generate_packet_size(self):
        """
        Generate random packet size (64~1500 bytes)
        Returns:
            int: packet size (bits)
        """
        return random.randint(PACKET_SIZE_MIN, PACKET_SIZE_MAX)
    
    def generate_packets_for_base_duration(self):
        """
        Generate packets for base time unit (5ms) - must be overridden by subclasses
        Returns:
            int: total generated bits
        """
        raise NotImplementedError("Subclasses must implement generate_packets_for_base_duration()")
    
    def generate_packets(self, total_duration_ms):
        """
        Generate packets for total duration (divided into base units and repeated)
        
        Args:
            total_duration_ms (float): total simulation time (milliseconds)
        
        Returns:
            int: total generated bits
        """
        total_bits = 0
        num_iterations = int(total_duration_ms / self.base_duration)
        
        for _ in range(num_iterations):
            total_bits += self.generate_packets_for_base_duration()
        
        return total_bits

class PoissonProcess(BaseArrivalProcess):
    """
    Poisson packet arrival process class
    
    Parameters:
        lambda_rate (float): bit arrival rate (bits/sec)
    """
    def __init__(self, lambda_rate):
        super().__init__()
        self.lambda_rate = lambda_rate
    
    def generate_packets_for_base_duration(self):
        """
        Generate Poisson packets for base time unit (5ms)
        Returns:
            int: total generated bits
        """
        current_time = 0
        bits = 0
        
        while current_time < self.base_duration:
            # Time until next packet arrival follows exponential distribution
            inter_arrival_time = np.random.exponential(scale=1.0 / self.lambda_rate)
            current_time += inter_arrival_time
            
            if current_time < self.base_duration:
                # Packet arrived, so add random size bits
                bits += self.generate_packet_size()
                
        return bits

class ExponentialProcess(BaseArrivalProcess):
    """
    Exponential packet arrival process class
    
    Parameters:
        rate (float): bit arrival rate (bits/sec)
    """
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
    
    def generate_packets_for_base_duration(self):
        """
        Generate Exponential packets for base time unit (5ms)
        Returns:
            int: total generated bits
        """
        current_time = 0
        bits = 0
        
        while current_time < self.base_duration:
            # Time until next packet arrival follows exponential distribution
            inter_arrival_time = np.random.exponential(scale=1.0 / self.rate)
            current_time += inter_arrival_time
            
            if current_time < self.base_duration:
                # Packet arrived, so add random size bits
                bits += self.generate_packet_size()
                
        return bits

class UniformProcess(BaseArrivalProcess):
    """
    Uniform packet arrival process class
    
    Parameters:
        min_rate (float): minimum bit arrival rate (bits/sec)
        max_rate (float): maximum bit arrival rate (bits/sec)
    """
    def __init__(self, min_rate, max_rate):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
    
    def generate_packets_for_base_duration(self):
        """
        Generate Uniform packets for base time unit (5ms)
        Returns:
            int: total generated bits
        """
        current_time = 0
        bits = 0
        
        while current_time < self.base_duration:
            # Randomly select current arrival rate from uniform distribution
            current_rate = random.uniform(self.min_rate, self.max_rate)
            
            # Time until next packet arrival follows exponential distribution
            inter_arrival_time = np.random.exponential(scale=1.0 / current_rate)
            current_time += inter_arrival_time
            
            if current_time < self.base_duration:
                # Packet arrived, so add random size bits
                bits += self.generate_packet_size()
                
        return bits