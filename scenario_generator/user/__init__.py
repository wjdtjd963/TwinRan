# Import only the main classes, not functions that cause circular imports
from .user import User, Mobility, Traffic

# Import mobility and traffic modules
from . import mobility
from . import traffic

__all__ = ['User', 'Mobility', 'Traffic', 'mobility', 'traffic']
