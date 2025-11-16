# Import only the main functions, avoid circular dependencies
from .mobility import initialize, update_user

__all__ = ['initialize', 'update_user']
