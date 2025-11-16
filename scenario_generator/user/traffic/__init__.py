# Import only the main functions, avoid circular dependencies
from .traffic import initialize, update_users

__all__ = ['initialize', 'update_users']
