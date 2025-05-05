"""Comprehensive Error Handling for API Clients."""

class APIError(Exception):
    """Base class for API-related errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code
        super().__init__(message)

class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)

class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)

class NetworkError(APIError):
    """Raised for network-related connectivity issues."""
    def __init__(self, message: str = "Network connection error"):
        super().__init__(message, status_code=503)

class InputValidationError(ValueError):
    """Raised for invalid input parameters."""
    pass