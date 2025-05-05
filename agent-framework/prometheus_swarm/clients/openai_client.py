"""OpenAI API client implementation with comprehensive error handling."""

from typing import Dict, Any, Optional, List, Union
import requests
from openai import OpenAI
from .base_client import Client
from ..types import (
    ToolDefinition,
    MessageContent,
    TextContent,
    ToolCallContent,
    ToolChoice,
)
from ..utils.errors import (
    APIError, 
    RateLimitError, 
    AuthenticationError, 
    NetworkError, 
    InputValidationError
)
import json

class OpenAIClient(Client):
    """OpenAI API client with enhanced error handling."""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        
        # Validate API key
        if not api_key:
            raise AuthenticationError("API key cannot be empty")
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,  # Use default OpenAI URL if not specified
            )
            self.default_headers = default_headers
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize OpenAI client: {str(e)}")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Public method for generating text with comprehensive validation."""
        # Input validation
        if not isinstance(prompt, str):
            raise InputValidationError("Prompt must be a string")
        
        if not prompt.strip():
            raise InputValidationError("Prompt cannot be empty")
        
        if len(prompt) > 10000:  # Arbitrary maximum length
            raise InputValidationError("Prompt exceeds maximum length")

        try:
            response = self._make_api_call([{"role": "user", "content": prompt}])
            return response.content
        except requests.exceptions.ConnectionError:
            raise NetworkError("Unable to connect to OpenAI API")
        except requests.exceptions.Timeout:
            raise APIError("Request timed out")
        except Exception as e:
            # Generic error handling
            if "429" in str(e):
                raise RateLimitError()
            elif "401" in str(e):
                raise AuthenticationError("Invalid OpenAI API key")
            else:
                raise APIError(f"OpenAI API error: {str(e)}")

    # ... [rest of the original implementation remains the same] ...