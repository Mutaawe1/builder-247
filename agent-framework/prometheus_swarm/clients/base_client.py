"""Base client for LLM API implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import importlib.util
from .conversation_manager import ConversationManager
from ..types import (
    ToolDefinition,
    MessageContent,
    ToolCall,
    ToolChoice,
    ToolCallContent,
)
from prometheus_swarm.utils.logging import log_section, log_key_value, log_error
from prometheus_swarm.utils.errors import (
    APIError, 
    RateLimitError, 
    AuthenticationError, 
    NetworkError, 
    InputValidationError
)
from prometheus_swarm.utils.retry import (
    is_retryable_error,
    send_message_with_retry,
    execute_tool_with_retry,
)
import json
import ast

# Alias the most generic API error for backwards compatibility
ClientAPIError = APIError

class Client(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        model: Optional[str] = None,
    ):
        """Initialize the client."""
        self.storage = ConversationManager()
        self.model = model or self._get_default_model()
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.api_name = self._get_api_name()

    # ... [rest of the code remains the same]

    def make_api_call(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make API call with error handling."""
        try:
            # Build kwargs based on what the specific client implementation supports
            kwargs = {
                "messages": messages,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "tools": tools,
                "tool_choice": tool_choice,
            }

            # Only pass extra_headers to OpenAI-based clients
            if extra_headers:
                kwargs["extra_headers"] = extra_headers

            return self._make_api_call(**kwargs)
        except (RateLimitError, AuthenticationError, NetworkError) as e:
            # Preserve specific error types
            raise
        except Exception as e:
            # Wrap non-specific errors
            log_error(
                e,
                context=f"Error making API call to {self.api_name}",
                include_traceback=not is_retryable_error(e),
            )
            raise APIError(str(e))

    # ... [rest of the code remains the same]