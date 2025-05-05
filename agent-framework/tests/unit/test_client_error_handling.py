import pytest
from unittest.mock import patch, MagicMock
import requests
from prometheus_swarm.clients.base_client import BaseClient
from prometheus_swarm.clients.openai_client import OpenAIClient
from prometheus_swarm.clients.anthropic_client import AnthropicClient
from prometheus_swarm.clients.ollama_client import OllamaClient
from prometheus_swarm.utils.errors import APIError, RateLimitError, AuthenticationError

def test_base_client_error_handling():
    """Test base client error handling mechanisms."""
    base_client = BaseClient()

    # Test network error handling
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(APIError, match="Network connection error"):
            base_client._make_api_request('test_url', {}, 'POST')

    # Test timeout error handling
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(APIError, match="Request timed out"):
            base_client._make_api_request('test_url', {}, 'POST')

def test_client_rate_limit_handling():
    """Test rate limit error handling across different clients."""
    clients = [
        OpenAIClient(),
        AnthropicClient(),
        OllamaClient()
    ]

    for client in clients:
        with patch.object(client, '_make_api_request') as mock_request:
            mock_request.side_effect = RateLimitError("Rate limit exceeded")
            
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                client.generate_text("Test prompt")

def test_client_authentication_error():
    """Test authentication error scenarios."""
    clients = [
        OpenAIClient(),
        AnthropicClient(),
        OllamaClient()
    ]

    for client in clients:
        with patch.object(client, '_make_api_request') as mock_request:
            mock_request.side_effect = AuthenticationError("Invalid API key")
            
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                client.generate_text("Test prompt")

def test_client_invalid_input_handling():
    """Test handling of invalid input across clients."""
    clients = [
        OpenAIClient(),
        AnthropicClient(),
        OllamaClient()
    ]

    for client in clients:
        # Test empty prompt
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.generate_text("")

        # Test non-string prompt
        with pytest.raises(TypeError, match="Prompt must be a string"):
            client.generate_text(123)

def test_boundary_conditions():
    """Test various boundary condition scenarios."""
    clients = [
        OpenAIClient(),
        AnthropicClient(),
        OllamaClient()
    ]

    for client in clients:
        # Test extremely long prompts
        long_prompt = "x" * 100000  # Very long prompt
        with pytest.raises(ValueError, match="Prompt exceeds maximum length"):
            client.generate_text(long_prompt)

def test_client_response_validation():
    """Test validation of API responses."""
    clients = [
        OpenAIClient(),
        AnthropicClient(),
        OllamaClient()
    ]

    for client in clients:
        # Mock a malformed response
        with patch.object(client, '_make_api_request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {}  # Empty response
            mock_request.return_value = mock_response

            with pytest.raises(APIError, match="Invalid API response"):
                client.generate_text("Test prompt")