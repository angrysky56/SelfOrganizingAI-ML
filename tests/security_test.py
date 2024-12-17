"""
Security Integration Test Suite
Tests end-to-end security implementation with real SSL certificates.

Usage:
    1. Generate certificates: python scripts/generate_certs.py
    2. Run tests: pytest tests/security_test.py -v
"""

import pytest
import asyncio
import ssl
import aiohttp
import jwt
from datetime import datetime, timedelta
from pathlib import Path
from fastapi.testclient import TestClient
from src.core.security.middleware import SecurityMiddleware
from src.api.server import app

# Test Configuration
BASE_URL = "https://localhost:8443"
TEST_TOKEN_SECRET = "test_secret_key_12345"
SSL_DIR = Path(__file__).parent.parent / "config" / "ssl"

@pytest.fixture
def ssl_context():
    """Create SSL context for testing."""
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

@pytest.fixture
def valid_token():
    """Generate valid JWT token."""
    return jwt.encode(
        {
            'sub': 'test_user',
            'exp': datetime.utcnow() + timedelta(hours=1)
        },
        TEST_TOKEN_SECRET,
        algorithm='HS256'
    )

@pytest.fixture
def expired_token():
    """Generate expired JWT token."""
    return jwt.encode(
        {
            'sub': 'test_user',
            'exp': datetime.utcnow() - timedelta(hours=1)
        },
        TEST_TOKEN_SECRET,
        algorithm='HS256'
    )

async def test_ssl_connection(ssl_context):
    """Test SSL connection and protocol enforcement."""
    async with aiohttp.ClientSession() as session:
        # Test HTTPS connection
        async with session.get(
            f"{BASE_URL}/health",
            ssl=ssl_context
        ) as response:
            assert response.status == 200
            
        # Test HTTP connection (should fail)
        with pytest.raises(aiohttp.ClientError):
            async with session.get(
                f"http://localhost:8000/health"
            ) as response:
                pass

async def test_security_headers(ssl_context):
    """Test security header injection."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{BASE_URL}/health",
            ssl=ssl_context
        ) as response:
            headers = response.headers
            assert 'Strict-Transport-Security' in headers
            assert 'Content-Security-Policy' in headers
            assert 'X-Frame-Options' in headers
            assert headers['X-Frame-Options'] == 'DENY'

async def test_rate_limiting(ssl_context):
    """Test rate limiting functionality."""
    async with aiohttp.ClientSession() as session:
        # Make multiple requests
        responses = []
        for _ in range(150):  # Exceeds default rate limit
            async with session.get(
                f"{BASE_URL}/health",
                ssl=ssl_context
            ) as response:
                responses.append(response.status)
                
        # Should see some 429 responses
        assert 429 in responses

async def test_authentication(ssl_context, valid_token, expired_token):
    """Test JWT authentication."""
    async with aiohttp.ClientSession() as session:
        # Test valid token
        async with session.get(
            f"{BASE_URL}/protected",
            ssl=ssl_context,
            headers={'Authorization': f'Bearer {valid_token}'}
        ) as response:
            assert response.status == 200
            
        # Test expired token
        async with session.get(
            f"{BASE_URL}/protected",
            ssl=ssl_context,
            headers={'Authorization': f'Bearer {expired_token}'}
        ) as response:
            assert response.status == 401

def test_metrics_collection():
    """Test security metrics collection."""
    client = TestClient(app)
    
    # Make some requests to generate metrics
    for _ in range(5):
        client.get("/health", verify=False)
        
    # Check metrics
    response = client.get("/metrics", verify=False)
    metrics = response.text
    
    assert 'http_requests_total' in metrics
    assert 'request_duration_seconds' in metrics
    assert 'active_connections' in metrics

if __name__ == '__main__':
    pytest.main([__file__, '-v'])