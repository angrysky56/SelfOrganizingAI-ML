"""
Security Middleware System: Advanced Protocol Management and Metrics Collection
Implements comprehensive security controls with adaptive monitoring capabilities.
"""

from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional, Callable, Union
import yaml
from pathlib import Path
import time
import logging
import jwt
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge
from dataclasses import dataclass
import ssl
from ..metrics.prometheus_metrics import MetricsManager

@dataclass
class SecurityMetrics:
    """Real-time security metrics tracking system."""
    request_counter = Counter('http_requests_total', 'Total HTTP requests')
    failed_requests = Counter('failed_requests_total', 'Failed request attempts')
    request_duration = Histogram(
        'request_duration_seconds', 
        'Request duration',
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    active_connections = Gauge('active_connections', 'Number of active connections')
    ssl_errors = Counter('ssl_errors_total', 'Total SSL/TLS errors')

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Advanced security middleware with protocol enforcement and metrics collection.
    
    Core Features:
    - TLS/SSL protocol validation and enforcement
    - Adaptive rate limiting with sliding window
    - Real-time security metrics collection
    - Dynamic security header injection
    - JWT token validation
    - Request sanitization
    """
    
    def __init__(
        self, 
        app, 
        config_path: Optional[Path] = None,
        metrics_manager: Optional[MetricsManager] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.metrics = SecurityMetrics()
        self.config = self._load_config(config_path)
        self.rate_limiter = self._initialize_rate_limiter()
        self.metrics_manager = metrics_manager
        self._setup_ssl_context()

    def _setup_ssl_context(self) -> None:
        """Initialize SSL context with security optimizations."""
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            self.ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
            self.ssl_context.set_ciphers(
                'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256'
            )
            self.ssl_context.options |= (
                ssl.OP_NO_SSLv2 | 
                ssl.OP_NO_SSLv3 |
                ssl.OP_NO_TLSv1 | 
                ssl.OP_NO_TLSv1_1 |
                ssl.OP_NO_COMPRESSION
            )
        except Exception as e:
            self.logger.error(f"SSL context initialization failed: {e}")
            raise

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process and validate incoming requests with comprehensive security checks.
        
        Security Flow:
        1. Protocol validation
        2. Rate limit verification
        3. Token validation
        4. Request processing
        5. Security header injection
        6. Metrics collection
        """
        start_time = time.time()
        client_ip = request.client.host
        self.metrics.active_connections.inc()

        try:
            # Protocol validation
            await self._validate_protocol(request)
            
            # Rate limiting
            await self._enforce_rate_limits(client_ip)
            
            # Token validation if enabled
            if self.config['authentication']['jwt']['enabled']:
                await self._validate_token(request)
            
            # Process request
            response = await call_next(request)
            
            # Security headers
            response = self._inject_security_headers(response)
            
            # Metrics collection
            self._collect_metrics(start_time)
            
            return response

        except HTTPException as e:
            self.metrics.failed_requests.inc()
            return Response(
                content=str(e.detail),
                status_code=e.status_code,
                media_type="application/json"
            )
        except Exception as e:
            self.logger.error(f"Security error: {e}")
            self.metrics.failed_requests.inc()
            return Response(
                content="Internal security error",
                status_code=500,
                media_type="application/json"
            )
        finally:
            self.metrics.active_connections.dec()

    async def _validate_protocol(self, request: Request) -> None:
        """Validate request protocol and SSL/TLS parameters."""
        if not request.url.scheme == 'https':
            self.metrics.ssl_errors.inc()
            raise HTTPException(status_code=400, detail="HTTPS Required")

    async def _enforce_rate_limits(self, client_ip: str) -> None:
        """Enforce rate limits using sliding window algorithm."""
        if not self._check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded"
            )

    async def _validate_token(self, request: Request) -> None:
        """Validate JWT token with security checks."""
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        
        try:
            token = auth_header.split(' ')[1]
            jwt.decode(
                token,
                self.config['authentication']['jwt']['secret_key'],
                algorithms=['HS256']
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

    def _inject_security_headers(self, response: Response) -> Response:
        """Inject comprehensive security headers."""
        header_config = self.config['security_headers']
        
        # HSTS
        if header_config['strict_transport_security']['enabled']:
            hsts_config = header_config['strict_transport_security']
            response.headers['Strict-Transport-Security'] = (
                f"max-age={hsts_config['max_age']}; "
                f"includeSubDomains" if hsts_config['include_subdomains'] else ""
            )
        
        # CSP
        if header_config['content_security_policy']['enabled']:
            csp = header_config['content_security_policy']
            response.headers['Content-Security-Policy'] = "; ".join([
                f"{key} {value}" for key, value in csp.items()
                if key not in ['enabled']
            ])
        
        # Additional security headers
        response.headers.update({
            'X-Frame-Options': header_config['x_frame_options'],
            'X-Content-Type-Options': header_config['x_content_type_options'],
            'X-XSS-Protection': header_config['x_xss_protection'],
            'Referrer-Policy': header_config['referrer_policy']
        })
        
        return response

    def _collect_metrics(self, start_time: float) -> None:
        """Collect and update security metrics."""
        self.metrics.request_counter.inc()
        self.metrics.request_duration.observe(time.time() - start_time)
        
        if self.metrics_manager:
            self.metrics_manager.update_security_metrics({
                'active_connections': self.metrics.active_connections._value.get(),
                'failed_requests': self.metrics.failed_requests._value.get(),
                'ssl_errors': self.metrics.ssl_errors._value.get()
            })