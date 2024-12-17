# Adaptive Simulation Server with Enhanced Security Controls
# Implements secure API endpoints with protocol enforcement

import uvicorn
import logging
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ..config.ssl import SSLConfig

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Enforces security protocols and connection requirements"""
    
    async def dispatch(self, request: Request, call_next):
        # Protocol validation
        if request.url.scheme != "https":
            return JSONResponse(
                status_code=400,
                content=self._generate_error_response("HTTPS Required")
            )
            
        # Add security headers
        response = await call_next(request)
        return self._enhance_response_security(response)
        
    def _generate_error_response(self, error_type: str) -> Dict:
        """Generate structured error information"""
        return {
            "error": error_type,
            "message": "Connection must use HTTPS protocol",
            "requirements": {
                "protocol": "HTTPS",
                "min_tls": "TLS 1.2",
                "recommended_tls": "TLS 1.3"
            }
        }
        
    def _enhance_response_security(self, response):
        """Apply security headers to response"""
        headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        for header, value in headers.items():
            response.headers[header] = value
        return response

class SimulationServer:
    """Core server implementation with security integration"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Self-Organizing AI ML System",
            description="Secure API for AI simulation and analysis",
            version="1.0.0",
            docs_url=None,  # Disable on non-HTTPS
            redoc_url=None  # Disable on non-HTTPS
        )
        
        self.ssl_config = SSLConfig()
        self._configure_middleware()
        self._register_routes()
        
    def _configure_middleware(self):
        """Setup security middleware and CORS"""
        self.app.add_middleware(SecurityMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://*"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        
    def _register_routes(self):
        """Configure API endpoints"""
        
        @self.app.get("/health")
        async def health_check():
            """System health and security status"""
            return {
                "status": "healthy",
                "security": self.ssl_config.get_security_parameters(),
                "timestamp": import_time.time()
            }
            
    def run(self):
        """Launch server with security configuration"""
        ssl_context = self.ssl_config.get_ssl_context()
        if not ssl_context:
            raise RuntimeError("Failed to initialize SSL context")
            
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=8000,
            ssl_keyfile=str(self.ssl_config.key_file),
            ssl_certfile=str(self.ssl_config.cert_file),
            ssl_version=2,  # Force TLS 1.2 minimum
            http="h2",     # Enable HTTP/2
            reload=False   # Disable in production
        )

# Server initialization
if __name__ == "__main__":
    server = SimulationServer()
    server.run()