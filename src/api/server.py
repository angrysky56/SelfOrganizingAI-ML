# Adaptive Server Implementation
# Implements environment-aware API endpoints

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import logging

from ..core.system_configuration import SystemConfiguration

class AdaptiveServer:
    """
    Environment-aware server implementation.
    
    Core Features:
    1. Environment-specific configuration
    2. Adaptive security protocols
    3. Dynamic resource management
    4. Intelligent error handling
    """
    
    def __init__(self, base_path: Path):
        self.logger = logging.getLogger(__name__)
        self.config = SystemConfiguration(base_path)
        self.app = self._initialize_app()
    
    def _initialize_app(self) -> FastAPI:
        """
        Initialize FastAPI application with environment configuration.
        
        Returns:
            FastAPI: Configured application instance
        """
        app = FastAPI(
            title="Nexus Prime Development Server",
            description="Adaptive AI System Development Interface",
            version="1.0.0"
        )
        
        # Configure middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if self.config._env_context.env_type == 'development' else ["https://localhost"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: FastAPI):
        """Register API endpoints with environment awareness."""
        
        @app.get("/health")
        async def health_check():
            """System health and configuration status."""
            return {
                "status": "healthy",
                "environment": self.config._env_context.env_type,
                "security_level": self.config._env_context.security_level
            }
        
        @app.get("/config")
        async def get_config():
            """Current system configuration state."""
            if self.config._env_context.env_type != 'development':
                return {"error": "Configuration endpoint disabled in production"}
            return json.loads(self.config.export_config())
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch server with environment-specific configuration.
        
        Args:
            host: Server host address
            port: Server port number
        """
        ssl_config = self.config.get_ssl_config()
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            ssl_keyfile=str(Path(ssl_config['cert_dir']) / 'server.key'),
            ssl_certfile=str(Path(ssl_config['cert_dir']) / 'server.crt'),
            ssl_version=2,
            reload=self.config._env_context.runtime_flags['allow_hot_reload']
        )