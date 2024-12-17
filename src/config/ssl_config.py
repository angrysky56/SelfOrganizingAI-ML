"""
SSL Configuration Manager
Implements robust SSL/TLS protocol handling with security refinement
"""

import ssl
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from dataclasses import dataclass

@dataclass
class TLSProfile:
    """TLS protocol configuration."""
    min_version: ssl.TLSVersion
    max_version: ssl.TLSVersion
    cipher_suite: str
    session_timeout: int
    ticket_lifetime: int

class SSLConfigManager:
    """Manages SSL configuration and security parameters."""
    
    def __init__(self, cert_dir: str = None):
        self.cert_dir = Path(cert_dir) if cert_dir else Path(__file__).parent.parent.parent / "config" / "ssl"
        self.logger = logging.getLogger(__name__)
        self.cert_file = self.cert_dir / "server.crt"
        self.key_file = self.cert_dir / "server.key"
        self._active_profile = self._create_default_profile()

    def _create_default_profile(self) -> TLSProfile:
        """Create secure default TLS configuration."""
        return TLSProfile(
            min_version=ssl.TLSVersion.TLSv1_2,
            max_version=ssl.TLSVersion.TLSv1_3,
            cipher_suite='ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256',
            session_timeout=7200,
            ticket_lifetime=300
        )

    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context with security optimizations."""
        try:
            if not self.cert_file.exists() or not self.key_file.exists():
                self.logger.error("SSL certificates not found")
                return None

            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(
                certfile=str(self.cert_file),
                keyfile=str(self.key_file)
            )

            # Set protocol versions
            context.minimum_version = self._active_profile.min_version
            context.maximum_version = self._active_profile.max_version
            context.set_ciphers(self._active_profile.cipher_suite)

            # Security optimizations
            context.options |= (
                ssl.OP_NO_SSLv2 | 
                ssl.OP_NO_SSLv3 |
                ssl.OP_NO_TLSv1 | 
                ssl.OP_NO_TLSv1_1 |
                ssl.OP_NO_COMPRESSION
            )

            # Session handling
            context.set_session_cache_mode(ssl.SESS_CACHE_SERVER)
            context.set_session_ticket_keys(self._generate_session_keys())

            return context

        except Exception as e:
            self.logger.error(f"Failed to create SSL context: {e}")
            return None

    def _generate_session_keys(self) -> bytes:
        """Generate secure session keys using HKDF."""
        try:
            import os
            salt = os.urandom(32)
            key_material = os.urandom(48)

            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=48,
                salt=salt,
                info=b'session_ticket_keys'
            )
            return hkdf.derive(key_material)

        except Exception as e:
            self.logger.error(f"Session key generation failed: {e}")
            return os.urandom(48)

    def validate_cert_files(self) -> bool:
        """Validate certificate files exist and are readable."""
        try:
            return (self.cert_file.exists() and 
                    self.key_file.exists() and
                    self.cert_file.stat().st_size > 0 and
                    self.key_file.stat().st_size > 0)
        except Exception as e:
            self.logger.error(f"Certificate validation failed: {e}")
            return False

    def get_security_parameters(self) -> Dict[str, str]:
        """Get current security configuration."""
        return {
            'min_protocol': self._active_profile.min_version.name,
            'max_protocol': self._active_profile.max_version.name,
            'cipher_suite': self._active_profile.cipher_suite,
            'session_timeout': str(self._active_profile.session_timeout),
            'ticket_lifetime': str(self._active_profile.ticket_lifetime)
        }

    def get_cert_paths(self) -> Tuple[Path, Path]:
        """Get certificate file paths."""
        return self.cert_file, self.key_file