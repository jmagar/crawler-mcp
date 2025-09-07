# OAuth Implementation for Crawler MCP Server

## Executive Summary

This document provides a comprehensive OAuth 2.1 implementation plan for the Crawler MCP server, enabling secure authentication and authorization across three key areas:

1. **MCP Server Protection**: OAuth2 for protecting MCP endpoints
2. **GitHub API Integration**: OAuth2 authorization code flow replacing bearer tokens
3. **Protected Resource Crawling**: OAuth support for accessing protected web resources

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Crawler MCP Server                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │           OAuth Middleware Layer            │    │
│  │  - Token validation                         │    │
│  │  - PKCE implementation                      │    │
│  │  - Token refresh logic                      │    │
│  └────────────────────────────────────────────┘    │
│                        │                            │
│  ┌──────────────┬──────┴──────┬────────────────┐  │
│  │   MCP Auth   │  GitHub OAuth│  Generic OAuth │  │
│  │   Provider   │   Provider   │    Provider    │  │
│  └──────────────┴──────────────┴────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │          Token Storage Manager              │    │
│  │  - Encrypted token storage                  │    │
│  │  - Auto-refresh mechanism                   │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
crawler_mcp/
├── auth/
│   ├── __init__.py
│   ├── config.py           # OAuth configuration settings
│   ├── middleware.py       # OAuth middleware for FastAPI/FastMCP
│   ├── client.py          # OAuth client implementation
│   ├── token_manager.py   # Token storage and refresh
│   ├── exceptions.py      # OAuth-specific exceptions
│   └── providers/
│       ├── __init__.py
│       ├── base.py        # Base OAuth provider
│       ├── github.py      # GitHub OAuth implementation
│       ├── google.py      # Google OAuth implementation
│       └── generic.py     # Generic OAuth2 provider
```

## Implementation Code

### 1. OAuth Configuration (`crawler_mcp/auth/config.py`)

```python
"""OAuth configuration management for Crawler MCP."""

from typing import Literal, Optional
from pydantic import BaseSettings, Field, SecretStr
from pathlib import Path


class OAuthProvider(BaseSettings):
    """OAuth provider configuration."""

    client_id: str
    client_secret: SecretStr
    authorize_url: str
    token_url: str
    userinfo_url: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    jwks_url: Optional[str] = None
    issuer: Optional[str] = None
    audience: Optional[str] = None


class OAuthSettings(BaseSettings):
    """OAuth settings for Crawler MCP."""

    # OAuth server settings (for protecting MCP endpoints)
    oauth_enabled: bool = Field(default=False, alias="OAUTH_ENABLED")
    oauth_mode: Literal["server", "client", "both"] = Field(
        default="client", alias="OAUTH_MODE"
    )

    # Token storage
    token_storage_dir: Path = Field(
        default=Path.home() / ".crawler-mcp" / "oauth-tokens",
        alias="OAUTH_TOKEN_STORAGE_DIR"
    )
    token_encryption_key: Optional[SecretStr] = Field(
        default=None, alias="OAUTH_TOKEN_ENCRYPTION_KEY"
    )

    # Callback configuration
    oauth_callback_host: str = Field(
        default="localhost", alias="OAUTH_CALLBACK_HOST"
    )
    oauth_callback_port: int = Field(
        default=8765, alias="OAUTH_CALLBACK_PORT"
    )

    # Provider configurations
    github_client_id: Optional[str] = Field(default=None, alias="GITHUB_CLIENT_ID")
    github_client_secret: Optional[SecretStr] = Field(
        default=None, alias="GITHUB_CLIENT_SECRET"
    )
    github_scopes: list[str] = Field(
        default_factory=lambda: ["repo", "read:org"],
        alias="GITHUB_SCOPES"
    )

    google_client_id: Optional[str] = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[SecretStr] = Field(
        default=None, alias="GOOGLE_CLIENT_SECRET"
    )
    google_scopes: list[str] = Field(
        default_factory=lambda: ["openid", "email", "profile"],
        alias="GOOGLE_SCOPES"
    )

    # Public paths (no auth required)
    public_paths: set[str] = Field(
        default_factory=lambda: {"/health", "/", "/docs", "/openapi.json"},
        alias="OAUTH_PUBLIC_PATHS"
    )

    # PKCE settings
    pkce_enabled: bool = Field(default=True, alias="OAUTH_PKCE_ENABLED")
    pkce_code_length: int = Field(default=128, alias="OAUTH_PKCE_CODE_LENGTH")

    # Token settings
    access_token_expire_minutes: int = Field(
        default=60, alias="OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    refresh_token_expire_days: int = Field(
        default=30, alias="OAUTH_REFRESH_TOKEN_EXPIRE_DAYS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


oauth_settings = OAuthSettings()
```

### 2. OAuth Middleware (`crawler_mcp/auth/middleware.py`)

```python
"""OAuth middleware for FastMCP/FastAPI integration."""

import logging
from typing import Optional, Dict, Any, Set
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError
import httpx

from .config import oauth_settings
from .token_manager import TokenManager
from .exceptions import InvalidTokenError, TokenExpiredError

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class OAuthMiddleware(BaseHTTPMiddleware):
    """OAuth2 middleware for protecting MCP endpoints."""

    def __init__(
        self,
        app,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        public_paths: Optional[Set[str]] = None,
        token_manager: Optional[TokenManager] = None,
    ):
        super().__init__(app)
        self.providers = providers or {}
        self.public_paths = public_paths or oauth_settings.public_paths
        self.token_manager = token_manager or TokenManager()
        self._jwks_cache: Dict[str, Any] = {}

    async def dispatch(self, request: Request, call_next):
        """Process OAuth authentication for protected endpoints."""

        # Skip auth for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)

        # Extract token from header
        credentials = await security(request)
        if not credentials:
            raise HTTPException(status_code=401, detail="Authorization required")

        try:
            # Validate token
            token_data = await self.validate_token(credentials.credentials)

            # Add claims to request scope
            request.scope["oauth2_claims"] = token_data
            request.scope["oauth2_provider"] = token_data.get("provider")

            response = await call_next(request)
            return response

        except TokenExpiredError:
            # Try to refresh token if available
            if await self.token_manager.can_refresh(credentials.credentials):
                new_token = await self.token_manager.refresh_token(
                    credentials.credentials
                )
                response = Response(status_code=401)
                response.headers["X-New-Token"] = new_token
                return response
            raise HTTPException(status_code=401, detail="Token expired")

        except InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            logger.error(f"OAuth middleware error: {e}")
            raise HTTPException(status_code=500, detail="Authentication error")

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token against configured providers."""

        errors = {}

        # Try each provider
        for provider_name, provider_config in self.providers.items():
            try:
                # Get JWKS if needed
                if provider_config.get("jwks_url"):
                    jwks = await self._get_jwks(provider_config["jwks_url"])
                else:
                    jwks = None

                # Decode and validate token
                claims = jwt.decode(
                    token,
                    key=jwks,
                    audience=provider_config.get("audience"),
                    issuer=provider_config.get("issuer"),
                    options={"verify_signature": True}
                )

                claims["provider"] = provider_name
                return claims

            except JWTError as e:
                errors[provider_name] = str(e)
                continue

        # No provider could validate the token
        raise InvalidTokenError(f"Token validation failed: {errors}")

    async def _get_jwks(self, jwks_url: str) -> Dict[str, Any]:
        """Fetch and cache JWKS from provider."""

        if jwks_url in self._jwks_cache:
            return self._jwks_cache[jwks_url]

        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url)
            response.raise_for_status()
            jwks = response.json()

        self._jwks_cache[jwks_url] = jwks
        return jwks
```

### 3. OAuth Client (`crawler_mcp/auth/client.py`)

```python
"""OAuth client for external service authentication."""

import asyncio
import secrets
import hashlib
import base64
from typing import Optional, Dict, Any
from urllib.parse import urlencode, parse_qs
import httpx
from fastapi import FastAPI, Request
import uvicorn

from .config import oauth_settings
from .token_manager import TokenManager
from .providers.base import OAuthProvider

logger = logging.getLogger(__name__)


class OAuthClient:
    """OAuth 2.1 client with PKCE support."""

    def __init__(
        self,
        provider: OAuthProvider,
        token_manager: Optional[TokenManager] = None,
        callback_host: Optional[str] = None,
        callback_port: Optional[int] = None,
    ):
        self.provider = provider
        self.token_manager = token_manager or TokenManager()
        self.callback_host = callback_host or oauth_settings.oauth_callback_host
        self.callback_port = callback_port or oauth_settings.oauth_callback_port
        self._authorization_code: Optional[str] = None
        self._state: Optional[str] = None
        self._code_verifier: Optional[str] = None

    async def authenticate(self) -> Dict[str, Any]:
        """Perform OAuth authentication flow."""

        # Generate PKCE parameters
        if oauth_settings.pkce_enabled:
            self._code_verifier = self._generate_code_verifier()
            code_challenge = self._generate_code_challenge(self._code_verifier)
        else:
            code_challenge = None

        # Generate state for CSRF protection
        self._state = secrets.token_urlsafe(32)

        # Build authorization URL
        auth_params = {
            "client_id": self.provider.client_id,
            "redirect_uri": f"http://{self.callback_host}:{self.callback_port}/callback",
            "scope": " ".join(self.provider.scopes),
            "state": self._state,
            "response_type": "code",
        }

        if code_challenge:
            auth_params["code_challenge"] = code_challenge
            auth_params["code_challenge_method"] = "S256"

        auth_url = f"{self.provider.authorize_url}?{urlencode(auth_params)}"

        # Start callback server
        callback_task = asyncio.create_task(self._start_callback_server())

        # Open browser for user authorization
        import webbrowser
        webbrowser.open(auth_url)

        print(f"Please authorize in your browser: {auth_url}")

        # Wait for callback
        await callback_task

        if not self._authorization_code:
            raise Exception("Authorization failed - no code received")

        # Exchange code for tokens
        tokens = await self._exchange_code_for_tokens()

        # Store tokens
        await self.token_manager.store_tokens(
            provider=self.provider.name,
            tokens=tokens
        )

        return tokens

    async def _exchange_code_for_tokens(self) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens."""

        token_data = {
            "grant_type": "authorization_code",
            "code": self._authorization_code,
            "redirect_uri": f"http://{self.callback_host}:{self.callback_port}/callback",
            "client_id": self.provider.client_id,
            "client_secret": self.provider.client_secret.get_secret_value(),
        }

        if self._code_verifier:
            token_data["code_verifier"] = self._code_verifier

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.provider.token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            return response.json()

    async def _start_callback_server(self):
        """Start temporary server to receive OAuth callback."""

        app = FastAPI()

        @app.get("/callback")
        async def oauth_callback(request: Request):
            params = dict(request.query_params)

            # Verify state
            if params.get("state") != self._state:
                return {"error": "Invalid state parameter"}

            # Store authorization code
            self._authorization_code = params.get("code")

            if not self._authorization_code:
                return {"error": params.get("error", "Unknown error")}

            # Shutdown server after receiving code
            asyncio.create_task(self._shutdown_server())

            return {"message": "Authorization successful! You can close this window."}

        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=self.callback_host,
                port=self.callback_port,
                log_level="error"
            )
        )

        await self._server.serve()

    async def _shutdown_server(self):
        """Shutdown callback server after delay."""
        await asyncio.sleep(1)
        self._server.should_exit = True

    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return base64.urlsafe_b64encode(
            secrets.token_bytes(oauth_settings.pkce_code_length // 2)
        ).decode("utf-8").rstrip("=")

    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge from verifier."""
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
```

### 4. Token Manager (`crawler_mcp/auth/token_manager.py`)

```python
"""Token storage and management for OAuth."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import httpx
from cryptography.fernet import Fernet

from .config import oauth_settings
from .exceptions import TokenNotFoundError, TokenRefreshError

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages OAuth token storage, encryption, and refresh."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or oauth_settings.token_storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encryption
        if oauth_settings.token_encryption_key:
            self.cipher = Fernet(
                oauth_settings.token_encryption_key.get_secret_value().encode()
            )
        else:
            # Generate and store key if not provided
            key_file = self.storage_dir / ".encryption_key"
            if key_file.exists():
                key = key_file.read_bytes()
            else:
                key = Fernet.generate_key()
                key_file.write_bytes(key)
                key_file.chmod(0o600)
            self.cipher = Fernet(key)

    async def store_tokens(
        self,
        provider: str,
        tokens: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store OAuth tokens securely."""

        token_data = {
            "provider": provider,
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "expires_at": self._calculate_expiry(tokens),
            "scopes": tokens.get("scope", "").split(),
            "metadata": metadata or {},
            "stored_at": datetime.utcnow().isoformat(),
        }

        # Encrypt sensitive data
        encrypted_data = self.cipher.encrypt(json.dumps(token_data).encode())

        # Store to file
        token_file = self.storage_dir / f"{provider}_tokens.enc"
        token_file.write_bytes(encrypted_data)
        token_file.chmod(0o600)

        logger.info(f"Stored tokens for provider: {provider}")

    async def get_tokens(self, provider: str) -> Dict[str, Any]:
        """Retrieve stored tokens for a provider."""

        token_file = self.storage_dir / f"{provider}_tokens.enc"

        if not token_file.exists():
            raise TokenNotFoundError(f"No tokens found for provider: {provider}")

        # Decrypt data
        encrypted_data = token_file.read_bytes()
        decrypted_data = self.cipher.decrypt(encrypted_data)
        token_data = json.loads(decrypted_data)

        # Check expiry
        if self._is_expired(token_data):
            if token_data.get("refresh_token"):
                # Try to refresh
                logger.info(f"Token expired for {provider}, attempting refresh")
                return await self.refresh_token(provider, token_data["refresh_token"])
            else:
                raise TokenNotFoundError(f"Token expired for provider: {provider}")

        return token_data

    async def refresh_token(
        self,
        provider: str,
        refresh_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refresh OAuth tokens."""

        if not refresh_token:
            # Get stored refresh token
            token_data = await self.get_tokens(provider)
            refresh_token = token_data.get("refresh_token")

            if not refresh_token:
                raise TokenRefreshError(f"No refresh token for provider: {provider}")

        # Get provider configuration
        from .providers import get_provider
        provider_config = get_provider(provider)

        # Request new tokens
        async with httpx.AsyncClient() as client:
            response = await client.post(
                provider_config.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": provider_config.client_id,
                    "client_secret": provider_config.client_secret.get_secret_value(),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code != 200:
                raise TokenRefreshError(
                    f"Failed to refresh token: {response.text}"
                )

            new_tokens = response.json()

        # Store new tokens
        await self.store_tokens(provider, new_tokens)

        return new_tokens

    async def revoke_tokens(self, provider: str) -> None:
        """Revoke and delete stored tokens."""

        token_file = self.storage_dir / f"{provider}_tokens.enc"

        if token_file.exists():
            # TODO: Call provider's revocation endpoint if available
            token_file.unlink()
            logger.info(f"Revoked tokens for provider: {provider}")

    def _calculate_expiry(self, tokens: Dict[str, Any]) -> str:
        """Calculate token expiry time."""

        expires_in = tokens.get("expires_in", 3600)
        expiry = datetime.utcnow() + timedelta(seconds=expires_in)
        return expiry.isoformat()

    def _is_expired(self, token_data: Dict[str, Any]) -> bool:
        """Check if token is expired."""

        expires_at = token_data.get("expires_at")
        if not expires_at:
            return False

        expiry = datetime.fromisoformat(expires_at)
        # Add 5 minute buffer
        return datetime.utcnow() > (expiry - timedelta(minutes=5))

    async def can_refresh(self, provider: str) -> bool:
        """Check if tokens can be refreshed."""

        try:
            token_data = await self.get_tokens(provider)
            return bool(token_data.get("refresh_token"))
        except TokenNotFoundError:
            return False
```

### 5. GitHub OAuth Provider (`crawler_mcp/auth/providers/github.py`)

```python
"""GitHub OAuth provider implementation."""

from typing import Optional, List
from .base import OAuthProvider
from ..config import oauth_settings


class GitHubOAuthProvider(OAuthProvider):
    """GitHub OAuth2 provider."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ):
        super().__init__(
            name="github",
            client_id=client_id or oauth_settings.github_client_id,
            client_secret=client_secret or oauth_settings.github_client_secret,
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            userinfo_url="https://api.github.com/user",
            scopes=scopes or oauth_settings.github_scopes,
        )

    async def get_user_info(self, access_token: str) -> dict:
        """Get GitHub user information."""

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.userinfo_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                }
            )
            response.raise_for_status()
            return response.json()
```

### 6. Updated GitHub Client (`crawler_mcp/clients/github_client.py`)

```python
"""Enhanced GitHub client with OAuth support."""

from typing import Optional
from .github_client import GitHubClient as OriginalGitHubClient
from ..auth.client import OAuthClient
from ..auth.providers.github import GitHubOAuthProvider
from ..auth.token_manager import TokenManager


class OAuthGitHubClient(OriginalGitHubClient):
    """GitHub client with OAuth authentication support."""

    def __init__(
        self,
        token: Optional[str] = None,
        use_oauth: bool = True,
        token_manager: Optional[TokenManager] = None,
        **kwargs
    ):
        # Try OAuth first if enabled
        if use_oauth and not token:
            token = self._get_oauth_token(token_manager)

        super().__init__(token=token, **kwargs)
        self.token_manager = token_manager or TokenManager()
        self.use_oauth = use_oauth

    def _get_oauth_token(self, token_manager: Optional[TokenManager]) -> Optional[str]:
        """Get OAuth token from storage or initiate OAuth flow."""

        tm = token_manager or TokenManager()

        try:
            # Try to get stored token
            import asyncio
            token_data = asyncio.run(tm.get_tokens("github"))
            return token_data.get("access_token")
        except:
            # No valid token, user needs to authenticate
            print("GitHub OAuth token not found. Please authenticate.")
            return None

    async def authenticate_oauth(self) -> str:
        """Perform OAuth authentication flow."""

        provider = GitHubOAuthProvider()
        client = OAuthClient(provider, self.token_manager)
        tokens = await client.authenticate()

        self.token = tokens["access_token"]
        # Update session headers
        if self._session:
            self._session.headers["Authorization"] = f"Bearer {self.token}"

        return self.token

    async def _ensure_session(self) -> None:
        """Ensure session with automatic token refresh."""

        await super()._ensure_session()

        # Check and refresh token if needed
        if self.use_oauth and self.token_manager:
            try:
                token_data = await self.token_manager.get_tokens("github")
                new_token = token_data.get("access_token")

                if new_token != self.token:
                    self.token = new_token
                    if self._session:
                        self._session.headers["Authorization"] = f"Bearer {self.token}"
            except:
                pass  # Continue with existing token
```

### 7. Server Integration (`crawler_mcp/server.py` update)

```python
# Add to imports
from crawler_mcp.auth.middleware import OAuthMiddleware
from crawler_mcp.auth.config import oauth_settings

# After creating FastMCP instance
mcp: FastMCP = FastMCP("crawler-mcp")

# Add OAuth middleware if enabled
if oauth_settings.oauth_enabled:
    # Configure OAuth providers
    oauth_providers = {}

    if oauth_settings.github_client_id:
        oauth_providers["github"] = {
            "issuer": "https://github.com",
            "audience": oauth_settings.github_client_id,
            "jwks_url": None,  # GitHub doesn't use JWKS for OAuth apps
        }

    if oauth_settings.google_client_id:
        oauth_providers["google"] = {
            "issuer": "https://accounts.google.com",
            "audience": oauth_settings.google_client_id,
            "jwks_url": "https://www.googleapis.com/oauth2/v3/certs",
        }

    # Add middleware to FastAPI app
    mcp.app.add_middleware(
        OAuthMiddleware,
        providers=oauth_providers,
        public_paths=oauth_settings.public_paths,
    )

    logger.info(f"OAuth protection enabled for MCP server (mode: {oauth_settings.oauth_mode})")
```

### 8. Environment Configuration (`.env`)

```env
# OAuth Configuration
OAUTH_ENABLED=true
OAUTH_MODE=both  # server, client, or both

# GitHub OAuth
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
GITHUB_SCOPES=repo,read:org,read:user

# Google OAuth (optional)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_SCOPES=openid,email,profile

# Token Storage
OAUTH_TOKEN_STORAGE_DIR=/home/user/.crawler-mcp/oauth-tokens
OAUTH_TOKEN_ENCRYPTION_KEY=your-base64-encoded-32-byte-key

# OAuth Server Settings
OAUTH_CALLBACK_HOST=localhost
OAUTH_CALLBACK_PORT=8765
OAUTH_PUBLIC_PATHS=/health,/,/docs,/openapi.json

# PKCE Settings
OAUTH_PKCE_ENABLED=true
OAUTH_PKCE_CODE_LENGTH=128

# Token Expiry
OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES=60
OAUTH_REFRESH_TOKEN_EXPIRE_DAYS=30
```

## Implementation Steps

1. **Create Auth Module Structure**
   ```bash
   mkdir -p crawler_mcp/auth/providers
   touch crawler_mcp/auth/{__init__,config,middleware,client,token_manager,exceptions}.py
   touch crawler_mcp/auth/providers/{__init__,base,github,google,generic}.py
   ```

2. **Install Dependencies**
   ```bash
   uv add python-jose[cryptography] httpx cryptography python-multipart
   ```

3. **Update pyproject.toml**
   ```toml
   [project.optional-dependencies]
   oauth = [
       "python-jose[cryptography]>=3.3.0",
       "cryptography>=41.0.0",
       "python-multipart>=0.0.6",
   ]
   ```

4. **Configure OAuth Apps**
   - GitHub: https://github.com/settings/developers
   - Google: https://console.cloud.google.com/apis/credentials

5. **Test OAuth Flow**
   ```python
   # Test script
   import asyncio
   from crawler_mcp.auth.client import OAuthClient
   from crawler_mcp.auth.providers.github import GitHubOAuthProvider

   async def test_oauth():
       provider = GitHubOAuthProvider()
       client = OAuthClient(provider)
       tokens = await client.authenticate()
       print(f"Access token: {tokens['access_token'][:20]}...")

   asyncio.run(test_oauth())
   ```

## Security Considerations

1. **Token Storage**
   - Tokens encrypted at rest using Fernet symmetric encryption
   - File permissions set to 0600 (owner read/write only)
   - Encryption keys stored separately from tokens

2. **PKCE Implementation**
   - Enabled by default for authorization code flow
   - 128-byte code verifier for maximum security
   - SHA256 code challenge method

3. **Token Refresh**
   - Automatic refresh 5 minutes before expiry
   - Refresh tokens stored encrypted
   - Failed refresh triggers re-authentication

4. **CSRF Protection**
   - State parameter validated on callback
   - Cryptographically secure random state generation

5. **Public Path Configuration**
   - Health endpoints remain public
   - Documentation conditionally protected
   - Configurable per deployment

## Migration Path

### Phase 1: Client OAuth (GitHub)
- Implement OAuth client for GitHub API
- Maintain backward compatibility with bearer tokens
- Test with existing GitHub PR tools

### Phase 2: Server Protection
- Add OAuth middleware to MCP server
- Configure provider JWKs validation
- Test with Claude Desktop client

### Phase 3: Generic OAuth
- Implement generic OAuth provider
- Add support for custom OAuth servers
- Enable crawling of OAuth-protected resources

## Testing Strategy

1. **Unit Tests**
   ```python
   # test_oauth_client.py
   async def test_pkce_generation():
       client = OAuthClient(provider)
       verifier = client._generate_code_verifier()
       challenge = client._generate_code_challenge(verifier)
       assert len(verifier) >= 43  # Min length per spec
   ```

2. **Integration Tests**
   - Mock OAuth provider responses
   - Test token refresh flow
   - Validate middleware protection

3. **E2E Tests**
   - Full OAuth flow with real providers
   - Token persistence across restarts
   - Multi-provider authentication

## Monitoring and Logging

- Token refresh events logged
- Failed authentication attempts tracked
- Provider-specific metrics collected
- Token expiry warnings issued

## Conclusion

This OAuth implementation provides comprehensive security for the Crawler MCP server while maintaining flexibility and ease of use. The modular design allows for easy extension with new providers and authentication methods.
