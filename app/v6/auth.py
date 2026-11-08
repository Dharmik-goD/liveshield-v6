
import os
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError


SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://kiiljbtaklbiwncbvmxl.supabase.co",
).rstrip("/")

SUPABASE_JWT_ISSUER = os.getenv(
    "SUPABASE_JWT_ISSUER",
    f"{SUPABASE_URL}/auth/v1",
)

SUPABASE_JWKS_URL = os.getenv(
    "SUPABASE_JWKS_URL",
    f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json",
)

bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_jwks_client():
    return PyJWKClient(
        SUPABASE_JWKS_URL,
        cache_keys=True,
        lifespan=300,
        timeout=10,
    )


def unauthorized():
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required or access token is invalid.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
):
    if credentials is None:
        raise unauthorized()

    if credentials.scheme.lower() != "bearer":
        raise unauthorized()

    token = credentials.credentials

    try:
        signing_key = get_jwks_client().get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            issuer=SUPABASE_JWT_ISSUER,
            audience="authenticated",
            options={
                "require": ["exp", "iat", "iss", "aud", "sub"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
            leeway=30,
        )

        if payload.get("role") != "authenticated":
            raise unauthorized()

        if not isinstance(payload.get("sub"), str) or not payload["sub"]:
            raise unauthorized()

        return {
            "user_id": payload["sub"],
            "role": payload["role"],
            "email": payload.get("email"),
        }

    except HTTPException:
        raise

    except (PyJWTError, Exception) as exc:
        # Do not expose token contents or internal verification errors.
        raise unauthorized() from exc
