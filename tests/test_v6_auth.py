
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.v6 import auth


@pytest.fixture
def signing_setup(monkeypatch):
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()

    class FakeJWKSClient:
        def get_signing_key_from_jwt(self, token):
            return SimpleNamespace(key=public_key)

    monkeypatch.setattr(
        auth,
        "get_jwks_client",
        lambda: FakeJWKSClient(),
    )

    def create_token(**overrides):
        now = datetime.now(timezone.utc)

        payload = {
            "sub": "test-user-123",
            "role": "authenticated",
            "aud": "authenticated",
            "iss": auth.SUPABASE_JWT_ISSUER,
            "iat": now,
            "exp": now + timedelta(minutes=10),
            "email": "researcher@example.test",
        }

        payload.update(overrides)

        return jwt.encode(
            payload,
            private_key,
            algorithm="ES256",
            headers={"kid": "test-key"},
        )

    return create_token


@pytest.fixture
def client():
    app = FastAPI()

    @app.get("/protected")
    async def protected(user=Depends(auth.require_authenticated_user)):
        return {"status": "ok", "user": user}

    return TestClient(app)


def test_missing_token_returns_401(client):
    response = client.get("/protected")

    assert response.status_code == 401


def test_invalid_token_returns_401(client):
    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer invalid.token.value"},
    )

    assert response.status_code == 401


def test_expired_token_returns_401(client, signing_setup):
    token = signing_setup(
        exp=datetime.now(timezone.utc) - timedelta(minutes=2)
    )

    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_valid_token_returns_authenticated_user(client, signing_setup):
    token = signing_setup()

    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["user"]["user_id"] == "test-user-123"
    assert response.json()["user"]["role"] == "authenticated"


def test_wrong_role_returns_401(client, signing_setup):
    token = signing_setup(role="service_role")

    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_wrong_issuer_returns_401(client, signing_setup):
    token = signing_setup(iss="https://wrong.example.com")

    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_wrong_audience_returns_401(client, signing_setup):
    token = signing_setup(aud="wrong-audience")

    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
