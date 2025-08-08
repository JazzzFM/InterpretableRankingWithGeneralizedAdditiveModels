import pytest
from unittest.mock import patch, MagicMock
from datetime import timedelta
import jwt

from src.auth import (
    verify_password, get_password_hash, authenticate_user,
    create_access_token, get_current_user, fake_users_db,
    SECRET_KEY, ALGORITHM
)


class TestAuthentication:
    """Test suite for authentication functionality."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        # Hash should be different from original password
        assert hashed != password
        
        # Should verify correctly
        assert verify_password(password, hashed) is True
        
        # Wrong password should not verify
        assert verify_password("wrong_password", hashed) is False

    def test_authenticate_user_success(self):
        """Test successful user authentication."""
        user = authenticate_user(fake_users_db, "admin", "admin123")
        
        assert user is not None
        assert user.username == "admin"
        assert user.disabled is False

    def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password."""
        user = authenticate_user(fake_users_db, "admin", "wrong_password")
        assert user is None

    def test_authenticate_user_nonexistent(self):
        """Test authentication with non-existent user."""
        user = authenticate_user(fake_users_db, "nonexistent", "password")
        assert user is None

    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "testuser"}
        token = create_access_token(data, expires_delta=timedelta(minutes=30))
        
        # Should be a valid JWT token
        assert isinstance(token, str)
        
        # Should be decodable
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert decoded["sub"] == "testuser"
        assert "exp" in decoded

    def test_create_access_token_with_expiry(self):
        """Test JWT token creation with custom expiry."""
        data = {"sub": "testuser"}
        expires_delta = timedelta(minutes=60)
        token = create_access_token(data, expires_delta=expires_delta)
        
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert decoded["sub"] == "testuser"

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        """Test getting current user with valid token."""
        # Create a valid token
        token = create_access_token({"sub": "admin"})
        
        # Mock credentials
        mock_credentials = MagicMock()
        mock_credentials.credentials = token
        
        user = await get_current_user(mock_credentials)
        
        assert user is not None
        assert user.username == "admin"

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test getting current user with invalid token."""
        mock_credentials = MagicMock()
        mock_credentials.credentials = "invalid_token"
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await get_current_user(mock_credentials)

    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(self):
        """Test getting current user with expired token."""
        # Create an expired token
        token = create_access_token({"sub": "admin"}, expires_delta=timedelta(seconds=-1))
        
        mock_credentials = MagicMock()
        mock_credentials.credentials = token
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await get_current_user(mock_credentials)