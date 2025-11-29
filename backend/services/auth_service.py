"""
Authentication Service
=====================
Handles user authentication, JWT tokens, and security
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from models.database import User, AuditLog
from models.schemas import UserRegister


class AuthService:
    """Authentication service"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    async def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    async def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token"""
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            token_type = payload.get("type", "access")
            
            if user_id is None:
                return None
            
            # Check if token has expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                return None
            
            # For this example, we'll return user data from token
            # In production, you might want to fetch fresh user data from DB
            return {
                "id": user_id,
                "token_type": token_type,
                "exp": exp
            }
            
        except jwt.PyJWTError:
            return None
    
    async def create_user(self, db: AsyncSession, user_data: UserRegister) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = await self.get_user_by_email(db, user_data.email)
        if existing_user:
            raise ValueError("Email already registered")
        
        # Hash password
        hashed_password = self.hash_password(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            password_hash=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            phone=user_data.phone,
            is_active=True,
            is_verified=False  # Email verification would be implemented separately
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Log registration
        await self.log_audit_action(db, user.id, "user_registered")
        
        return user
    
    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = await self.get_user_by_email(db, email)
        
        if not user:
            return None
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        if not self.verify_password(password, user.password_hash):
            # Log failed login attempt
            await self.log_audit_action(db, user.id, "login_failed")
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        
        # Log successful login
        await self.log_audit_action(db, user.id, "login_successful")
        
        return user
    
    async def get_user_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email"""
        result = await db.execute(select(User).filter(User.email == email))
        return result.scalar_one_or_none()
    
    async def get_user_by_id(self, db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(select(User).filter(User.id == user_id))
        return result.scalar_one_or_none()
    
    async def update_user_password(self, db: AsyncSession, user_id: str, new_password: str) -> bool:
        """Update user password"""
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return False
        
        user.password_hash = self.hash_password(new_password)
        user.updated_at = datetime.utcnow()
        await db.commit()
        
        # Log password change
        await self.log_audit_action(db, user_id, "password_changed")
        
        return True
    
    async def deactivate_user(self, db: AsyncSession, user_id: str) -> bool:
        """Deactivate user account"""
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return False
        
        user.is_active = False
        user.updated_at = datetime.utcnow()
        await db.commit()
        
        # Log account deactivation
        await self.log_audit_action(db, user_id, "account_deactivated")
        
        return True
    
    async def verify_user_email(self, db: AsyncSession, user_id: str) -> bool:
        """Mark user email as verified"""
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return False
        
        user.is_verified = True
        user.updated_at = datetime.utcnow()
        await db.commit()
        
        # Log email verification
        await self.log_audit_action(db, user_id, "email_verified")
        
        return True
    
    async def log_audit_action(self, db: AsyncSession, user_id: str, action: str, 
                              details: Optional[Dict] = None):
        """Log audit action"""
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            details=details,
            created_at=datetime.utcnow()
        )
        
        db.add(audit_log)
        await db.commit()
    
    async def get_user_sessions(self, db: AsyncSession, user_id: str, limit: int = 10):
        """Get user's recent login sessions"""
        result = await db.execute(
            select(AuditLog)
            .filter(AuditLog.user_id == user_id)
            .filter(AuditLog.action.in_(["login_successful", "login_failed"]))
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def is_password_compromised(self, password: str) -> bool:
        """Check if password is in common compromised password list"""
        # This would integrate with a service like HaveIBeenPwned
        # For now, just check against common passwords
        common_passwords = [
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "1234567890"
        ]
        return password.lower() in common_passwords
    
    async def generate_password_reset_token(self, user_id: str) -> str:
        """Generate password reset token"""
        expire = datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "password_reset"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    async def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Verify password reset token and return user ID"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            token_type = payload.get("type")
            
            if user_id is None or token_type != "password_reset":
                return None
            
            # Check if token has expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                return None
            
            return user_id
            
        except jwt.PyJWTError:
            return None
    
    async def check_rate_limit(self, db: AsyncSession, email: str, action: str = "login") -> bool:
        """Check if user has exceeded rate limit for specific action"""
        # Get failed attempts in last 15 minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=15)
        
        result = await db.execute(
            select(AuditLog)
            .filter(AuditLog.details.op("->>")('"email"') == email)
            .filter(AuditLog.action == f"{action}_failed")
            .filter(AuditLog.created_at > cutoff_time)
        )
        
        failed_attempts = len(result.scalars().all())
        
        # Allow max 5 failed attempts in 15 minutes
        return failed_attempts < 5