"""
Database Models and Configuration
=================================
SQLAlchemy models for the Bank OCR system
"""

from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import asyncio
from typing import AsyncGenerator
from config import settings

# Create async engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Database URL conversion for async
if settings.DATABASE_URL.startswith("sqlite"):
    ASYNC_DATABASE_URL = settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
else:
    # For PostgreSQL
    ASYNC_DATABASE_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    future=True
)

# Create async session
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


def generate_uuid():
    """Generate UUID string"""
    return str(uuid.uuid4())


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    phone = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    def __repr__(self):
        return f"<User {self.email}>"


class Document(Base):
    """Document model"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String, nullable=True)
    processed_file_path = Column(String, nullable=True)
    document_type = Column(String, default="other")
    status = Column(String, default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, default=0)
    status_message = Column(Text, nullable=True)
    
    # OCR Results
    extracted_fields = Column(JSON, nullable=True)
    confidence_scores = Column(JSON, nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Foreign Keys
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    
    def __repr__(self):
        return f"<Document {self.filename} - {self.status}>"


class Notification(Base):
    """Notification model"""
    __tablename__ = "notifications"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String, default="info")  # info, success, warning, error
    is_read = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime, nullable=True)
    
    # Foreign Keys
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    
    def __repr__(self):
        return f"<Notification {self.title} for {self.user_id}>"


class ProcessingJob(Base):
    """Background processing job model"""
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    job_type = Column(String, nullable=False)  # ocr, batch_ocr, reprocess
    status = Column(String, default="pending")  # pending, running, completed, failed
    priority = Column(Integer, default=1)  # 1-5, higher is more priority
    progress = Column(Integer, default=0)
    
    # Job data
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_completion = Column(DateTime, nullable=True)
    
    # Foreign Keys
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    
    def __repr__(self):
        return f"<ProcessingJob {self.job_type} - {self.status}>"


class AuditLog(Base):
    """Audit log model for tracking user actions"""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    action = Column(String, nullable=False)  # login, upload, delete, etc.
    resource_type = Column(String, nullable=True)  # document, user, etc.
    resource_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    def __repr__(self):
        return f"<AuditLog {self.action} by {self.user_id}>"


class SystemMetrics(Base):
    """System metrics model"""
    __tablename__ = "system_metrics"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String, nullable=True)
    tags = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemMetrics {self.metric_name}: {self.metric_value}>"


class APIUsage(Base):
    """API usage tracking model"""
    __tablename__ = "api_usage"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    endpoint = Column(String, nullable=False)
    method = Column(String, nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time = Column(Float, nullable=False)
    request_size = Column(Integer, nullable=True)
    response_size = Column(Integer, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    def __repr__(self):
        return f"<APIUsage {self.method} {self.endpoint} - {self.status_code}>"


# Database initialization functions
async def init_db():
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Database utilities
async def create_user_with_defaults(db: AsyncSession, user_data: dict) -> User:
    """Create a new user with default settings"""
    user = User(**user_data)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    # Create welcome notification
    notification = Notification(
        user_id=user.id,
        title="Welcome to Bank OCR System!",
        message="Your account has been created successfully. You can now start uploading documents for OCR processing.",
        type="success"
    )
    db.add(notification)
    await db.commit()
    
    return user


async def get_user_by_email(db: AsyncSession, email: str) -> User:
    """Get user by email"""
    from sqlalchemy import select
    result = await db.execute(select(User).filter(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> User:
    """Get user by ID"""
    from sqlalchemy import select
    result = await db.execute(select(User).filter(User.id == user_id))
    return result.scalar_one_or_none()


async def create_document_record(db: AsyncSession, **kwargs) -> Document:
    """Create a new document record"""
    document = Document(**kwargs)
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


async def update_document_status(db: AsyncSession, document_id: str, status: str, **kwargs) -> Document:
    """Update document status and other fields"""
    from sqlalchemy import select
    result = await db.execute(select(Document).filter(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if document:
        document.status = status
        document.updated_at = datetime.utcnow()
        
        for key, value in kwargs.items():
            if hasattr(document, key):
                setattr(document, key, value)
        
        await db.commit()
        await db.refresh(document)
    
    return document


async def log_audit_action(db: AsyncSession, user_id: str, action: str, **kwargs):
    """Log an audit action"""
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        **kwargs
    )
    db.add(audit_log)
    await db.commit()


async def record_api_usage(db: AsyncSession, **kwargs):
    """Record API usage metrics"""
    api_usage = APIUsage(**kwargs)
    db.add(api_usage)
    await db.commit()


# Health check functions
async def check_database_health() -> bool:
    """Check if database is healthy"""
    try:
        async with AsyncSessionLocal() as session:
            # Simple query to check connection
            await session.execute("SELECT 1")
            return True
    except Exception:
        return False


# Migration utilities
async def run_migrations():
    """Run database migrations if needed"""
    # This is where you would implement migration logic
    # For now, we'll just ensure tables are created
    await init_db()


# Cleanup utilities
async def cleanup_old_records(days: int = 30):
    """Cleanup old records (logs, metrics, etc.)"""
    from sqlalchemy import delete
    from datetime import timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    async with AsyncSessionLocal() as session:
        # Clean old audit logs
        await session.execute(
            delete(AuditLog).where(AuditLog.created_at < cutoff_date)
        )
        
        # Clean old API usage records
        await session.execute(
            delete(APIUsage).where(APIUsage.timestamp < cutoff_date)
        )
        
        # Clean old system metrics
        await session.execute(
            delete(SystemMetrics).where(SystemMetrics.timestamp < cutoff_date)
        )
        
        await session.commit()


if __name__ == "__main__":
    # Run database initialization
    asyncio.run(init_db())