"""
Configuration Settings
=====================
Application configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Bank OCR System"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:4200",  # Angular dev server
        "http://localhost:3000",  # Alternative frontend
        "https://your-frontend-domain.com"
    ]
    
    # Trusted hosts
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "your-domain.com"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./bank_ocr.db"
    DATABASE_ECHO: bool = False
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_FORMATS: List[str] = ["jpg", "jpeg", "png"]
    UPLOAD_DIR: str = "uploads"
    PROCESSED_DIR: str = "processed"
    
    # OCR Settings
    OCR_CONFIDENCE_THRESHOLD: float = 0.5
    OCR_TIMEOUT: int = 30  # seconds
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Redis (for caching and rate limiting)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Email (for notifications)
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "noreply@bankocr.com"
    
    # Model paths
    MODEL_BASE_PATH: str = "models"
    CHARACTER_MODEL_PATH: str = "models/complete_ocr_character_model.h5"
    SEQUENCE_MODEL_PATH: str = "models/complete_ocr_sequence_model.h5"
    DETECTION_MODEL_PATH: str = "models/complete_ocr_detection_model.h5"
    METADATA_PATH: str = "models/complete_ocr_system_metadata.json"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        "logs",
        "static",
        settings.MODEL_BASE_PATH
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories()