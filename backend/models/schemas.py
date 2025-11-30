"""
Pydantic Schemas
===============
Data models for API requests and responses
"""

from pydantic import BaseModel, EmailStr, field_validator, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Document type"""
    CHEQUE = "cheque"
    DEPOSIT_SLIP = "deposit_slip"
    BANK_STATEMENT = "bank_statement"
    OTHER = "other"


# ==================== Authentication Schemas ====================

class UserRegister(BaseModel):
    """User registration schema"""
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    phone: Optional[str] = Field(None, pattern=r'^[\+]?[1-9][\d]{0,15}$')
    
    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """User login schema"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response schema"""
    id: str
    email: str
    first_name: str
    last_name: str
    phone: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthResponse(BaseModel):
    """Authentication response schema"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    expires_in: int


# ==================== OCR Schemas ====================

class ExtractedField(BaseModel):
    """Extracted field with confidence"""
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    coordinates: Optional[Dict[str, int]] = None


class ExtractedFields(BaseModel):
    """All extracted fields from document"""
    bank_name: ExtractedField
    account_holder_name: ExtractedField
    account_number: ExtractedField
    routing_number: ExtractedField
    amount: ExtractedField
    date: ExtractedField
    additional_fields: Optional[Dict[str, ExtractedField]] = {}


class OCRResponse(BaseModel):
    """OCR processing response"""
    document_id: str
    extracted_fields: ExtractedFields
    confidence_scores: Dict[str, float]
    processing_time: float
    status: ProcessingStatus
    message: Optional[str] = None


class ProcessingStatusResponse(BaseModel):
    """Processing status response"""
    document_id: str
    status: ProcessingStatus
    progress: int = Field(..., ge=0, le=100)
    message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# ==================== Document Schemas ====================

class DocumentCreate(BaseModel):
    """Document creation schema"""
    filename: str
    file_size: int
    document_type: Optional[DocumentType] = DocumentType.OTHER
    metadata: Optional[Dict[str, Any]] = {}


class DocumentResponse(BaseModel):
    """Document response schema"""
    id: str
    filename: str
    file_size: int
    document_type: DocumentType
    status: ProcessingStatus
    progress: int
    extracted_fields: Optional[ExtractedFields] = None
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    user_id: str
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Document list response with pagination"""
    documents: List[DocumentResponse]
    total: int
    page: int
    limit: int
    pages: int


class DocumentUpdate(BaseModel):
    """Document update schema"""
    extracted_fields: Optional[ExtractedFields] = None
    confidence_scores: Optional[Dict[str, float]] = None
    status: Optional[ProcessingStatus] = None
    progress: Optional[int] = Field(None, ge=0, le=100)


# ==================== Analytics Schemas ====================

class AnalyticsResponse(BaseModel):
    """Analytics response schema"""
    total_documents: int
    documents_today: int
    documents_this_week: int
    documents_this_month: int
    average_processing_time: float
    success_rate: float
    most_common_bank: Optional[str] = None
    processing_stats: Dict[str, Any]


class ProcessingMetrics(BaseModel):
    """Processing metrics"""
    avg_confidence: float
    field_accuracies: Dict[str, float]
    processing_times: List[float]
    error_rates: Dict[str, float]


# ==================== System Schemas ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str  # healthy, degraded, unhealthy
    version: str
    timestamp: datetime
    services: Dict[str, bool]


class ConfigResponse(BaseModel):
    """Configuration response"""
    max_file_size: int
    allowed_formats: List[str]
    supported_languages: List[str]
    rate_limit: int
    features: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime


class ValidationError(BaseModel):
    """Validation error schema"""
    field: str
    message: str
    value: Any


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "validation_error"
    message: str = "Validation failed"
    details: List[ValidationError]
    timestamp: datetime


# ==================== File Upload Schemas ====================

class FileValidationResult(BaseModel):
    """File validation result"""
    is_valid: bool
    error: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None


class UploadResponse(BaseModel):
    """File upload response"""
    filename: str
    file_size: int
    file_type: str
    document_id: str
    message: str


# ==================== Batch Processing Schemas ====================

class BatchProcessRequest(BaseModel):
    """Batch processing request"""
    document_ids: List[str]
    priority: Optional[int] = Field(1, ge=1, le=5)
    notification_email: Optional[EmailStr] = None


class BatchProcessResponse(BaseModel):
    """Batch processing response"""
    batch_id: str
    document_count: int
    estimated_completion: datetime
    status: str


class BatchStatus(BaseModel):
    """Batch processing status"""
    batch_id: str
    status: str
    progress: int
    completed_documents: int
    failed_documents: int
    total_documents: int
    created_at: datetime
    estimated_completion: Optional[datetime] = None


# ==================== Notification Schemas ====================

class NotificationCreate(BaseModel):
    """Notification creation schema"""
    user_id: str
    title: str
    message: str
    type: str = "info"  # info, success, warning, error
    metadata: Optional[Dict[str, Any]] = {}


class NotificationResponse(BaseModel):
    """Notification response schema"""
    id: str
    title: str
    message: str
    type: str
    is_read: bool
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = {}
    
    class Config:
        from_attributes = True


# ==================== Search Schemas ====================

class SearchRequest(BaseModel):
    """Document search request"""
    query: Optional[str] = None
    document_type: Optional[DocumentType] = None
    status: Optional[ProcessingStatus] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    bank_name: Optional[str] = None
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    page: int = Field(1, ge=1)
    limit: int = Field(20, ge=1, le=100)


class SearchResponse(BaseModel):
    """Document search response"""
    documents: List[DocumentResponse]
    total: int
    page: int
    limit: int
    query: str
    filters: Dict[str, Any]