"""
Bank OCR System - Enhanced Backend API
=====================================
Production-ready FastAPI backend for Angular frontend integration
Author: Suresh Chaudhary
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from contextlib import asynccontextmanager

# Import our modules
from config import Settings
from models.database import init_db, get_db
from models.schemas import *
from services.ocr_service import OCRService
from services.auth_service import AuthService
from services.document_service import DocumentService
from middleware.logging_middleware import LoggingMiddleware
from middleware.rate_limiting import RateLimitMiddleware
from utils.exception_handlers import setup_exception_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global services
ocr_service = None
auth_service = None
document_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ocr_service, auth_service, document_service
    
    logger.info("🚀 Starting Bank OCR API...")
    
    # Initialize database
    await init_db()
    
    # Initialize services
    ocr_service = OCRService()
    auth_service = AuthService()
    document_service = DocumentService()
    
    # Load ML models
    await ocr_service.load_models()
    
    logger.info("✅ Bank OCR API started successfully!")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down Bank OCR API...")
    await ocr_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Bank OCR System API",
    description="""
    Production-ready Bank Document OCR API for Angular frontend integration.
    
    ## Features
    - Document upload and OCR processing
    - User authentication and authorization
    - Document history and management
    - Real-time processing status
    - Comprehensive error handling
    
    ## Supported Documents
    - Bank cheques
    - Deposit slips
    - Account statements
    - Mixed language documents (English/Nepali)
    """,
    version="2.0.0",
    contact={
        "name": "Suresh Chaudhary",
        "email": "suresh@example.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)

# Load settings
settings = Settings()

# Configure CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Setup exception handlers
setup_exception_handlers(app)

# Security
security = HTTPBearer()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        user = await auth_service.verify_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==================== AUTH ENDPOINTS ====================

@app.post("/api/auth/register", response_model=AuthResponse, tags=["Authentication"])
async def register(user_data: UserRegister, db=Depends(get_db)):
    """Register a new user"""
    try:
        user = await auth_service.create_user(db, user_data)
        token = await auth_service.create_access_token(user.id)
        
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse.from_orm(user),
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/auth/login", response_model=AuthResponse, tags=["Authentication"])
async def login(user_data: UserLogin, db=Depends(get_db)):
    """User login"""
    try:
        user = await auth_service.authenticate_user(db, user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        token = await auth_service.create_access_token(user.id)
        
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse.from_orm(user),
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/api/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """Refresh access token"""
    try:
        new_token = await auth_service.create_access_token(current_user["id"])
        return TokenResponse(
            access_token=new_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@app.get("/api/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(**current_user)


# ==================== OCR ENDPOINTS ====================

@app.post("/api/ocr/upload", response_model=OCRResponse, tags=["OCR"])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Upload and process bank document
    
    - **file**: Bank document image (PNG, JPG, JPEG)
    - **Returns**: Extracted fields with confidence scores
    """
    try:
        # Validate file
        validation_result = await ocr_service.validate_file(file)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {validation_result.error}"
            )
        
        # Create document record
        document = await document_service.create_document(
            db=db,
            user_id=current_user["id"],
            filename=file.filename,
            file_size=file.size
        )
        
        # Process document
        result = await ocr_service.process_document(file, document.id)
        
        # Update document with results
        await document_service.update_document_results(
            db=db,
            document_id=document.id,
            extracted_fields=result.extracted_fields,
            confidence_scores=result.confidence_scores,
            processing_time=result.processing_time
        )
        
        # Schedule background tasks
        background_tasks.add_task(
            document_service.save_processed_image,
            document.id,
            result.processed_image
        )
        
        return OCRResponse(
            document_id=document.id,
            extracted_fields=result.extracted_fields,
            confidence_scores=result.confidence_scores,
            processing_time=result.processing_time,
            status="completed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Document processing failed"
        )


@app.get("/api/ocr/status/{document_id}", response_model=ProcessingStatus, tags=["OCR"])
async def get_processing_status(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Get document processing status"""
    try:
        document = await document_service.get_document(db, document_id, current_user["id"])
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return ProcessingStatus(
            document_id=document_id,
            status=document.status,
            progress=document.progress,
            message=document.status_message,
            created_at=document.created_at,
            completed_at=document.completed_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


@app.post("/api/ocr/reprocess/{document_id}", response_model=OCRResponse, tags=["OCR"])
async def reprocess_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Reprocess an existing document"""
    try:
        document = await document_service.get_document(db, document_id, current_user["id"])
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reprocess document
        result = await ocr_service.reprocess_document(document_id)
        
        # Update document
        await document_service.update_document_results(
            db=db,
            document_id=document_id,
            extracted_fields=result.extracted_fields,
            confidence_scores=result.confidence_scores,
            processing_time=result.processing_time
        )
        
        return OCRResponse(
            document_id=document_id,
            extracted_fields=result.extracted_fields,
            confidence_scores=result.confidence_scores,
            processing_time=result.processing_time,
            status="completed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reprocessing error: {e}")
        raise HTTPException(status_code=500, detail="Reprocessing failed")


# ==================== DOCUMENT MANAGEMENT ====================

@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
async def get_user_documents(
    page: int = 1,
    limit: int = 20,
    status_filter: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Get user's documents with pagination"""
    try:
        documents = await document_service.get_user_documents(
            db=db,
            user_id=current_user["id"],
            page=page,
            limit=limit,
            status_filter=status_filter
        )
        
        return DocumentListResponse(
            documents=[DocumentResponse.from_orm(doc) for doc in documents.items],
            total=documents.total,
            page=page,
            limit=limit,
            pages=documents.pages
        )
    except Exception as e:
        logger.error(f"Document list error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")


@app.get("/api/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Get specific document details"""
    try:
        document = await document_service.get_document(db, document_id, current_user["id"])
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse.from_orm(document)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch document")


@app.delete("/api/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Delete a document"""
    try:
        success = await document_service.delete_document(db, document_id, current_user["id"])
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@app.get("/api/documents/{document_id}/download", tags=["Documents"])
async def download_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Download processed document image"""
    try:
        file_path = await document_service.get_document_file_path(
            db, document_id, current_user["id"]
        )
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=f"document_{document_id}.png"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document download error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download document")


# ==================== ANALYTICS ENDPOINTS ====================

@app.get("/api/analytics/summary", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics_summary(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Get user analytics summary"""
    try:
        analytics = await document_service.get_user_analytics(db, current_user["id"])
        return AnalyticsResponse(**analytics)
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")


# ==================== SYSTEM ENDPOINTS ====================

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        ocr_status = await ocr_service.health_check() if ocr_service else False
        
        return HealthResponse(
            status="healthy" if ocr_status else "degraded",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            services={
                "ocr": ocr_status,
                "database": True,  # Would check DB connection
                "storage": True    # Would check file storage
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version="2.0.0",
            timestamp=datetime.utcnow(),
            services={
                "ocr": False,
                "database": False,
                "storage": False
            }
        )


@app.get("/api/config", response_model=ConfigResponse, tags=["System"])
async def get_config():
    """Get public configuration"""
    return ConfigResponse(
        max_file_size=settings.MAX_FILE_SIZE,
        allowed_formats=settings.ALLOWED_FILE_FORMATS,
        supported_languages=["English", "Nepali"],
        rate_limit=settings.RATE_LIMIT_PER_MINUTE,
        features={
            "batch_processing": True,
            "real_time_processing": True,
            "document_history": True,
            "analytics": True
        }
    )


# Root endpoint redirect
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bank OCR System API v2.0",
        "description": "Production-ready OCR API for Angular frontend",
        "documentation": "/docs",
        "health": "/api/health",
        "version": "2.0.0"
    }


# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )