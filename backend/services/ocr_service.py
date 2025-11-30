"""
OCR Service
===========
Enhanced OCR service with async support and comprehensive error handling
"""

import asyncio
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from fastapi import UploadFile, HTTPException
from PIL import Image
import io
import uuid
from datetime import datetime

# Import the original OCR system
import sys
sys.path.append(str(Path(__file__).parent.parent))
from bank_ocr_api import OCRSystem

from config import settings
from models.schemas import FileValidationResult, ExtractedFields, ExtractedField


class OCRProcessingResult:
    """OCR processing result container"""
    
    def __init__(self):
        self.extracted_fields: ExtractedFields = None
        self.confidence_scores: Dict[str, float] = {}
        self.processing_time: float = 0.0
        self.processed_image: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self.status: str = "completed"
        self.error: Optional[str] = None


class EnhancedOCRService:
    """Enhanced OCR service with async support"""
    
    def __init__(self):
        self.ocr_system = OCRSystem()
        self.processing_queue = asyncio.Queue()
        self.active_jobs = {}
        self.is_initialized = False
    
    async def load_models(self):
        """Load OCR models asynchronously"""
        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                self.ocr_system.load_models
            )
            
            if success:
                self.is_initialized = True
                print("✅ OCR models loaded successfully")
            else:
                print("⚠️ OCR models failed to load")
                
            return success
            
        except Exception as e:
            print(f"❌ Error loading OCR models: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if OCR service is healthy"""
        return (
            self.is_initialized and 
            self.ocr_system.character_model is not None and
            self.ocr_system.metadata is not None
        )
    
    async def validate_file(self, file: UploadFile) -> FileValidationResult:
        """Validate uploaded file"""
        try:
            # Check file size
            if file.size > settings.MAX_FILE_SIZE:
                return FileValidationResult(
                    is_valid=False,
                    error=f"File size ({file.size} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)"
                )
            
            # Check file extension
            if file.filename:
                file_ext = file.filename.split('.')[-1].lower()
                if file_ext not in settings.ALLOWED_FILE_FORMATS:
                    return FileValidationResult(
                        is_valid=False,
                        error=f"File format '{file_ext}' not allowed. Supported formats: {', '.join(settings.ALLOWED_FILE_FORMATS)}"
                    )
            else:
                return FileValidationResult(
                    is_valid=False,
                    error="No filename provided"
                )
            
            # Try to read and validate image
            file_content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            try:
                # Validate image using PIL
                img = Image.open(io.BytesIO(file_content))
                img.verify()
                
                # Additional checks
                if img.width < 100 or img.height < 100:
                    return FileValidationResult(
                        is_valid=False,
                        error="Image too small (minimum 100x100 pixels)"
                    )
                
                if img.width > 4000 or img.height > 4000:
                    return FileValidationResult(
                        is_valid=False,
                        error="Image too large (maximum 4000x4000 pixels)"
                    )
                
                return FileValidationResult(
                    is_valid=True,
                    file_type=file_ext,
                    file_size=file.size
                )
                
            except Exception as img_error:
                return FileValidationResult(
                    is_valid=False,
                    error=f"Invalid image file: {str(img_error)}"
                )
            
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error=f"File validation failed: {str(e)}"
            )
    
    async def process_document(self, file: UploadFile, document_id: str) -> OCRProcessingResult:
        """Process document with OCR"""
        result = OCRProcessingResult()
        start_time = time.time()
        
        try:
            # Read file content
            file_content = await file.read()
            
            # Convert to numpy array
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Store original image info
            result.metadata = {
                "original_filename": file.filename,
                "file_size": file.size,
                "image_shape": image.shape,
                "document_id": document_id,
                "processing_started": datetime.utcnow().isoformat()
            }
            
            # Run OCR processing in thread pool
            loop = asyncio.get_event_loop()
            ocr_result = await loop.run_in_executor(
                None,
                self.ocr_system.process_document,
                image
            )
            
            # Convert result to our format
            result.extracted_fields = self._convert_ocr_result(ocr_result)
            result.confidence_scores = self._extract_confidence_scores(ocr_result)
            result.processed_image = image  # Could be enhanced/annotated image
            result.processing_time = time.time() - start_time
            result.status = "completed"
            
            # Additional metadata
            result.metadata.update({
                "processing_completed": datetime.utcnow().isoformat(),
                "model_version": "2.0.0",
                "confidence_threshold": settings.OCR_CONFIDENCE_THRESHOLD
            })
            
            return result
            
        except Exception as e:
            result.error = str(e)
            result.status = "failed"
            result.processing_time = time.time() - start_time
            raise HTTPException(
                status_code=500,
                detail=f"OCR processing failed: {str(e)}"
            )
    
    async def reprocess_document(self, document_id: str) -> OCRProcessingResult:
        """Reprocess an existing document"""
        # This would load the original image from storage and reprocess
        # For now, return a placeholder
        result = OCRProcessingResult()
        result.status = "failed"
        result.error = "Reprocessing not implemented yet"
        return result
    
    def _convert_ocr_result(self, ocr_result: Dict) -> ExtractedFields:
        """Convert OCR result to ExtractedFields schema"""
        
        def create_field(field_data: Dict) -> ExtractedField:
            return ExtractedField(
                value=field_data.get("value", ""),
                confidence=field_data.get("confidence", 0.0),
                coordinates=None  # Could add bounding box coordinates
            )
        
        return ExtractedFields(
            bank_name=create_field(ocr_result.get("bankName", {})),
            account_holder_name=create_field(ocr_result.get("accountHolderName", {})),
            account_number=create_field(ocr_result.get("accountNumber", {})),
            routing_number=create_field(ocr_result.get("routingNumber", {})),
            amount=create_field(ocr_result.get("amount", {})),
            date=create_field(ocr_result.get("date", {})),
            additional_fields={}
        )
    
    def _extract_confidence_scores(self, ocr_result: Dict) -> Dict[str, float]:
        """Extract confidence scores from OCR result"""
        confidence_scores = {}
        
        for field_name, field_data in ocr_result.items():
            if isinstance(field_data, dict) and "confidence" in field_data:
                confidence_scores[field_name] = field_data["confidence"]
        
        return confidence_scores
    
    async def save_processed_image(self, document_id: str, image: np.ndarray, 
                                 processed_dir: Optional[str] = None):
        """Save processed image to storage"""
        try:
            if processed_dir is None:
                processed_dir = settings.PROCESSED_DIR
            
            Path(processed_dir).mkdir(exist_ok=True)
            
            filename = f"{document_id}_processed.png"
            filepath = Path(processed_dir) / filename
            
            # Save image
            cv2.imwrite(str(filepath), image)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving processed image: {e}")
            return None
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "is_initialized": self.is_initialized,
            "active_jobs": len(self.active_jobs),
            "queue_size": self.processing_queue.qsize(),
            "models_loaded": {
                "character_model": self.ocr_system.character_model is not None,
                "sequence_model": self.ocr_system.sequence_model is not None,
                "detection_model": self.ocr_system.detection_model is not None,
                "metadata": self.ocr_system.metadata is not None
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear processing queue
            while not self.processing_queue.empty():
                try:
                    self.processing_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Clear active jobs
            self.active_jobs.clear()
            
            # Could add model cleanup here if needed
            print("🧹 OCR service cleaned up")
            
        except Exception as e:
            print(f"Error during OCR service cleanup: {e}")


class BatchOCRService:
    """Service for batch OCR processing"""
    
    def __init__(self, ocr_service: EnhancedOCRService):
        self.ocr_service = ocr_service
        self.batch_queue = asyncio.Queue()
        self.batch_jobs = {}
    
    async def process_batch(self, document_ids: List[str], 
                          priority: int = 1) -> str:
        """Process multiple documents in batch"""
        batch_id = str(uuid.uuid4())
        
        batch_job = {
            "id": batch_id,
            "document_ids": document_ids,
            "priority": priority,
            "status": "pending",
            "progress": 0,
            "completed_documents": [],
            "failed_documents": [],
            "created_at": datetime.utcnow()
        }
        
        self.batch_jobs[batch_id] = batch_job
        
        # Start batch processing in background
        asyncio.create_task(self._process_batch_job(batch_id))
        
        return batch_id
    
    async def _process_batch_job(self, batch_id: str):
        """Process batch job in background"""
        job = self.batch_jobs.get(batch_id)
        if not job:
            return
        
        try:
            job["status"] = "processing"
            total_docs = len(job["document_ids"])
            
            for i, doc_id in enumerate(job["document_ids"]):
                try:
                    # This would load and process the document
                    # For now, we'll simulate processing
                    await asyncio.sleep(1)  # Simulate processing time
                    
                    job["completed_documents"].append(doc_id)
                    job["progress"] = int((i + 1) / total_docs * 100)
                    
                except Exception as e:
                    job["failed_documents"].append({"document_id": doc_id, "error": str(e)})
            
            job["status"] = "completed"
            job["completed_at"] = datetime.utcnow()
            
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Get batch processing status"""
        return self.batch_jobs.get(batch_id)


# Global OCR service instance
OCRService = EnhancedOCRService