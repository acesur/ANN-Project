"""
Document Service
===============
Handles document management, storage, and analytics
"""

import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from fastapi import HTTPException

from models.database import Document, User, Notification
from models.schemas import ProcessingStatus, DocumentType
from config import settings


class DocumentPagination:
    """Document pagination result"""
    
    def __init__(self, items: List[Document], total: int, page: int, limit: int):
        self.items = items
        self.total = total
        self.page = page
        self.limit = limit
        self.pages = (total + limit - 1) // limit


class DocumentService:
    """Document management service"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.processed_dir = Path(settings.PROCESSED_DIR)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    async def create_document(self, db: AsyncSession, user_id: str, filename: str, 
                            file_size: int, document_type: str = "other") -> Document:
        """Create a new document record"""
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create file path
        file_extension = filename.split('.')[-1] if '.' in filename else 'bin'
        stored_filename = f"{document_id}.{file_extension}"
        file_path = self.upload_dir / stored_filename
        
        # Create document record
        document = Document(
            id=document_id,
            filename=stored_filename,
            original_filename=filename,
            file_size=file_size,
            file_path=str(file_path),
            document_type=document_type,
            status="pending",
            progress=0,
            user_id=user_id,
            created_at=datetime.utcnow()
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        return document
    
    async def get_document(self, db: AsyncSession, document_id: str, 
                          user_id: str) -> Optional[Document]:
        """Get document by ID and user ID"""
        result = await db.execute(
            select(Document).filter(
                and_(
                    Document.id == document_id,
                    Document.user_id == user_id
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_user_documents(self, db: AsyncSession, user_id: str, 
                               page: int = 1, limit: int = 20, 
                               status_filter: Optional[str] = None) -> DocumentPagination:
        """Get user's documents with pagination"""
        
        # Base query
        query = select(Document).filter(Document.user_id == user_id)
        
        # Apply status filter
        if status_filter:
            query = query.filter(Document.status == status_filter)
        
        # Order by creation time (newest first)
        query = query.order_by(Document.created_at.desc())
        
        # Get total count
        count_query = select(func.count(Document.id)).filter(Document.user_id == user_id)
        if status_filter:
            count_query = count_query.filter(Document.status == status_filter)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        documents = result.scalars().all()
        
        return DocumentPagination(documents, total, page, limit)
    
    async def update_document_results(self, db: AsyncSession, document_id: str,
                                    extracted_fields: Dict, confidence_scores: Dict,
                                    processing_time: float):
        """Update document with OCR results"""
        
        result = await db.execute(
            select(Document).filter(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if document:
            document.extracted_fields = extracted_fields.dict() if hasattr(extracted_fields, 'dict') else extracted_fields
            document.confidence_scores = confidence_scores
            document.processing_time = processing_time
            document.status = "completed"
            document.progress = 100
            document.completed_at = datetime.utcnow()
            document.updated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(document)
            
            # Create notification for completion
            await self._create_completion_notification(db, document)
        
        return document
    
    async def update_document_status(self, db: AsyncSession, document_id: str, 
                                   status: str, progress: int = 0, 
                                   message: Optional[str] = None):
        """Update document processing status"""
        
        result = await db.execute(
            select(Document).filter(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if document:
            document.status = status
            document.progress = progress
            document.status_message = message
            document.updated_at = datetime.utcnow()
            
            if status == "processing":
                document.processed_at = datetime.utcnow()
            elif status == "completed":
                document.completed_at = datetime.utcnow()
                document.progress = 100
            
            await db.commit()
            await db.refresh(document)
        
        return document
    
    async def delete_document(self, db: AsyncSession, document_id: str, 
                            user_id: str) -> bool:
        """Delete document and associated files"""
        
        document = await self.get_document(db, document_id, user_id)
        if not document:
            return False
        
        try:
            # Delete physical files
            if document.file_path and Path(document.file_path).exists():
                Path(document.file_path).unlink()
            
            if document.processed_file_path and Path(document.processed_file_path).exists():
                Path(document.processed_file_path).unlink()
            
            # Delete database record
            await db.delete(document)
            await db.commit()
            
            return True
            
        except Exception as e:
            await db.rollback()
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    async def save_processed_image(self, document_id: str, processed_image):
        """Save processed image to storage"""
        try:
            processed_filename = f"{document_id}_processed.png"
            processed_path = self.processed_dir / processed_filename
            
            # This would save the processed image
            # For now, we'll just create a placeholder
            processed_path.touch()
            
            return str(processed_path)
            
        except Exception as e:
            print(f"Error saving processed image: {e}")
            return None
    
    async def get_document_file_path(self, db: AsyncSession, document_id: str, 
                                   user_id: str) -> Optional[str]:
        """Get document file path for download"""
        
        document = await self.get_document(db, document_id, user_id)
        if not document:
            return None
        
        # Return processed file if available, otherwise original
        if document.processed_file_path:
            return document.processed_file_path
        
        return document.file_path
    
    async def get_user_analytics(self, db: AsyncSession, user_id: str) -> Dict[str, Any]:
        """Get user analytics and statistics"""
        
        now = datetime.utcnow()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # Total documents
        total_result = await db.execute(
            select(func.count(Document.id)).filter(Document.user_id == user_id)
        )
        total_documents = total_result.scalar()
        
        # Documents today
        today_result = await db.execute(
            select(func.count(Document.id)).filter(
                and_(
                    Document.user_id == user_id,
                    Document.created_at >= today
                )
            )
        )
        documents_today = today_result.scalar()
        
        # Documents this week
        week_result = await db.execute(
            select(func.count(Document.id)).filter(
                and_(
                    Document.user_id == user_id,
                    Document.created_at >= week_ago
                )
            )
        )
        documents_this_week = week_result.scalar()
        
        # Documents this month
        month_result = await db.execute(
            select(func.count(Document.id)).filter(
                and_(
                    Document.user_id == user_id,
                    Document.created_at >= month_ago
                )
            )
        )
        documents_this_month = month_result.scalar()
        
        # Average processing time
        avg_time_result = await db.execute(
            select(func.avg(Document.processing_time)).filter(
                and_(
                    Document.user_id == user_id,
                    Document.processing_time.isnot(None)
                )
            )
        )
        avg_processing_time = avg_time_result.scalar() or 0.0
        
        # Success rate
        completed_result = await db.execute(
            select(func.count(Document.id)).filter(
                and_(
                    Document.user_id == user_id,
                    Document.status == "completed"
                )
            )
        )
        completed_documents = completed_result.scalar()
        
        success_rate = (completed_documents / total_documents * 100) if total_documents > 0 else 0.0
        
        # Most common bank (from extracted fields)
        # This would require more complex JSON querying
        most_common_bank = await self._get_most_common_bank(db, user_id)
        
        # Processing statistics by status
        processing_stats = await self._get_processing_stats(db, user_id)
        
        return {
            "total_documents": total_documents,
            "documents_today": documents_today,
            "documents_this_week": documents_this_week,
            "documents_this_month": documents_this_month,
            "average_processing_time": round(avg_processing_time, 2),
            "success_rate": round(success_rate, 1),
            "most_common_bank": most_common_bank,
            "processing_stats": processing_stats
        }
    
    async def _get_most_common_bank(self, db: AsyncSession, user_id: str) -> Optional[str]:
        """Get most commonly processed bank"""
        # This would require JSON field querying which is database-specific
        # For now, return a placeholder
        return "Nepal Bank Limited"
    
    async def _get_processing_stats(self, db: AsyncSession, user_id: str) -> Dict[str, Any]:
        """Get processing statistics by status"""
        
        stats_query = await db.execute(
            select(
                Document.status,
                func.count(Document.id).label('count')
            ).filter(
                Document.user_id == user_id
            ).group_by(Document.status)
        )
        
        stats = dict(stats_query.fetchall())
        
        return {
            "pending": stats.get("pending", 0),
            "processing": stats.get("processing", 0),
            "completed": stats.get("completed", 0),
            "failed": stats.get("failed", 0)
        }
    
    async def _create_completion_notification(self, db: AsyncSession, document: Document):
        """Create notification when document processing is completed"""
        
        # Determine notification type based on success
        if document.status == "completed":
            notification_type = "success"
            title = "Document Processing Completed"
            message = f"Your document '{document.original_filename}' has been processed successfully."
        else:
            notification_type = "error"
            title = "Document Processing Failed"
            message = f"Failed to process document '{document.original_filename}'. Please try again."
        
        notification = Notification(
            user_id=document.user_id,
            title=title,
            message=message,
            type=notification_type,
            notification_metadata={
                "document_id": document.id,
                "document_filename": document.original_filename,
                "processing_time": document.processing_time
            }
        )
        
        db.add(notification)
        await db.commit()
    
    async def search_documents(self, db: AsyncSession, user_id: str, 
                             query: Optional[str] = None,
                             document_type: Optional[str] = None,
                             status: Optional[str] = None,
                             date_from: Optional[datetime] = None,
                             date_to: Optional[datetime] = None,
                             page: int = 1, limit: int = 20) -> DocumentPagination:
        """Search user's documents"""
        
        # Base query
        base_query = select(Document).filter(Document.user_id == user_id)
        
        # Apply filters
        conditions = []
        
        if document_type:
            conditions.append(Document.document_type == document_type)
        
        if status:
            conditions.append(Document.status == status)
        
        if date_from:
            conditions.append(Document.created_at >= date_from)
        
        if date_to:
            conditions.append(Document.created_at <= date_to)
        
        # Text search in filename
        if query:
            conditions.append(
                or_(
                    Document.original_filename.ilike(f"%{query}%"),
                    Document.filename.ilike(f"%{query}%")
                )
            )
        
        if conditions:
            base_query = base_query.filter(and_(*conditions))
        
        # Count total results
        count_query = select(func.count(Document.id)).filter(Document.user_id == user_id)
        if conditions:
            count_query = count_query.filter(and_(*conditions))
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination and ordering
        base_query = base_query.order_by(Document.created_at.desc())
        offset = (page - 1) * limit
        base_query = base_query.offset(offset).limit(limit)
        
        # Execute query
        result = await db.execute(base_query)
        documents = result.scalars().all()
        
        return DocumentPagination(documents, total, page, limit)
    
    async def cleanup_old_documents(self, db: AsyncSession, days: int = 30):
        """Cleanup old documents and files"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Find old completed documents
        old_docs_result = await db.execute(
            select(Document).filter(
                and_(
                    Document.status == "completed",
                    Document.completed_at < cutoff_date
                )
            )
        )
        
        old_documents = old_docs_result.scalars().all()
        
        for document in old_documents:
            try:
                # Delete physical files
                if document.file_path and Path(document.file_path).exists():
                    Path(document.file_path).unlink()
                
                if document.processed_file_path and Path(document.processed_file_path).exists():
                    Path(document.processed_file_path).unlink()
                
                # Delete database record
                await db.delete(document)
                
            except Exception as e:
                print(f"Error cleaning up document {document.id}: {e}")
        
        await db.commit()
        return len(old_documents)