"""
Logging Middleware
=================
Custom middleware for request/response logging and monitoring
"""

import time
import uuid
import json
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime


# Configure logger
logger = logging.getLogger("api_requests")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses"""
    
    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = {
            "/docs", "/redoc", "/openapi.json", 
            "/static", "/health", "/favicon.ico"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        timestamp = datetime.utcnow()
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Log request
        request_log = {
            "request_id": request_id,
            "timestamp": timestamp.isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "headers": dict(request.headers),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length", 0)
        }
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        logger.info(f"REQUEST {request_id}: {request.method} {request.url.path}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            response_log = {
                "request_id": request_id,
                "status_code": response.status_code,
                "processing_time": round(process_time, 4),
                "response_headers": dict(response.headers),
                "content_length": response.headers.get("content-length", 0)
            }
            
            # Add processing time to response headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            # Log based on status code
            if response.status_code >= 400:
                logger.error(f"RESPONSE {request_id}: {response.status_code} in {process_time:.4f}s")
            else:
                logger.info(f"RESPONSE {request_id}: {response.status_code} in {process_time:.4f}s")
            
            # Store metrics (could be sent to monitoring service)
            await self._store_metrics(request_log, response_log)
            
            return response
            
        except Exception as e:
            # Calculate processing time for error
            process_time = time.time() - start_time
            
            logger.error(f"ERROR {request_id}: {str(e)} after {process_time:.4f}s")
            
            # Create error response
            error_response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An internal server error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            error_response.headers["X-Process-Time"] = str(process_time)
            error_response.headers["X-Request-ID"] = request_id
            
            return error_response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return "unknown"
    
    async def _store_metrics(self, request_log: dict, response_log: dict):
        """Store request/response metrics"""
        # This could be enhanced to store metrics in a database,
        # send to monitoring service, etc.
        
        # For now, we'll just log detailed metrics for certain conditions
        if response_log["processing_time"] > 5.0:  # Slow requests
            logger.warning(f"SLOW REQUEST: {request_log['request_id']} took {response_log['processing_time']}s")
        
        if response_log["status_code"] >= 500:  # Server errors
            logger.error(f"SERVER ERROR: {request_log['request_id']} returned {response_log['status_code']}")


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for security-related logging"""
    
    def __init__(self, app):
        super().__init__(app)
        self.suspicious_patterns = [
            "../../",  # Path traversal
            "<script",  # XSS attempts
            "UNION SELECT",  # SQL injection
            "OR 1=1",  # SQL injection
            "javascript:",  # XSS
            "eval(",  # Code injection
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check for suspicious patterns
        await self._check_security_threats(request)
        
        # Process request
        response = await call_next(request)
        
        # Log authentication failures
        if response.status_code == 401:
            await self._log_auth_failure(request)
        
        return response
    
    async def _check_security_threats(self, request: Request):
        """Check for potential security threats"""
        url_str = str(request.url).lower()
        
        # Check URL for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.lower() in url_str:
                logger.warning(f"SECURITY THREAT: Suspicious pattern '{pattern}' in URL: {request.url}")
                break
        
        # Check for unusual user agents
        user_agent = request.headers.get("user-agent", "").lower()
        if any(bot in user_agent for bot in ["sqlmap", "nmap", "nikto", "dirb"]):
            logger.warning(f"SECURITY THREAT: Suspicious user agent: {user_agent}")
        
        # Check for excessive header size
        total_header_size = sum(len(f"{k}: {v}") for k, v in request.headers.items())
        if total_header_size > 8192:  # 8KB limit
            logger.warning(f"SECURITY THREAT: Excessive header size: {total_header_size} bytes")
    
    async def _log_auth_failure(self, request: Request):
        """Log authentication failures"""
        client_ip = self._get_client_ip(request)
        
        logger.warning(f"AUTH FAILURE: {request.method} {request.url.path} from {client_ip}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return "unknown"


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring"""
    
    def __init__(self, app):
        super().__init__(app)
        self.slow_threshold = 2.0  # 2 seconds
        self.memory_threshold = 100 * 1024 * 1024  # 100MB
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory
        
        # Log performance metrics
        if process_time > self.slow_threshold:
            logger.warning(f"PERFORMANCE: Slow request - {request.method} {request.url.path} "
                          f"took {process_time:.4f}s")
        
        if memory_used > self.memory_threshold:
            logger.warning(f"PERFORMANCE: High memory usage - {request.method} {request.url.path} "
                          f"used {memory_used / 1024 / 1024:.2f}MB")
        
        # Add performance headers
        response.headers["X-Memory-Used"] = str(memory_used)
        response.headers["X-CPU-Time"] = str(process_time)
        
        return response