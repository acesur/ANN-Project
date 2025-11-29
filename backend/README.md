# Bank OCR System - Backend API

Production-ready FastAPI backend for the Bank OCR System with Angular frontend integration.

## 🚀 Features

- **Advanced OCR Processing**: Multi-language support (English/Nepali) with confidence scoring
- **Authentication & Security**: JWT-based auth with rate limiting and security middleware
- **Document Management**: Upload, process, store, and manage bank documents
- **Real-time Processing**: Async processing with status tracking
- **Analytics Dashboard**: User analytics and processing statistics
- **Production Ready**: Docker deployment with monitoring and logging

## 📋 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh token
- `GET /api/auth/me` - Get current user

### OCR Processing
- `POST /api/ocr/upload` - Upload and process document
- `GET /api/ocr/status/{document_id}` - Get processing status
- `POST /api/ocr/reprocess/{document_id}` - Reprocess document

### Document Management
- `GET /api/documents` - List user documents (with pagination)
- `GET /api/documents/{document_id}` - Get document details
- `DELETE /api/documents/{document_id}` - Delete document
- `GET /api/documents/{document_id}/download` - Download processed image

### Analytics
- `GET /api/analytics/summary` - User analytics summary

### System
- `GET /api/health` - Health check
- `GET /api/config` - Public configuration
- `GET /docs` - API documentation (Swagger)

## 🛠️ Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL (or SQLite for development)
- Redis (optional, for caching)
- Your trained OCR models

### Installation

1. **Clone and setup**:
```bash
cd backend
cp .env.example .env
# Edit .env with your configuration
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup database**:
```bash
# For PostgreSQL
createdb bankocr_db

# For SQLite (development)
# Database will be created automatically
```

4. **Copy your models**:
```bash
# Copy your trained models to models/ directory
cp path/to/your/models/* models/
```

5. **Run the server**:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 🐳 Docker Deployment

### Development
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d
```

### Production
```bash
# Update environment variables in .env
# Copy your models to models/ directory
# Then deploy
docker-compose -f docker-compose.prod.yml up -d
```

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Security
SECRET_KEY=your-super-secret-key
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# CORS (for Angular frontend)
CORS_ORIGINS=http://localhost:4200,https://your-domain.com

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_FILE_FORMATS=jpg,jpeg,png

# OCR
OCR_CONFIDENCE_THRESHOLD=0.5
```

## 🔗 Angular Frontend Integration

### Example Angular Service

```typescript
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class OcrService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  // Upload document for OCR processing
  uploadDocument(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.http.post(`${this.apiUrl}/ocr/upload`, formData);
  }

  // Get processing status
  getStatus(documentId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/ocr/status/${documentId}`);
  }

  // Get user documents
  getDocuments(page: number = 1, limit: number = 20): Observable<any> {
    return this.http.get(`${this.apiUrl}/documents?page=${page}&limit=${limit}`);
  }
}
```

### Example Upload Component

```typescript
export class DocumentUploadComponent {
  selectedFile: File | null = null;
  processing = false;
  result: any = null;

  constructor(private ocrService: OcrService) {}

  onFileSelect(event: any) {
    this.selectedFile = event.target.files[0];
  }

  async uploadDocument() {
    if (!this.selectedFile) return;

    this.processing = true;
    
    try {
      const response = await this.ocrService.uploadDocument(this.selectedFile).toPromise();
      this.result = response.data;
      console.log('Extracted fields:', this.result.extracted_fields);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      this.processing = false;
    }
  }
}
```

## 📊 Response Format

### Successful OCR Response
```json
{
  "document_id": "uuid-here",
  "extracted_fields": {
    "bank_name": {
      "value": "Nepal Bank Limited",
      "confidence": 0.95
    },
    "account_number": {
      "value": "1234567890123456",
      "confidence": 0.92
    },
    "amount": {
      "value": "25000",
      "confidence": 0.88
    },
    "date": {
      "value": "2024-01-15",
      "confidence": 0.90
    }
  },
  "confidence_scores": {
    "bankName": 0.95,
    "accountNumber": 0.92,
    "amount": 0.88,
    "date": 0.90
  },
  "processing_time": 1.2,
  "status": "completed"
}
```

### Error Response
```json
{
  "error": "INVALID_FILE_FORMAT",
  "message": "Invalid file format. Only JPG, PNG, JPEG are supported",
  "timestamp": "2024-11-29T10:30:00Z"
}
```

## 🔐 Authentication

The API uses JWT tokens for authentication:

```typescript
// Login
const loginResponse = await this.http.post('/api/auth/login', {
  email: 'user@example.com',
  password: 'password'
}).toPromise();

const token = loginResponse.access_token;

// Use token in requests
const headers = new HttpHeaders({
  'Authorization': `Bearer ${token}`
});
```

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Metrics
- Prometheus metrics: `http://localhost:9090`
- Grafana dashboard: `http://localhost:3000`

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=.

# Test specific endpoint
pytest tests/test_auth.py -v
```

## 🔧 Development

### Adding New Endpoints

1. **Define schema** in `models/schemas.py`:
```python
class NewRequest(BaseModel):
    field: str
```

2. **Add endpoint** in `main.py`:
```python
@app.post("/api/new-endpoint")
async def new_endpoint(request: NewRequest):
    return {"result": "success"}
```

3. **Add tests** in `tests/`:
```python
def test_new_endpoint():
    response = client.post("/api/new-endpoint", json={"field": "value"})
    assert response.status_code == 200
```

### Database Migrations

```python
# Add new model
class NewModel(Base):
    __tablename__ = "new_table"
    id = Column(String, primary_key=True)

# Run migration
python -c "
import asyncio
from models.database import init_db
asyncio.run(init_db())
"
```

## 🚨 Security Features

- **Rate Limiting**: 60 requests per minute per IP
- **CORS Protection**: Configurable origins
- **Input Validation**: Pydantic schemas
- **File Validation**: Size and type checking
- **Security Headers**: Added automatically
- **Audit Logging**: All actions logged

## 📝 Logging

Logs are structured and include:
- Request/response logging
- Performance monitoring
- Security events
- Error tracking

View logs:
```bash
# Real-time
tail -f logs/app.log

# In Docker
docker-compose logs -f app
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test: `pytest`
4. Commit: `git commit -am 'Add new feature'`
5. Push: `git push origin feature/new-feature`
6. Create Pull Request

## 📞 Support

- Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`
- Logs: `logs/app.log`

## 📄 License

MIT License - see LICENSE file for details.