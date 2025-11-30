# Deployment Guide for Droplet (157.230.145.242)

## Quick Deployment Steps

### From Your Local Machine:

#### 1. Upload all scripts and backend files to your droplet:
```bash
# From Windows PowerShell or WSL
cd C:\Users\LegendChaudhary\Documents\ANN-Project\backend

# Upload everything
scp -r * root@157.230.145.242:/var/www/bank-ocr-backend/
```

### On Your Droplet:

#### 2. SSH into your droplet:
```bash
ssh root@157.230.145.242
```

#### 3. Run the deployment scripts in order:
```bash
cd /var/www/bank-ocr-backend

# Step 1: Initial setup
bash deploy_to_droplet.sh

# Step 2: Setup backend (after files are uploaded)
bash setup_backend.sh

# Step 3: Configure Nginx
bash configure_nginx.sh
```

## Testing Your Deployment

### 1. Check if backend is running:
```bash
# On droplet
sudo systemctl status bank-ocr-backend

# From your local machine
curl http://157.230.145.242/health
```

### 2. View API documentation:
Open in browser:
- http://157.230.145.242/docs (Swagger UI)
- http://157.230.145.242/redoc (ReDoc)

### 3. Test API endpoints:
```bash
# Register a user
curl -X POST "http://157.230.145.242/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "TestPass123",
    "first_name": "Test",
    "last_name": "User"
  }'
```

## Update Your Angular Frontend

In your Angular app, update the API URL to use your backend:

**src/environments/environment.prod.ts:**
```typescript
export const environment = {
  production: true,
  apiUrl: 'http://157.230.145.242/api'  // Your backend API
};
```

**src/environments/environment.ts:**
```typescript
export const environment = {
  production: false,
  apiUrl: 'http://157.230.145.242/api'  // For testing
};
```

## Troubleshooting

### Check logs:
```bash
# Backend logs
sudo journalctl -u bank-ocr-backend -f

# Nginx logs
sudo tail -f /var/log/nginx/error.log
```

### Restart services:
```bash
# Backend
sudo systemctl restart bank-ocr-backend

# Nginx
sudo systemctl restart nginx
```

### Common Issues:

1. **Port 8000 already in use:**
   ```bash
   sudo lsof -i :8000
   sudo kill -9 <PID>
   ```

2. **Permission errors:**
   ```bash
   sudo chown -R www-data:www-data /var/www/bank-ocr-backend
   sudo chmod -R 755 /var/www/bank-ocr-backend
   ```

3. **Module not found errors:**
   ```bash
   cd /var/www/bank-ocr-backend
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## URLs After Deployment

- Frontend: http://157.230.145.242/
- Backend API: http://157.230.145.242/api
- API Docs: http://157.230.145.242/docs
- Health Check: http://157.230.145.242/health