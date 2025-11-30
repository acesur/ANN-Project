# Deploying Backend on Existing Droplet with Frontend

## Quick Deployment Steps

### 1. **Connect to your Droplet**
```bash
ssh root@your-droplet-ip
```

### 2. **Install Python 3.12 (if not installed)**
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip
```

### 3. **Create backend directory**
```bash
sudo mkdir -p /var/www/bank-ocr-backend
cd /var/www/bank-ocr-backend
```

### 4. **Upload your backend code**
From your local machine:
```bash
# Option A: Using SCP (from your local terminal)
scp -r /mnt/c/Users/LegendChaudhary/Documents/ANN-Project/backend/* root@your-droplet-ip:/var/www/bank-ocr-backend/

# Option B: Using Git (on the droplet)
git clone https://github.com/yourusername/your-repo.git /var/www/bank-ocr-backend
```

### 5. **Set up Python environment**
On the droplet:
```bash
cd /var/www/bank-ocr-backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

### 6. **Configure environment variables**
```bash
# Create production .env file
nano /var/www/bank-ocr-backend/.env
```

Add these (update values):
```env
DEBUG=false
SECRET_KEY=your-generated-secret-key
DATABASE_URL=sqlite+aiosqlite:///./bank_ocr.db
CORS_ORIGINS=["http://your-domain.com","https://your-domain.com"]
```

### 7. **Create systemd service**
```bash
sudo nano /etc/systemd/system/bank-ocr-backend.service
```

Paste:
```ini
[Unit]
Description=Bank OCR Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/bank-ocr-backend
Environment="PATH=/var/www/bank-ocr-backend/venv/bin"
ExecStart=/var/www/bank-ocr-backend/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000

[Install]
WantedBy=multi-user.target
```

### 8. **Start the backend service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable bank-ocr-backend
sudo systemctl start bank-ocr-backend
sudo systemctl status bank-ocr-backend
```

### 9. **Update Nginx configuration**
```bash
sudo nano /etc/nginx/sites-available/default
```

Add the location blocks from nginx_config.conf to proxy `/api` to your backend.

### 10. **Reload Nginx**
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 11. **Update your Angular frontend**
In your Angular app, update the API URL to use relative paths:
```typescript
// Instead of: http://localhost:8000/api
// Use: /api
apiUrl = '/api';
```

## Test Your Deployment

1. **Check backend health:**
   ```bash
   curl http://your-domain.com/health
   ```

2. **View API docs:**
   - http://your-domain.com/docs
   - http://your-domain.com/redoc

3. **Check logs if issues:**
   ```bash
   sudo journalctl -u bank-ocr-backend -f
   ```

## SSL Certificate (Important!)
If you haven't already, set up SSL:
```bash
sudo certbot --nginx -d your-domain.com
```

## File Permissions
```bash
sudo chown -R www-data:www-data /var/www/bank-ocr-backend
sudo chmod -R 755 /var/www/bank-ocr-backend
```