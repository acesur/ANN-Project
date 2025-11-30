#!/bin/bash

# Nginx Configuration Script
# This updates your existing Nginx to proxy backend requests

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DROPLET_IP="157.230.145.242"

echo -e "${GREEN}Configuring Nginx for Backend API...${NC}"

# Backup existing Nginx config
echo -e "\n${YELLOW}Backing up existing Nginx config...${NC}"
sudo cp /etc/nginx/sites-available/default /etc/nginx/sites-available/default.backup.$(date +%Y%m%d)

# Create new Nginx configuration
echo -e "\n${YELLOW}Creating Nginx configuration...${NC}"
sudo tee /etc/nginx/sites-available/default > /dev/null << 'EOL'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    server_name 157.230.145.242;
    
    # Frontend (Angular) - Root path
    location / {
        root /var/www/html;
        try_files $uri $uri/ /index.html;
    }
    
    # Backend API endpoints
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for file uploads
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
    
    # API Documentation
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_set_header Host $host;
    }
    
    location /redoc {
        proxy_pass http://127.0.0.1:8000/redoc;
        proxy_set_header Host $host;
    }
    
    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
        proxy_set_header Host $host;
    }
    
    # Static files for uploads
    location /uploads {
        alias /var/www/bank-ocr-backend/uploads;
        expires 30d;
    }
    
    # File upload size
    client_max_body_size 10M;
}
EOL

# Test Nginx configuration
echo -e "\n${YELLOW}Testing Nginx configuration...${NC}"
sudo nginx -t

# Reload Nginx
echo -e "\n${YELLOW}Reloading Nginx...${NC}"
sudo systemctl reload nginx

echo -e "\n${GREEN}✓ Nginx configured successfully!${NC}"
echo -e "\n${GREEN}==================================${NC}"
echo -e "${GREEN}  DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}==================================${NC}"
echo -e "\n${YELLOW}Your services are available at:${NC}"
echo -e "  Frontend: ${GREEN}http://$DROPLET_IP/${NC}"
echo -e "  Backend API: ${GREEN}http://$DROPLET_IP/api${NC}"
echo -e "  API Docs: ${GREEN}http://$DROPLET_IP/docs${NC}"
echo -e "  Health Check: ${GREEN}http://$DROPLET_IP/health${NC}"
echo -e "\n${YELLOW}Test your backend:${NC}"
echo -e "  ${GREEN}curl http://$DROPLET_IP/health${NC}"