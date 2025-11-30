#!/bin/bash

# Backend Deployment Script for existing Droplet with Frontend
# This will deploy backend API alongside your existing frontend

set -e

echo "🚀 Deploying Backend API alongside existing frontend..."

# Variables
BACKEND_DIR="/var/www/bank-ocr-backend"
BACKEND_PORT=8000
FRONTEND_DIR="/var/www/html"  # Adjust if your frontend is elsewhere

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Step 1: Creating backend directory...${NC}"
sudo mkdir -p $BACKEND_DIR
cd $BACKEND_DIR

echo -e "${GREEN}Step 2: Copying backend files...${NC}"
# You'll upload your backend files here

echo -e "${GREEN}Step 3: Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

echo -e "${GREEN}Step 4: Creating systemd service...${NC}"
sudo tee /etc/systemd/system/bank-ocr-backend.service > /dev/null <<EOF
[Unit]
Description=Bank OCR Backend API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=$BACKEND_DIR
Environment="PATH=$BACKEND_DIR/venv/bin"
ExecStart=$BACKEND_DIR/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:$BACKEND_PORT

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}Step 5: Starting backend service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable bank-ocr-backend
sudo systemctl start bank-ocr-backend

echo -e "${YELLOW}Backend deployed on port $BACKEND_PORT${NC}"
echo -e "${YELLOW}Configure Nginx to proxy /api requests to backend${NC}"