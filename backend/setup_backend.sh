#!/bin/bash

# Backend Setup Script - Run this AFTER uploading files
# Run ON YOUR DROPLET

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BACKEND_DIR="/var/www/bank-ocr-backend"
DROPLET_IP="157.230.145.242"

echo -e "${GREEN}Setting up Backend Application...${NC}"

cd $BACKEND_DIR

# Step 1: Install Python packages
echo -e "\n${YELLOW}Installing Python packages...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

# Step 2: Create production .env file
echo -e "\n${YELLOW}Creating production environment file...${NC}"
cat > .env << EOL
# Production Configuration
APP_NAME=Bank OCR System
VERSION=2.0.0
DEBUG=false

# Security
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS - Allow your frontend IP
CORS_ORIGINS=["http://$DROPLET_IP","https://$DROPLET_IP"]
ALLOWED_HOSTS=["$DROPLET_IP","localhost","127.0.0.1"]

# Database - Using SQLite for simplicity
DATABASE_URL=sqlite+aiosqlite:///./bank_ocr.db
DATABASE_ECHO=false

# File paths
UPLOAD_DIR=$BACKEND_DIR/uploads
PROCESSED_DIR=$BACKEND_DIR/processed

# Model paths
MODEL_BASE_PATH=$BACKEND_DIR/models
CHARACTER_MODEL_PATH=$BACKEND_DIR/models/complete_ocr_character_model.h5
SEQUENCE_MODEL_PATH=$BACKEND_DIR/models/complete_ocr_sequence_model.h5
DETECTION_MODEL_PATH=$BACKEND_DIR/models/complete_ocr_detection_model.h5
METADATA_PATH=$BACKEND_DIR/models/complete_ocr_system_metadata.json
EOL

# Step 3: Create systemd service
echo -e "\n${YELLOW}Creating systemd service...${NC}"
sudo tee /etc/systemd/system/bank-ocr-backend.service > /dev/null << EOL
[Unit]
Description=Bank OCR Backend API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=$BACKEND_DIR
Environment="PATH=$BACKEND_DIR/venv/bin"
ExecStart=$BACKEND_DIR/venv/bin/gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000 --timeout 120

Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Step 4: Set permissions
echo -e "\n${YELLOW}Setting permissions...${NC}"
sudo chown -R www-data:www-data $BACKEND_DIR
sudo chmod -R 755 $BACKEND_DIR

# Step 5: Start the service
echo -e "\n${YELLOW}Starting backend service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable bank-ocr-backend
sudo systemctl start bank-ocr-backend

# Step 6: Check status
sleep 2
sudo systemctl status bank-ocr-backend --no-pager

echo -e "\n${GREEN}✓ Backend service started!${NC}"
echo -e "${YELLOW}Now run: sudo bash configure_nginx.sh${NC}"