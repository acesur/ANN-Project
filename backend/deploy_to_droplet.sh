#!/bin/bash

# Deployment Script for your Droplet at 157.230.145.242
# This script should be run ON YOUR DROPLET

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

DROPLET_IP="157.230.145.242"
BACKEND_DIR="/var/www/bank-ocr-backend"

echo -e "${GREEN}==================================${NC}"
echo -e "${GREEN}  Deploying Backend to $DROPLET_IP${NC}"
echo -e "${GREEN}==================================${NC}"

# Step 1: Install Python and dependencies
echo -e "\n${YELLOW}Installing Python and dependencies...${NC}"
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx sqlite3

# Step 2: Create backend directory structure
echo -e "\n${YELLOW}Creating backend directories...${NC}"
sudo mkdir -p $BACKEND_DIR
sudo mkdir -p $BACKEND_DIR/uploads
sudo mkdir -p $BACKEND_DIR/processed
sudo mkdir -p $BACKEND_DIR/logs
sudo mkdir -p $BACKEND_DIR/models

# Step 3: Set up Python virtual environment
echo -e "\n${YELLOW}Setting up Python environment...${NC}"
cd $BACKEND_DIR
python3 -m venv venv
source venv/bin/activate

# Step 4: Message for manual file upload
echo -e "\n${RED}IMPORTANT: Now upload your backend files!${NC}"
echo -e "${YELLOW}From your LOCAL machine, run:${NC}"
echo -e "${GREEN}cd /mnt/c/Users/LegendChaudhary/Documents/ANN-Project/backend${NC}"
echo -e "${GREEN}scp -r * root@$DROPLET_IP:$BACKEND_DIR/${NC}"

echo -e "\n${YELLOW}After uploading files, continue with setup_backend.sh${NC}"