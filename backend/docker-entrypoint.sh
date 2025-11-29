#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Bank OCR Backend...${NC}"

# Wait for database to be ready
echo -e "${YELLOW}📊 Waiting for database...${NC}"
while ! pg_isready -h postgres -p 5432 -U bankocr; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done
echo -e "${GREEN}✅ Database is ready!${NC}"

# Wait for Redis
echo -e "${YELLOW}🗄️ Waiting for Redis...${NC}"
while ! redis-cli -h redis ping > /dev/null 2>&1; do
  echo "Waiting for Redis..."
  sleep 2
done
echo -e "${GREEN}✅ Redis is ready!${NC}"

# Run database migrations
echo -e "${YELLOW}🔄 Running database migrations...${NC}"
python -c "
import asyncio
from models.database import init_db
asyncio.run(init_db())
print('✅ Database initialized')
"

# Create default directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p uploads processed logs static models
echo -e "${GREEN}✅ Directories created${NC}"

# Check if models exist
echo -e "${YELLOW}🤖 Checking ML models...${NC}"
if [ ! -f "models/complete_ocr_character_model.h5" ]; then
    echo -e "${YELLOW}⚠️ ML models not found in models/ directory${NC}"
    echo -e "${YELLOW}Please copy your trained models to the models/ directory${NC}"
fi

# Start the application
echo -e "${GREEN}🎯 Starting application...${NC}"
exec "$@"