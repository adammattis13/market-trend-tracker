#!/bin/bash

# Market Trend Tracker - Deployment Script
# =======================================

echo "📊 Market Trend Tracker - Deployment Helper"
echo "=========================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${YELLOW}Initializing Git repository...${NC}"
    git init
fi

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Found uncommitted changes${NC}"
    
    # Remove database files
    echo -e "${GREEN}Removing database files...${NC}"
    rm -f *.db
    rm -f *.sqlite
    rm -f *.sqlite3
    
    # Remove log files
    echo -e "${GREEN}Removing log files...${NC}"
    rm -f *.log
    
    # Remove pycache
    echo -e "${GREEN}Cleaning Python cache...${NC}"
    find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
    
    # Add files to git
    echo -e "${GREEN}Adding files to Git...${NC}"
    git add .
    
    # Commit
    echo -e "${GREEN}Creating commit...${NC}"
    read -p "Enter commit message (default: 'Update Market Trend Tracker'): " commit_msg
    commit_msg=${commit_msg:-"Update Market Trend Tracker"}
    git commit -m "$commit_msg"
fi

# Check if remote exists
if ! git remote | grep -q origin; then
    echo -e "${YELLOW}No remote origin found${NC}"
    read -p "Enter your GitHub repository URL: " repo_url
    git remote add origin "$repo_url"
fi

# Push to GitHub
echo -e "${GREEN}Pushing to GitHub...${NC}"
git push -u origin main || git push -u origin master

echo ""
echo -e "${GREEN}✅ Code pushed to GitHub successfully!${NC}"
echo ""

# Deployment options
echo "🚀 Deployment Options:"
echo "====================="
echo ""
echo "1. Streamlit Cloud (Recommended - FREE)"
echo "   - Go to: https://share.streamlit.io"
echo "   - Click 'New app'"
echo "   - Select your repository"
echo "   - Add your FINNHUB_API_KEY in secrets"
echo "   - Deploy!"
echo ""
echo "2. Run Locally"
echo "   - streamlit run dashboard.py"
echo "   - or: python3 -m streamlit run dashboard.py"
echo ""
echo "3. Docker"
echo "   - docker build -t market-tracker ."
echo "   - docker run -p 8501:8501 -e FINNHUB_API_KEY=your_key market-tracker"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  Warning: No .env file found${NC}"
    echo "Create .env file with: FINNHUB_API_KEY=your_api_key_here"
fi

# Final checks
echo ""
echo "📋 Pre-deployment Checklist:"
echo -e "[ ] ${GREEN}Git repository initialized${NC}"
echo -e "[ ] ${GREEN}Code pushed to GitHub${NC}"
echo -e "[ ] Database files excluded from Git"
echo -e "[ ] .env file created (local only)"
echo -e "[ ] API key ready for production"
echo ""
echo -e "${GREEN}Ready for deployment! 🎉${NC}"