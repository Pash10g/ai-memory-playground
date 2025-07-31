#!/bin/bash

# AI Memory Service Startup Script
# This script helps you start both the API service and the Streamlit app

set -e

echo "🧠 AI Memory Service Startup Script"
echo "===================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please copy sample.env to .env and configure it."
    echo "   cp sample.env .env"
    echo "   # Then edit .env with your MongoDB URI and AWS credentials"
    exit 1
fi

echo ""
echo "🚀 Starting services..."
echo ""

# Function to kill background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the API service in the background
echo "🔥 Starting AI Memory API service on http://localhost:8182..."
python main.py &
API_PID=$!

# Wait a moment for the API to start
sleep 3

# Start the Streamlit app
echo "🎨 Starting Streamlit app on http://localhost:8501..."
streamlit run streamlit_app.py --server.port 8501 &
STREAMLIT_PID=$!

echo ""
echo "✅ Both services are running!"
echo ""
echo "🔗 Access URLs:"
echo "   • AI Memory API: http://localhost:8182"
echo "   • API Documentation: http://localhost:8182/docs"
echo "   • Streamlit App: http://localhost:8501"
echo ""
echo "📝 Press Ctrl+C to stop both services"
echo ""

# Wait for background processes
wait
