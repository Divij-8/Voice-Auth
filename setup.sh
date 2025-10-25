#!/bin/bash

echo ""
echo "========================================================================"
echo "VOICE GUARDIAN - SETUP & INTEGRATION SCRIPT"
echo "========================================================================"
echo ""

# Check Python version
echo "✓ Checking Python installation..."
python3 --version

# Install dependencies
echo ""
echo "✓ Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully!"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Create training_data directory
echo ""
echo "✓ Creating training_data directory..."
mkdir -p training_data

if [ -d "training_data" ]; then
    echo "✓ training_data directory created/verified"
else
    echo "✗ Failed to create training_data directory"
    exit 1
fi

# Create pretrained_models directory if it doesn't exist
echo ""
echo "✓ Ensuring pretrained_models directory exists..."
mkdir -p pretrained_models

# Verify all key files exist
echo ""
echo "✓ Verifying key files..."
files=("main.py" "index.html" "voice_guardian.py" "voice_guardian_enhanced.py" "frontend_server.py" "requirements.txt")
missing=0

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        missing=$((missing + 1))
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "✗ Some files are missing!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Add training data:"
echo "   cp your_voice_samples.wav training_data/"
echo ""
echo "2. Start the backend (Terminal 1):"
echo "   python main.py"
echo ""
echo "3. Start the frontend (Terminal 2):"
echo "   python frontend_server.py"
echo ""
echo "4. Open in browser:"
echo "   http://localhost:3000"
echo ""
echo "========================================================================"
echo ""
