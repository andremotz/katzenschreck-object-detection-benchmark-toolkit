#!/bin/bash
# Jetson Xavier Installation Script
# This script handles the polars dependency issue on Jetson devices

echo "🚀 Installing AI Detection Toolkit on Jetson Xavier..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install PyTorch for Jetson (ARM64/CUDA 11.8)
echo "🔥 Installing PyTorch for Jetson..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install basic requirements (YOLO only)
echo "📚 Installing basic requirements..."
pip install Pillow>=9.0.0
pip install scipy
pip install opencv-python-headless>=4.8.0
pip install numpy>=1.21.0

# Install ultralytics without dependencies first
echo "🤖 Installing ultralytics..."
pip install ultralytics --no-deps

# Try to install polars, fallback to pandas if it fails
echo "📊 Installing data processing library..."
if pip install polars --no-deps; then
    echo "✅ Polars installed successfully"
else
    echo "⚠️  Polars failed, installing pandas instead..."
    pip install pandas>=1.5.0
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO loaded successfully')"

echo "✅ Installation complete!"
echo "🎯 You can now run: python ai-processor.py /path/to/video.mp4"
echo "📝 Note: Only YOLO models are supported on Jetson (OWLv2 removed for compatibility)"
