# AI Detection Model Comparison

This project uses Hugging Face OWLv2 (Open-World Localization) and YOLO models for automatic detection of cats and dogs in video recordings. The system analyzes video material through image sequence extraction and AI-based object detection for offline model comparison and evaluation.

## General Approach

### 1. Video-to-Image-Sequence Conversion
The project uses a two-stage approach for efficient processing of video files:

#### Single Video Conversion (`convert_video_to_image_sequence.py`)
- Converts individual MP4 videos into image sequences (JPG files)
- Extracts frames with configurable parameters (quality, frame skip)
- Saves images in a `_frames` subfolder next to the video file
- Progress display and performance optimization

#### Batch Processing (`convert_batch.py`)
- Recursively scans a directory for all video files
- Automatically checks if `_frames` folders already exist
- Automatically converts all videos without corresponding image sequences
- Overview and summary of all processed files

### 2. AI-based Object Detection (`ai-processor.py`)

#### Available AI Models
The script supports two different object detection models:

**OWLv2 (Default)**
- **Model**: `google/owlv2-base-patch16-ensemble`
- **Framework**: Hugging Face Transformers with PyTorch
- **Detection Categories**: Cats and dogs (text-based detection)
- **Advantages**: More flexible class definition, better text-based object detection
- **Disadvantages**: Slower processing

**YOLO v8/v11/v12 (Optional)**
- **Models**: `yolo8n.pt`, `yolo8s.pt`, `yolo8m.pt`, `yolo8l.pt`, `yolo8x.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- **Framework**: Ultralytics YOLO
- **Detection Categories**: Cats and dogs (from 80 COCO classes)
- **Advantages**: Significantly faster processing, better performance
- **Disadvantages**: Fixed class definition

#### Model Selection
```bash
# Use default OWLv2
python ai-processor.py /path/to/video.mp4

# Use YOLO (faster)
python ai-processor.py --model yolo /path/to/video.mp4

# Use specific YOLO model (YOLOv8x - default)
python ai-processor.py --model yolo --yolo-model yolo8x.pt /path/to/video.mp4

# Use YOLOv11 model
python ai-processor.py --model yolo --yolo-model yolo11l.pt /path/to/video.mp4

# Process with skip-frames for 5x faster processing
python ai-processor.py --model yolo --skip-frames 5 /path/to/video.mp4
```

#### YOLO Model Overview
| Model | Size | Speed | Accuracy | Usage |
|--------|-------|----------------|-------------|------------|
| `yolov8n.pt` | Nano 
| `yolov8s.pt` | Small 
| `yolov8m.pt` | Medium 
| `yolov8l.pt` | Large
| `yolov8x.pt` | Extra Large
| `yolo11n.pt` | Nano
| `yolo11s.pt` | Small
| `yolo11m.pt` | Medium
| `yolo11l.pt` | Large 
| `yolo11x.pt` | Extra Large

### Using YOLO Models

#### Quick Start with YOLO
```bash
# Basic YOLO usage (uses yolo11n.pt by default)
python ai-processor.py --model yolo /path/to/video.mp4

# Or process frame folders directly
python ai-processor.py --model yolo /path/to/video_frames/

# Fast processing with skip-frames (process every 5th frame = 5x speed)
python ai-processor.py --model yolo --skip-frames 5 /path/to/video.mp4
```

#### Specifying YOLO Model Variants
```bash
# Use nano model (fastest, lowest accuracy)
python ai-processor.py --model yolo --yolo-model yolo8n.pt /path/to/video.mp4

# Use small model (good balance)
python ai-processor.py --model yolo --yolo-model yolo11s.pt /path/to/video.mp4

# Use large model (high accuracy)
python ai-processor.py --model yolo --yolo-model yolo12l.pt /path/to/video.mp4

# Use extra large model (maximum accuracy)
python ai-processor.py --model yolo --yolo-model yolo12x.pt /path/to/video.mp4
```

#### YOLO Model Download
YOLO models are automatically downloaded on first use:
- Models are cached in your local environment
- First run may take longer due to download
- Subsequent runs use cached models for faster startup

#### YOLO vs OWLv2 Comparison
```bash
# Process same video with both models for comparison
python ai-processor.py --model owlv2 /path/to/video.mp4
python ai-processor.py --model yolo /path/to/video.mp4

# Results will be saved in the same folder with specific model suffixes:
# - detection_results/video_detection_results_owlv2.json
# - detection_results/video_detection_results_yolo12x.json
```

#### Performance Recommendations
- **Large datasets**: Use `yolo11n.pt` or `yolo11s.pt` for speed
- **Very large datasets**: Use `--skip-frames 5` or higher to process only every N-th frame
- **High accuracy needed**: Use `yolo11l.pt` or `yolo11x.pt`
- **GPU available**: All YOLO models benefit significantly from GPU acceleration
- **CPU only**: Nano and Small models are most practical, combine with `--skip-frames` for faster processing

#### Detection Process
1. **Frame Loading**: Sequential processing of extracted image sequence
2. **AI Analysis**: Object detection with configurable threshold (0.1)
3. **Bounding Box Extraction**: Precise localization of detected objects
4. **Confidence Assessment**: Confidence score for each detection

#### Output and Documentation
- **JSON Export**: Structured storage of all detection results
- **Metadata**: Processing statistics and configuration parameters (including used model)
- **Detailed Logs**: Progress display with time estimates
- **Performance Metrics**: Detection rate and processing speed
- **Model-specific Files**: JSON files with model suffix for easy identification

#### Results Folder Structure
```
/path/to/video/
├── video.mp4
└── detection_results/           # All detection results
    ├── video_detection_results_owlv2.json
    ├── video_detection_results_yolo12x.json
    └── video_detection_results_yolo11n.json
```

This enables parallel processing with both models and direct comparison of results in a single folder.

### 3. Results Data Structure

The generated JSON files contain:
- **Metadata**: Timestamp, processing statistics, configuration
- **Frame Information**: Image size, filename, path
- **Detection Details**: Object labels, confidence scores, bounding box coordinates

### 4. System Requirements

#### Base Requirements (for OWLv2)
```
transformers>=4.30.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
scipy
opencv-python>=4.8.0
```

#### Additional Requirements for YOLO Support
```
ultralytics>=8.0.0
numpy>=1.21.0
```

**Installation with YOLO support:**
```bash
pip install -r requirements.txt
```

## Installation

To install the scripts on a new machine, follow these steps:

### Prerequisites
- Python 3.8+ installed
- Git installed
- Access to the repository

### Step-by-Step Installation

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd katzenschreck-object-detection-benchmark-toolkit
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   **On Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**
   ```bash
   python -c "import torch; import transformers; print('Installation successful!')"
   ```

### Additional Notes for GPU Support

For optimal performance on powerful hardware:

```bash
# For CUDA-capable GPUs (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Jetson Xavier Installation

This project is fully compatible with NVIDIA Jetson Xavier devices running Ubuntu 20.04 and Python 3.8. The AI processor supports both OWLv2 and YOLO models with CUDA acceleration.

#### Prerequisites for Jetson Xavier
- Ubuntu 20.04 LTS
- Python 3.8
- CUDA 11.8 (included with JetPack)
- cuDNN (included with JetPack)
- Virtual environment (recommended)
- https://docs.nvidia.com/jetson/jetpack/5.1.5/install-setup/index.html

#### Step-by-Step Jetson Installation

1. **Activate virtual environment** (if using one):
   ```bash
   source venv/bin/activate
   ```

2. **Install PyTorch for Jetson** (critical for ARM64 compatibility):
   ```bash
   # Install PyTorch with CUDA 11.8 support for Jetson
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install other dependencies**:
   ```bash
   # Use Jetson-specific requirements (uses pandas instead of polars)
   pip install -r requirements-jetson.txt
   
   # If you need polars specifically, install it separately:
   # pip install polars --no-deps
   
   # Alternative: try minimal requirements if issues persist:
   # pip install -r requirements-minimal.txt
   # pip install ultralytics --no-deps
   ```

4. **Install OpenCV for Jetson** (if needed):
   ```bash
   pip install opencv-python-headless
   ```

5. **Verify CUDA support**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
   ```

#### Jetson-Specific Usage

The AI processor automatically detects CUDA on Jetson and uses GPU acceleration:

```bash
# Process video with YOLO (recommended for Jetson)
python ai-processor.py /path/to/video.mp4 --model yolo --yolo-model yolov8x.pt

# Process frame folder with YOLO
python ai-processor.py /path/to/frames/ --model yolo --yolo-model yolov8s.pt

# Fast processing with skip-frames (5x faster)
python ai-processor.py /path/to/video.mp4 --model yolo --skip-frames 5

# Use OWLv2 (slower but more flexible)
python ai-processor.py /path/to/video.mp4 --model owlv2
```

#### Jetson Performance Recommendations

**Model Selection for Jetson Xavier:**
- **YOLOv8n/YOLOv8s**: Best for real-time processing
- **YOLOv8m**: Good balance of speed and accuracy
- **YOLOv8l/YOLOv8x**: Maximum accuracy (may be slower)

**Memory Management:**
- The script includes aggressive CUDA memory clearing
- Processes frames in batches to prevent out-of-memory errors
- Automatic fallback to CPU if GPU memory is exhausted

**Performance Tips:**
- Use smaller YOLO models (nano/small) for faster processing
- Use `--skip-frames 5` or higher to speed up processing significantly
- Process shorter video segments if memory is limited
- Monitor GPU memory usage with `nvidia-smi`

#### Troubleshooting Jetson Issues

**Common Issues:**

1. **CUDA not detected**:
   ```bash
   # Check CUDA installation
   nvidia-smi
   # Reinstall PyTorch with correct CUDA version
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Out of memory errors**:
   - Use smaller YOLO models (`yolov8n.pt`, `yolov8s.pt`)
   - Process shorter video segments
   - The script automatically handles memory management

3. **Slow performance**:
   - Ensure CUDA is properly detected
   - Use YOLO instead of OWLv2 for better performance
   - Check that the model is running on GPU, not CPU

4. **Polars installation error** (puccinialin not found):
   ```bash
   # Solution 1: Install polars without dependencies
   pip install polars --no-deps
   
   # Solution 2: Use pandas instead of polars
   pip install pandas>=1.5.0
   
   # Solution 3: Skip polars entirely (ultralytics will work without it)
   pip install ultralytics --no-deps
   pip install torch torchvision transformers pillow opencv-python-headless numpy scipy
   ```

**Verification Commands:**
```bash
# Check CUDA availability
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Test YOLO model loading
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO loaded successfully')"

# Test video processing (short test)
python ai-processor.py --model yolo --yolo-model yolov8n.pt /path/to/short_test_video.mp4
```

The Jetson Xavier provides excellent performance for AI object detection tasks, especially with YOLO models optimized for edge computing.

### Path Adjustments

After installation, you may need to adjust paths in the scripts for your local environment:

- **`convert_batch.py`**: Line 103 - Path to video footage folder
- **`ai-processor.py`**: Line 14 - Path to frames_folder

### Getting Updates

To update the project to the latest version:

```bash
git pull origin main
pip install -r requirements.txt  # If new dependencies were added
```

## Functional Workflow

The complete workflow of the detection system follows a structured 6-stage process:

### 1. Video Collection
- **Action**: User downloads video material (e.g., from cameras)
- **Format**: Typically MP4 files
- **Storage**: Local machine for initial review

### 2. Manual Video Review
- **Action**: User manually reviews downloaded videos
- **Decision**: 
  - ✅ **Animals present** → Keep video for further processing
  - ❌ **No animals** → Delete video (saves storage and processing time)
- **Purpose**: Efficient pre-selection to reduce data volume

### 3. Video Transfer to Fileshare
- **Action**: Transfer relevant videos (with animals) to central fileshare
- **Goal**: Central availability for further automated processing
- **Organization**: Structured storage by date/camera/etc.

### 4. Automatic Image Sequence Extraction
- **Tool**: `convert_batch.py`
- **Execution Location**: Fileshare server
- **Process**: 
  - Recursive scanning of all videos in fileshare
  - Automatic conversion to image sequences (`_frames` folders)
  - Avoidance of duplicate processing of already converted videos

### 5. AI-based Object Detection
- **Tool**: `ai-processor.py`
- **Execution Location**: Powerful machine with GPU support
- **Process**:
  - Processing of generated image sequences
  - Choice between OWLv2 or YOLO for object detection
  - Generation of detailed JSON detection reports in model-specific folders

#### Performance Recommendations:
- **Large datasets**: Use YOLO (`--model yolo`) for significantly faster processing
- **Very large datasets**: Combine YOLO with `--skip-frames 5` for 5x speed increase
- **Highest accuracy**: OWLv2 for more precise text-based animal detection
- **GPU usage**: Both models benefit from CUDA/MPS support
- **Model comparison**: Parallel processing with both models for quality comparison

### 6. Central Results Aggregation
- **Action**: All generated JSON files are collected centrally
- **Purpose**: 
  - Cross-analysis and statistics
  - Central data storage for further evaluations
  - Long-term monitoring and trends

## Technical Workflow

For technical implementation:

1. **Video Preparation**: Place selected MP4 files in fileshare directory
2. **Batch Conversion**: Execute `convert_batch.py` for automatic image sequence extraction
3. **AI Analysis**: Run object detection on powerful hardware
   ```bash
   # Fast processing with YOLO
   python ai-processor.py --model yolo /path/to/video_frames/
   
   # Very fast processing with skip-frames (5x speed)
   python ai-processor.py --model yolo --skip-frames 5 /path/to/video_frames/
   
   # Precise processing with OWLv2
   python ai-processor.py --model owlv2 /path/to/video_frames/
   
   # Both models for comparison
   python ai-processor.py --model owlv2 /path/to/video_frames/
   python ai-processor.py --model yolo /path/to/video_frames/
   ```
4. **Result Aggregation**: Collection and consolidation of all JSON output files from model-specific folders

The system is optimized for large video datasets and provides a fully automated pipeline from manually pre-selected video input to structured detection output. The YOLO integration enables significantly faster processing of large data volumes while maintaining high detection quality.

## Detection Annotation Script

### Overview

The `annotate_detections.py` script reads JSON detection results from `ai-processor.py` and draws bounding boxes on the corresponding frame images.

### Features

#### Main Functions:
- ✅ **JSON Parser**: Loads detection results from JSON files
- ✅ **Bounding Box Drawing**: Draws colored bounding boxes around detected objects
- ✅ **Label Display**: Shows labels and confidence scores for each detection
- ✅ **Batch Processing**: Processes all detected frames automatically
- ✅ **Progress Display**: Shows processing progress in real-time
- ✅ **Summary Image**: Automatically creates a grid of the best detections

#### Color Coding:
- 🟠 **Orange**: Cats (`cat`)
- 🟢 **Green**: Dogs (`dog`)
- 🔴 **Red**: People (`person`)
- 🔵 **Blue**: Birds (`bird`)
- 🟡 **Yellow**: Other/Unknown labels

### Usage

#### Basic Usage:
```bash
python annotate_detections.py detection_results/Camera_Teich-20250907-155316-1757253196613-7_frames_detection_results.json
```

#### With custom output directory:
```bash
python annotate_detections.py detection_results.json --output-dir my_annotations
```

#### With OpenCV rendering (alternative to PIL):
```bash
python annotate_detections.py detection_results.json --opencv
```

#### Without summary image:
```bash
python annotate_detections.py detection_results.json --no-summary
```

### Parameters

| Parameter | Description | Default |
|-----------|--------------|----------|
| `json_file` | Path to JSON file with detection results | (required) |
| `--output-dir`, `-o` | Output directory for annotated images | `annotated_<json_basename>` |
| `--opencv` | Use OpenCV instead of PIL for rendering | PIL |
| `--no-summary` | Don't create summary image | Create summary image |

### Output

The script creates:

1. **Annotated Individual Images**: Each frame with detections gets bounding boxes
   - Format: `annotated_frame_XXXXXX.jpg`
   - Contains all detections with labels and confidence scores

2. **Summary Image**: `detection_summary.jpg`
   - Grid layout of best detections (highest confidence)
   - Up to 9 images in 3x3 arrangement
   - Thumbnail size for quick overview

### Example Workflow

1. **Analyze video** (with ai-processor.py):
   ```bash
   python ai-processor.py my_video.mp4
   # Creates: detection_results/my_video_detection_results.json
   ```

2. **Annotate detections**:
   ```bash
   python annotate_detections.py detection_results/my_video_detection_results.json
   # Creates: annotated_my_video_detection_results/
   ```

3. **View results**:
   - Individual frames: `annotated_my_video_detection_results/annotated_frame_*.jpg`
   - Summary: `annotated_my_video_detection_results/detection_summary.jpg`

### Technical Details

#### Supported Formats:
- **Input**: JSON files from ai-processor.py
- **Output**: JPEG images with 95% quality
- **Frame Formats**: All formats supported by PIL/OpenCV

#### Performance:
- **PIL Mode**: Better quality, slightly slower
- **OpenCV Mode**: Faster, suitable for large quantities
- **Memory**: Efficient, loads only one image at a time

#### Error Handling:
- Missing frame images are skipped
- Corrupt JSON files are detected
- Detailed error messages for debugging

### Customization

The script can be easily adapted:

- **Change colors**: Edit the `get_label_color()` function
- **Text style**: Adjust `font_size`, `box_thickness` etc.
- **New labels**: Extend the `color_map` in `get_label_color()`
- **Output format**: Change `quality` parameter when saving

### Troubleshooting

#### Common Issues:

1. **"Frame image not found"**:
   - Frame paths in JSON don't match actual files
   - Frame images were moved or deleted

2. **"Error parsing JSON file"**:
   - JSON file is corrupted or incomplete
   - Use a JSON validator

3. **Font issues**:
   - Script automatically uses available system fonts
   - Install additional fonts if needed

#### Debug Mode:
Add `print()` statements to track processing progress.

### Result Example

After execution you get:
```
annotated_Camera_Teich-20250907-155316-1757253196613-7_frames_detection_results/
├── annotated_frame_000390.jpg    # Dog with green bounding box
├── annotated_frame_004710.jpg    # 3 dog detections
├── annotated_frame_004740.jpg    # 4 cat detections (orange)
├── ...
└── detection_summary.jpg         # Summary of best detections
```

Each image shows the original frames with:
- Colored bounding boxes around detected objects
- Label and confidence score above each box
- Optimized readability through background boxes
