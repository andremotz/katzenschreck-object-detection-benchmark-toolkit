#!/usr/bin/env python3
"""
AI Object Detection Script for Videos and Frame Sequences

This script uses modern AI models (OWLv2 or YOLO) for automatic object detection 
in videos or image sequences. It was specifically developed for cat detection 
but can also recognize other objects.

Main Features:
- Direct video processing without intermediate storage
- Frame sequence processing (frame_*.jpg)
- Support for OWLv2 (Google) and YOLO models
- Automatic GPU/CPU detection and optimization
- Robust error handling and memory management
- Detailed progress display with time estimation
- JSON export of detection results

Supported Input Formats:
- Videos: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v
- Frame folders: Folders with frame_*.jpg files

Usage:
    python ai-processor.py /path/to/video.mp4
    python ai-processor.py /path/to/frames/folder --model yolo --yolo-model yolo12x.pt
    python ai-processor.py /path/to/video.mp4 --skip-frames 5  # 5x faster processing

Author: AI Assistant
Version: 1.0
"""

import os
import glob
import json
import argparse
import sys
import gc
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import torch
import cv2
import numpy as np

# OWLv2 imports removed for Jetson compatibility
# from transformers import Owlv2Processor, Owlv2ForObjectDetection

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO models not available.")

# Exit codes for different error cases
EXIT_SUCCESS = 0
EXIT_INPUT_NOT_FOUND = 1          # Input path does not exist
EXIT_UNKNOWN_INPUT_TYPE = 2       # Unknown input type
EXIT_VIDEO_NOT_FOUND = 3          # Video file not found
EXIT_VIDEO_OPEN_FAILED = 4        # Video could not be opened
EXIT_NO_FRAMES_FOUND = 5          # No frame images found
EXIT_VIDEO_PROCESSING_FAILED = 6  # General video processing error
EXIT_MODEL_LOADING_FAILED = 7     # AI model could not be loaded
EXIT_JSON_WRITE_FAILED = 8        # JSON file could not be written
EXIT_CRITICAL_ERROR = 9           # Critical unknown error


def resize_to_fullhd(image):
    """
    Resizes an image to Full HD (1920x1080) while maintaining aspect ratio
    
    Args:
        image: PIL Image object
    
    Returns:
        PIL Image: Resized image to Full HD
    """
    # Full HD dimensions
    target_width = 1920
    target_height = 1080
    
    # Get current dimensions
    current_width, current_height = image.size
    
    # Calculate scaling factor to fit within Full HD while maintaining aspect ratio
    scale_w = target_width / current_width
    scale_h = target_height / current_height
    scale = min(scale_w, scale_h)  # Use the smaller scale to ensure it fits
    
    # Calculate new dimensions
    new_width = int(current_width * scale)
    new_height = int(current_height * scale)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # If the image is smaller than Full HD, create a black canvas and center the image
    if new_width < target_width or new_height < target_height:
        # Create a black canvas with Full HD dimensions
        canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste the resized image onto the canvas
        canvas.paste(resized_image, (x_offset, y_offset))
        return canvas
    
    return resized_image


def clear_memory(device):
    """
    Clears GPU/MPS memory to avoid out-of-memory errors
    
    Args:
        device: The compute device being used
    """
    # Garbage collection for Python objects
    gc.collect()
    
    # Clear PyTorch memory cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()


def ensure_yolo_model_downloaded(model_name):
    """
    Ensures that the specified YOLO model is downloaded and available
    
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolo8x.pt')
    
    Returns:
        str: Path to the downloaded model file
    """
    try:
        # Try to load the model - this will automatically download if not present
        print(f"Checking YOLO model: {model_name}...")
        model = YOLO(model_name)
        print(f"YOLO model {model_name} successfully loaded/downloaded")
        # Return the model object instead of just the name
        return model
    except Exception as e:
        print(f"Error loading/downloading YOLO model {model_name}: {e}")
        raise e


# OWLv2 processing function removed for Jetson compatibility
# def process_frame_with_fallback_owlv2(processor, model, device, image, text_labels):


def process_frame_with_fallback_yolo(model, device, image, target_classes):
    """
    Attempts frame processing with YOLO and robust error handling
    
    Args:
        model: The YOLO model
        device: Primary device (GPU/MPS/CPU)
        image: The image to process (PIL Image)
        target_classes: List of classes to detect (e.g. ['cat', 'dog'])
    
    Returns:
        tuple: (success, detections, used_device) 
               detections: List of dicts with 'label', 'confidence', 'bounding_box'
    """
    try:
        # PIL Image to numpy array for YOLO
        img_array = np.array(image)
        
        # YOLO Inference
        results = model(img_array, device=device.type if hasattr(device, 'type') else str(device))
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Convert YOLO class ID to name
                    class_id = int(box.cls.cpu().numpy())
                    class_name = model.names[class_id]
                    confidence = float(box.conf.cpu().numpy())
                    
                    # Only consider relevant classes
                    if class_name in target_classes and confidence > 0.1:
                        # Bounding box coordinates (x1, y1, x2, y2)
                        coords = box.xyxy.cpu().numpy()[0]
                        
                        detection = {
                            'label': class_name,
                            'confidence': round(confidence, 3),
                            'bounding_box': {
                                'xmin': round(float(coords[0]), 2),
                                'ymin': round(float(coords[1]), 2),
                                'xmax': round(float(coords[2]), 2),
                                'ymax': round(float(coords[3]), 2)
                            }
                        }
                        detections.append(detection)
        
        return True, detections, device
        
    except Exception as e:
        print(f" -> YOLO error, skipping frame: {str(e)[:100]}...")
        return False, [], None


def process_video_directly(video_path, processor, model, device, text_labels, model_type='yolo', model_identifier='yolo', skip_frames=1):
    """
    Processes a video directly without intermediate frame storage
    
    Args:
        video_path (str): Path to the input video
        processor: The AI processor for object detection
        model: The AI model for object detection
        device: The compute device (CPU/GPU)
        text_labels: List of objects to detect
        skip_frames (int): Process only every N-th frame (default: 1)
    
    Returns:
        tuple: (success, results_data)
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(EXIT_VIDEO_NOT_FOUND)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(EXIT_VIDEO_OPEN_FAILED)
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video Information:")
    print(f"  File: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Processing: Directly from video stream")
    if skip_frames > 1:
        print(f"  Skip-Frames: Processing only every {skip_frames}. frame ({total_frames // skip_frames} frames)")
    print()
    
    # JSON data structure for results
    # Use absolute path to video
    absolute_video_path = os.path.abspath(video_path)
    
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_type": "video",
            "input_path": absolute_video_path,
            "video_properties": {
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": duration
            },
            "frames_to_process": total_frames,
            "skip_frames": skip_frames,
            "processed_frames": 0,
            "frames_with_detections": 0,
            "detection_threshold": 0.1,
            "text_labels": text_labels[0],
            "model_type": model_identifier,
            "processing_method": "direct_video_stream"
        },
        "detections": []
    }
    
    # Counter for progress display  
    processed_count = 0
    frame_counter = 0  # Counts all read frames
    local_start_time = datetime.now()
    
    print(f"\nStarting processing at {local_start_time.strftime('%H:%M:%S')}...")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Increment frame counter
            frame_counter += 1
            
            # Skip frames based on skip_frames parameter
            if (frame_counter - 1) % skip_frames != 0:
                continue
            
            # Convert OpenCV frame (BGR) to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Scale frame to Full HD
            # image = resize_to_fullhd(image)
            
            # Count frame processing
            processed_count += 1
            results_data["metadata"]["processed_frames"] = processed_count
            
            # Progress bar
            expected_total = total_frames // skip_frames
            progress_percent = (processed_count / expected_total) * 100 if expected_total > 0 else 0
            elapsed_time = datetime.now() - local_start_time
            
            # Calculate estimated remaining time
            if processed_count > 0:
                avg_time_per_frame = elapsed_time.total_seconds() / processed_count
                remaining_frames = expected_total - processed_count
                estimated_remaining = remaining_frames * avg_time_per_frame
                remaining_str = str(timedelta(seconds=int(estimated_remaining)))
            else:
                remaining_str = "unknown"
            
            print(f"Progress: {progress_percent:5.1f}% ({processed_count:3d}/{expected_total}) | Frame {frame_counter-1:6d} | Remaining: {remaining_str}", end="", flush=True)
            
            # AI processing with YOLO only
            target_classes = text_labels[0] if isinstance(text_labels[0], list) else text_labels
            success, detections, used_device = process_frame_with_fallback_yolo(
                model, device, image, target_classes
            )
            
            if not success:
                print(f" -> Frame processing failed, skipping...")
                continue
            
            # Add frame data to JSON
            frame_data = {
                "frame_number": frame_counter - 1,
                "frame_timestamp": (frame_counter - 1) / fps if fps > 0 else 0,
                "image_size": {
                    "width": image.width,
                    "height": image.height
                },
                "detections": detections
            }
            
            # Only add to results if detections were found
            if len(detections) > 0:
                results_data["detections"].append(frame_data)
                results_data["metadata"]["frames_with_detections"] += 1
                print(f" -> {len(detections)} detection(s)!")
            else:
                print(" -> no detections")
            
            # Memory management: Clear memory every 10 frames (aggressive)
            if processed_count % 10 == 0:
                clear_memory(device)
                
            # YOLO models handle memory management automatically
    
    except KeyboardInterrupt:
        print("\nProcessing cancelled by user")
    except Exception as e:
        print(f"\nError in video processing: {e}")
        cap.release()
        sys.exit(EXIT_VIDEO_PROCESSING_FAILED)
    finally:
        cap.release()
    
    return True, results_data


def detect_input_type(input_path):
    """
    Automatically detects if input is a video or a folder with frames
    
    Args:
        input_path (str): Path to the input
    
    Returns:
        str: 'video', 'frames', or 'unknown'
    """
    path = Path(input_path)
    
    if path.is_file():
        # Check video file extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        if path.suffix.lower() in video_extensions:
            return 'video'
    elif path.is_dir():
        # Check if folder contains frame images
        frame_pattern = os.path.join(input_path, "frame_*.jpg")
        frame_files = glob.glob(frame_pattern)
        if frame_files:
            return 'frames'
    
    return 'unknown'


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='AI-based object detection for videos or frame sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process video (directly from video stream):
  python ai-processor.py /path/to/video.mp4
  python ai-processor.py "/Users/name/Downloads/video.mp4"
  
  # Process frame folder:
  python ai-processor.py /path/to/frames/folder
  python ai-processor.py "/Users/name/Downloads/Camera_Teich-frames"
  
  # Process video with skip-frames (5x faster):
  python ai-processor.py /path/to/video.mp4 --skip-frames 5
  
Note: Videos are processed directly from the stream without intermediate storage.
        '''
    )
    parser.add_argument(
        'input_path',
        help='Path to video (mp4, avi, mov, etc.) or folder with frame images (frame_*.jpg)'
    )
    parser.add_argument(
        '--model',
        choices=['yolo'],
        default='yolo',
        help='AI model to use: yolo (only option for Jetson compatibility)'
    )
    parser.add_argument(
        '--yolo-model',
        default='yolo12x.pt',
        help='YOLO model file (default: yolo12x.pt). Can also be yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt, yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=1,
        help='Process only every N-th frame (default: 1 = all frames). For 5x speed: --skip-frames 5'
    )
    
    args = parser.parse_args()
    input_path = args.input_path
    model_type = args.model
    yolo_model_name = args.yolo_model
    skip_frames = args.skip_frames
    
    # Create specific model identifier for filenames and metadata
    if model_type == 'yolo':
        # Extract model name without .pt extension for cleaner naming
        model_identifier = yolo_model_name.replace('.pt', '')
    else:
        model_identifier = 'owlv2'
    
    # Validate input path
    if not os.path.exists(input_path):
        print(f"Error: The specified path '{input_path}' does not exist.")
        sys.exit(EXIT_INPUT_NOT_FOUND)
    
    # Detect input type
    input_type = detect_input_type(input_path)
    
    if input_type == 'unknown':
        print(f"Error: Unknown input type. Supported formats:")
        print("  - Videos: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v")
        print("  - Frame folders: Folders with frame_*.jpg files")
        sys.exit(EXIT_UNKNOWN_INPUT_TYPE)
    
    # Load YOLO model only
    processor = None
    model = None
    
    if not YOLO_AVAILABLE:
        print("Error: YOLO is required but ultralytics is not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(EXIT_MODEL_LOADING_FAILED)
    
    print(f"Loading YOLO model: {yolo_model_name}...")
    try:
        # Ensure model is downloaded and load it
        model = ensure_yolo_model_downloaded(yolo_model_name)
        print(f"YOLO model successfully loaded")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(EXIT_MODEL_LOADING_FAILED)
    
    # Check and activate GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # YOLO models handle device assignment automatically
    
    # Clear memory before start
    clear_memory(device)
    print(f"Memory cache cleared for optimal performance")
    
    text_labels = [["cat"]]
    
    # Start time for entire processing
    start_time = datetime.now()
    
    # Process based on input type
    if input_type == 'video':
        print(f"Detected input type: Video")
        print(f"Processing video directly without intermediate frame storage...")
        
        success, results_data = process_video_directly(input_path, processor, model, device, text_labels, model_type, model_identifier, skip_frames)
        if not success:
            print("Error in video processing")
            sys.exit(EXIT_VIDEO_PROCESSING_FAILED)
    else:
        print(f"Detected input type: Frame sequence")
        frames_folder = input_path
        
        print(f"Model loaded. Processing frames from: {frames_folder}")

        # Find and sort all frame images
        frame_pattern = os.path.join(frames_folder, "frame_*.jpg")
        frame_files = sorted(glob.glob(frame_pattern))

        if not frame_files:
            print(f"Error: No frame images found in folder {frames_folder}")
            sys.exit(EXIT_NO_FRAMES_FOUND)

        print(f"Found: {len(frame_files)} frame images")

        # Filter frames based on skip_frames parameter
        if skip_frames > 1:
            frame_files = frame_files[::skip_frames]
            print(f"Skip-Frames: Processing only every {skip_frames}. frame")
        
        frames_to_process = len(frame_files)
        print(f"Frames to be processed: {frames_to_process}")

        # JSON data structure for results
        # Use absolute paths
        absolute_input_path = os.path.abspath(input_path)
        absolute_frames_folder = os.path.abspath(frames_folder)
        
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "input_type": input_type,
                "input_path": absolute_input_path,
                "frames_folder": absolute_frames_folder,
                "total_frames": len(frame_files),
                "frames_to_process": frames_to_process,
                "skip_frames": skip_frames,
                "processed_frames": 0,
                "frames_with_detections": 0,
                "detection_threshold": 0.1,
                "text_labels": text_labels[0],
                "model_type": model_identifier,
                "processing_method": "frame_sequence"
            },
            "detections": []
        }

        # Counter for progress display
        processed_count = 0
        local_start_time = datetime.now()

        print(f"\nStarting processing at {local_start_time.strftime('%H:%M:%S')}...")
        print("="*60)

        # Process each frame
        for i, frame_path in enumerate(frame_files):
            try:
                # Load image directly
                image = Image.open(frame_path)
                
                # Scale frame to Full HD
                image = resize_to_fullhd(image)
                
                # Extract frame number from filename
                frame_filename = os.path.basename(frame_path)
                frame_number = int(frame_filename.split('_')[1].split('.')[0])
                
                # Count frame processing
                processed_count += 1
                results_data["metadata"]["processed_frames"] = processed_count
                
                # Progress display
                progress_percent = (processed_count / frames_to_process) * 100
                elapsed_time = datetime.now() - local_start_time
                
                # Calculate estimated remaining time
                if processed_count > 0:
                    avg_time_per_frame = elapsed_time.total_seconds() / processed_count
                    remaining_frames = frames_to_process - processed_count
                    estimated_remaining = remaining_frames * avg_time_per_frame
                    remaining_str = str(timedelta(seconds=int(estimated_remaining)))
                else:
                    remaining_str = "unknown"
                
                print(f"Progress: {progress_percent:5.1f}% ({processed_count:3d}/{frames_to_process}) | Frame {frame_number:6d} | Remaining: {remaining_str}", end="", flush=True)
                
                # AI processing with YOLO only
                target_classes = text_labels[0] if isinstance(text_labels[0], list) else text_labels
                success, detections, used_device = process_frame_with_fallback_yolo(
                    model, device, image, target_classes
                )
                
                if not success:
                    print(f" -> Frame processing failed, skipping...")
                    continue
                
                # Add frame data to JSON
                # Use absolute path to frame
                absolute_frame_path = os.path.abspath(frame_path)
                
                frame_data = {
                    "frame_number": frame_number,
                    "frame_filename": frame_filename,
                    "frame_path": absolute_frame_path,
                    "image_size": {
                        "width": image.width,
                        "height": image.height
                    },
                    "detections": detections
                }
                
                # Only add to results if detections were found
                if len(detections) > 0:
                    results_data["detections"].append(frame_data)
                    results_data["metadata"]["frames_with_detections"] += 1
                    print(f" -> {len(detections)} detection(s)!")
                else:
                    print(" -> no detections")
                
                # Memory management: Clear memory every 10 frames (aggressive)
                if processed_count % 10 == 0:
                    clear_memory(device)
                    
                # YOLO models handle memory management automatically
            
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                continue

    # Write JSON file - Name based on input with model suffix
    if input_type == 'video':
        # For videos: Name based on video filename with model suffix
        video_name = Path(input_path).stem
        output_filename = f"{video_name}_detection_results_{model_identifier}.json"
        parent_dir = Path(input_path).parent
    else:
        # For frame folders: Name based on folder name with model suffix
        folder_name = os.path.basename(input_path)
        output_filename = f"{folder_name}_detection_results_{model_identifier}.json"
        parent_dir = os.path.dirname(input_path)
    
    # Save JSON file in detection_results directory
    detection_results_dir = os.path.join(parent_dir, "detection_results")
    os.makedirs(detection_results_dir, exist_ok=True)
    output_path = os.path.join(detection_results_dir, output_filename)

    # Calculate final statistics
    end_time = datetime.now()
    total_processing_time = end_time - start_time
    detection_rate = (results_data['metadata']['frames_with_detections'] / results_data['metadata']['processed_frames']) * 100 if results_data['metadata']['processed_frames'] > 0 else 0

    print("\n" + "="*60)
    print("PROCESSING COMPLETED")
    print("="*60)
    print(f"Start time:          {start_time.strftime('%H:%M:%S')}")
    print(f"End time:            {end_time.strftime('%H:%M:%S')}")
    print(f"Total duration:      {total_processing_time}")
    print(f"Average/frame:       {total_processing_time.total_seconds()/results_data['metadata']['processed_frames']:.2f}s")
    print("-"*60)
    # Determine frame count based on input type
    if 'total_frames' in results_data['metadata']:
        total_frames_count = results_data['metadata']['total_frames']
    elif 'video_properties' in results_data['metadata']:
        total_frames_count = results_data['metadata']['video_properties']['total_frames']
    else:
        total_frames_count = results_data['metadata']['processed_frames']
    
    print(f"Found frames:        {total_frames_count}")
    print(f"Processed frames:    {results_data['metadata']['processed_frames']}")
    print(f"Frames w. detection: {results_data['metadata']['frames_with_detections']}")
    print(f"Detection rate:      {detection_rate:.1f}%")

    # Count detection details
    total_detections = sum(len(frame['detections']) for frame in results_data['detections'])
    cat_detections = sum(len([d for d in frame['detections'] if 'cat' in d['label']]) for frame in results_data['detections'])

    print(f"Total detections:    {total_detections}")
    print(f"  - Cats:            {cat_detections}")
    print("="*60)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        # Fallback: Output results to console
        print("\nFallback - Results in console:")
        print(json.dumps(results_data, indent=2, ensure_ascii=False))
        sys.exit(EXIT_JSON_WRITE_FAILED)
    


if __name__ == "__main__":
    try:
        main()
        sys.exit(EXIT_SUCCESS)
    except KeyboardInterrupt:
        print("\nProcessing cancelled by user")
        sys.exit(EXIT_SUCCESS)  # User cancellation is not an error
    except Exception as e:
        print(f"\nCritical error: {e}")
        sys.exit(EXIT_CRITICAL_ERROR)