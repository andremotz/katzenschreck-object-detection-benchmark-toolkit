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
        print(f"Überprüfe YOLO-Modell: {model_name}...")
        model = YOLO(model_name)
        print(f"YOLO-Modell {model_name} erfolgreich geladen/heruntergeladen")
        # Return the model object instead of just the name
        return model
    except Exception as e:
        print(f"Fehler beim Laden/Herunterladen des YOLO-Modells {model_name}: {e}")
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
                    # YOLO Klassen-ID zu Name konvertieren
                    class_id = int(box.cls.cpu().numpy())
                    class_name = model.names[class_id]
                    confidence = float(box.conf.cpu().numpy())
                    
                    # Only consider relevant classes
                    if class_name in target_classes and confidence > 0.1:
                        # Bounding Box Koordinaten (x1, y1, x2, y2)
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


def process_video_directly(video_path, processor, model, device, text_labels, model_type='yolo', model_identifier='yolo'):
    """
    Verarbeitet ein Video direkt ohne Frame-Zwischenspeicherung
    
    Args:
        video_path (str): Pfad zum Input-Video
        processor: The AI processor for object detection
        model: The AI model for object detection
        device: The compute device (CPU/GPU)
        text_labels: List of objects to detect
    
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
    
    # Video-Eigenschaften
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video-Informationen:")
    print(f"  Datei: {video_path}")
    print(f"  Frames gesamt: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Dauer: {duration:.2f} Sekunden")
    print(f"  Processing: Directly from video stream")
    print()
    
    # JSON data structure for results
    # Absoluten Pfad zum Video verwenden
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
    local_start_time = datetime.now()
    
    print(f"\nStarting processing at {local_start_time.strftime('%H:%M:%S')}...")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # OpenCV Frame (BGR) zu PIL Image (RGB) konvertieren
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Count frame processing
            processed_count += 1
            results_data["metadata"]["processed_frames"] = processed_count
            
            # Progress bar
            progress_percent = (processed_count / total_frames) * 100
            elapsed_time = datetime.now() - local_start_time
            
            # Calculate estimated remaining time
            if processed_count > 0:
                avg_time_per_frame = elapsed_time.total_seconds() / processed_count
                remaining_frames = total_frames - processed_count
                estimated_remaining = remaining_frames * avg_time_per_frame
                remaining_str = str(timedelta(seconds=int(estimated_remaining)))
            else:
                remaining_str = "unbekannt"
            
            print(f"Progress: {progress_percent:5.1f}% ({processed_count:3d}/{total_frames}) | Frame {processed_count-1:6d} | Restzeit: {remaining_str}", end="", flush=True)
            
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
                "frame_number": processed_count - 1,
                "frame_timestamp": (processed_count - 1) / fps if fps > 0 else 0,
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
                print(f" -> {len(detections)} Detektion(en)!")
            else:
                print(" -> keine Detektionen")
            
            # Memory-Management: Alle 10 Frames Memory leeren (aggressiver)
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
        input_path (str): Pfad zum Input
    
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
    # Argument Parser einrichten
    parser = argparse.ArgumentParser(
        description='AI-based object detection for videos or frame sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Beispiele:
  # Video verarbeiten (direkt aus Video-Stream):
  python ai-processor.py /pfad/zum/video.mp4
  python ai-processor.py "/Users/name/Downloads/video.mp4"
  
  # Frame-Ordner verarbeiten:
  python ai-processor.py /pfad/zum/frames/ordner
  python ai-processor.py "/Users/name/Downloads/Camera_Teich-frames"
  
Hinweis: Videos werden direkt aus dem Stream verarbeitet ohne Zwischenspeicherung.
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
        help='YOLO-Modell-Datei (default: yolo12x.pt). Kann auch yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt, yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt sein'
    )
    
    args = parser.parse_args()
    input_path = args.input_path
    model_type = args.model
    yolo_model_name = args.yolo_model
    
    # Create specific model identifier for filenames and metadata
    if model_type == 'yolo':
        # Extract model name without .pt extension for cleaner naming
        model_identifier = yolo_model_name.replace('.pt', '')
    else:
        model_identifier = 'owlv2'
    
    # Validierung des Input-Pfades
    if not os.path.exists(input_path):
        print(f"Error: The specified path '{input_path}' does not exist.")
        sys.exit(EXIT_INPUT_NOT_FOUND)
    
    # Input-Typ erkennen
    input_type = detect_input_type(input_path)
    
    if input_type == 'unknown':
        print(f"Error: Unknown input type. Supported formats:")
        print("  - Videos: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v")
        print("  - Frame-Ordner: Ordner mit frame_*.jpg Dateien")
        sys.exit(EXIT_UNKNOWN_INPUT_TYPE)
    
    # Load YOLO model only
    processor = None
    model = None
    
    if not YOLO_AVAILABLE:
        print("Error: YOLO is required but ultralytics is not installed.")
        print("Installiere mit: pip install ultralytics")
        sys.exit(EXIT_MODEL_LOADING_FAILED)
    
    print(f"Lade YOLO-Modell: {yolo_model_name}...")
    try:
        # Ensure model is downloaded and load it
        model = ensure_yolo_model_downloaded(yolo_model_name)
        print(f"YOLO-Modell erfolgreich geladen")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(EXIT_MODEL_LOADING_FAILED)
    
    # Check and activate GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Verwendetes Device: {device}")
    
    # YOLO models handle device assignment automatically
    
    # Memory bereinigen vor dem Start
    clear_memory(device)
    print(f"Memory cache cleared for optimal performance")
    
    text_labels = [["cat"]]
    
    # Start time for entire processing
    start_time = datetime.now()
    
    # Je nach Input-Typ unterschiedlich verarbeiten
    if input_type == 'video':
        print(f"Erkannter Input-Typ: Video")
        print(f"Verarbeite Video direkt ohne Frame-Zwischenspeicherung...")
        
        success, results_data = process_video_directly(input_path, processor, model, device, text_labels, model_type, model_identifier)
        if not success:
            print("Error in video processing")
            sys.exit(EXIT_VIDEO_PROCESSING_FAILED)
    else:
        print(f"Erkannter Input-Typ: Frame-Sequenz")
        frames_folder = input_path
        
        print(f"Modell geladen. Verarbeite Frames aus: {frames_folder}")

        # Find and sort all frame images
        frame_pattern = os.path.join(frames_folder, "frame_*.jpg")
        frame_files = sorted(glob.glob(frame_pattern))

        if not frame_files:
            print(f"Error: No frame images found in folder {frames_folder}")
            sys.exit(EXIT_NO_FRAMES_FOUND)

        print(f"Found: {len(frame_files)} frame images")

        # Alle Frames werden verarbeitet
        frames_to_process = len(frame_files)
        print(f"Frames to be processed: {frames_to_process}")

        # JSON data structure for results
        # Absolute Pfade verwenden
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

        # Jeden Frame verarbeiten
        for i, frame_path in enumerate(frame_files):
            try:
                # Bild direkt laden
                image = Image.open(frame_path)
                
                # Frame-Nummer aus Dateiname extrahieren
                frame_filename = os.path.basename(frame_path)
                frame_number = int(frame_filename.split('_')[1].split('.')[0])
                
                # Count frame processing
                processed_count += 1
                results_data["metadata"]["processed_frames"] = processed_count
                
                # Fortschrittsanzeige
                progress_percent = (processed_count / frames_to_process) * 100
                elapsed_time = datetime.now() - local_start_time
                
                # Calculate estimated remaining time
                if processed_count > 0:
                    avg_time_per_frame = elapsed_time.total_seconds() / processed_count
                    remaining_frames = frames_to_process - processed_count
                    estimated_remaining = remaining_frames * avg_time_per_frame
                    remaining_str = str(timedelta(seconds=int(estimated_remaining)))
                else:
                    remaining_str = "unbekannt"
                
                print(f"Progress: {progress_percent:5.1f}% ({processed_count:3d}/{frames_to_process}) | Frame {frame_number:6d} | Restzeit: {remaining_str}", end="", flush=True)
                
                # AI processing with YOLO only
                target_classes = text_labels[0] if isinstance(text_labels[0], list) else text_labels
                success, detections, used_device = process_frame_with_fallback_yolo(
                    model, device, image, target_classes
                )
                
                if not success:
                    print(f" -> Frame processing failed, skipping...")
                    continue
                
                # Add frame data to JSON
                # Absoluten Pfad zum Frame verwenden
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
                    print(f" -> {len(detections)} Detektion(en)!")
                else:
                    print(" -> keine Detektionen")
                
                # Memory-Management: Alle 10 Frames Memory leeren (aggressiver)
                if processed_count % 10 == 0:
                    clear_memory(device)
                    
                # YOLO models handle memory management automatically
            
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                continue

    # JSON-Datei schreiben - Name basierend auf Input mit Modell-Suffix
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
    
    # JSON-Datei im detection_results Verzeichnis speichern
    detection_results_dir = os.path.join(parent_dir, "detection_results")
    os.makedirs(detection_results_dir, exist_ok=True)
    output_path = os.path.join(detection_results_dir, output_filename)

    # Endstatistiken berechnen
    end_time = datetime.now()
    total_processing_time = end_time - start_time
    detection_rate = (results_data['metadata']['frames_with_detections'] / results_data['metadata']['processed_frames']) * 100 if results_data['metadata']['processed_frames'] > 0 else 0

    print("\n" + "="*60)
    print("VERARBEITUNG ABGESCHLOSSEN")
    print("="*60)
    print(f"Startzeit:           {start_time.strftime('%H:%M:%S')}")
    print(f"Endzeit:             {end_time.strftime('%H:%M:%S')}")
    print(f"Gesamtdauer:         {total_processing_time}")
    print(f"Durchschnitt/Frame:  {total_processing_time.total_seconds()/results_data['metadata']['processed_frames']:.2f}s")
    print("-"*60)
    # Anzahl Frames je nach Input-Typ ermitteln
    if 'total_frames' in results_data['metadata']:
        total_frames_count = results_data['metadata']['total_frames']
    elif 'video_properties' in results_data['metadata']:
        total_frames_count = results_data['metadata']['video_properties']['total_frames']
    else:
        total_frames_count = results_data['metadata']['processed_frames']
    
    print(f"Found frames:        {total_frames_count}")
    print(f"Verarbeitete Frames: {results_data['metadata']['processed_frames']}")
    print(f"Frames m. Detektion: {results_data['metadata']['frames_with_detections']}")
    print(f"Detektionsrate:      {detection_rate:.1f}%")

    # Count detection details
    total_detections = sum(len(frame['detections']) for frame in results_data['detections'])
    cat_detections = sum(len([d for d in frame['detections'] if 'cat' in d['label']]) for frame in results_data['detections'])

    print(f"Gesamt Detektionen:  {total_detections}")
    print(f"  - Katzen:          {cat_detections}")
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