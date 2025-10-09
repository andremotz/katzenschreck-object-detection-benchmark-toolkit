#!/usr/bin/env python3
"""
Process all files from a folder with the AI-Processor (parallel with 4 threads)
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lock for thread-safe output
print_lock = threading.Lock()

def process_single_file(file_path, ai_processor_path, model, yolo_model, skip_frames):
    """
    Process a single file with the AI-Processor
    
    Args:
        file_path: Path to the file to be processed
        ai_processor_path: Path to the ai-processor.py script
        model: Model type
        yolo_model: Specific YOLO model
        skip_frames: Number of frames to skip
    
    Returns:
        tuple: (file_path, success, error_message)
    """
    with print_lock:
        print(f"🔄 Starting: {file_path.name}")
    
    # Build the command
    cmd = [
        str(ai_processor_path),
        str(file_path),
        "--model", model,
        "--yolo-model", yolo_model,
        "--skip-frames", str(skip_frames)
    ]
    
    try:
        # Execute the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        with print_lock:
            print(f"✓ Success: {file_path.name}")
        return (file_path, True, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}"
        with print_lock:
            print(f"✗ Error: {file_path.name} - {error_msg}")
        return (file_path, False, error_msg)
    except Exception as e:
        with print_lock:
            print(f"✗ Exception: {file_path.name} - {str(e)}")
        return (file_path, False, str(e))

def process_folder(folder_path, model="yolo", yolo_model="yolo11x", skip_frames=12, max_workers=4):
    """
    Process all files in a folder with the AI-Processor (parallel)
    
    Args:
        folder_path: Path to the folder containing the files to process
        model: Model type (default: "yolo")
        yolo_model: Specific YOLO model (default: "yolo11x")
        skip_frames: Number of frames to skip (default: 12)
        max_workers: Maximum number of parallel threads (default: 4)
    """
    # Path to ai-processor.py (in the same directory)
    script_dir = Path(__file__).resolve().parent
    ai_processor_path = script_dir / "ai-processor.py"
    
    # Check if the folder exists
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        sys.exit(1)
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a folder!")
        sys.exit(1)
    
    # Check if ai-processor.py exists
    if not ai_processor_path.exists():
        print(f"Error: '{ai_processor_path}' does not exist!")
        sys.exit(1)
    
    # Collect all files (no subdirectories)
    files = [f for f in folder.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in '{folder_path}'!")
        return
    
    files_sorted = sorted(files)
    total_files = len(files_sorted)
    
    print(f"Found: {total_files} file(s) to process")
    print(f"Parallel processing with {max_workers} threads")
    print("=" * 60)
    
    # Statistics
    successful = 0
    failed = 0
    
    try:
        # Process files in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Start all tasks
            futures = {
                executor.submit(
                    process_single_file,
                    file_path,
                    ai_processor_path,
                    model,
                    yolo_model,
                    skip_frames
                ): file_path
                for file_path in files_sorted
            }
            
            # Wait for results
            for future in as_completed(futures):
                file_path, success, error = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Show progress
                processed = successful + failed
                with print_lock:
                    print(f"[{processed}/{total_files}] Progress: {successful} successful, {failed} failed")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user!")
        sys.exit(1)
    
    return successful, failed

if __name__ == "__main__":
    # Folder path is a required parameter
    if len(sys.argv) < 2:
        print("Error: No folder path specified!")
        print()
        print("Usage:")
        print(f"  {sys.argv[0]} <folder_path>")
        print()
        print("Example:")
        print(f"  {sys.argv[0]} /Users/andremotz/Downloads/katzenschreck_cam/20250926PM")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    print(f"Processing folder: {folder_path}")
    print()
    
    result = process_folder(folder_path)
    
    if result:
        successful, failed = result
        print("\n" + "=" * 60)
        print("Processing completed!")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"📊 Total: {successful + failed}")
        print("=" * 60)
