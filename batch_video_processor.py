#!/usr/bin/env python3
"""
Batch Video Processor - Recursive Video Processing

This script recursively searches a directory for video files and processes
them automatically through the complete pipeline:
1. Video → Frame Sequence (convert_video_to_image_sequence.py)
2. Frame Sequence → AI Detection (ai-processor.py)

Supported Video Formats:
- .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v

Usage:
    python batch_video_processor.py /path/to/video/directory
    python batch_video_processor.py /path/to/video/directory --model yolo --yolo-model yolo12x.pt
    python batch_video_processor.py /path/to/video/directory --skip-frames 5
    python batch_video_processor.py /path/to/video/directory --skip-conversion --skip-detection
    python batch_video_processor.py /path/to/video/directory --skip-conversion --model yolo --yolo-model yolo12x.pt

Author: Andre Motz
Version: 1.0
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json

# Exit codes
EXIT_SUCCESS = 0
EXIT_INPUT_NOT_FOUND = 1
EXIT_NO_VIDEOS_FOUND = 2
EXIT_CONVERSION_FAILED = 3
EXIT_DETECTION_FAILED = 4
EXIT_CRITICAL_ERROR = 5

# Supported Video Formats
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}


def find_video_files(root_dir, recursive=True):
    """
    Finds all video files in a directory
    
    Args:
        root_dir (str): Root directory for search
        recursive (bool): Search recursively in subdirectories
    
    Returns:
        list: List of video file paths
    """
    video_files = []
    
    if recursive:
        # Search recursively in all subdirectories
        for ext in VIDEO_EXTENSIONS:
            pattern = os.path.join(root_dir, "**", f"*{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))
    else:
        # Search only in the specified directory
        for ext in VIDEO_EXTENSIONS:
            pattern = os.path.join(root_dir, f"*{ext}")
            video_files.extend(glob.glob(pattern))
    
    # Sort for consistent processing
    return sorted(video_files)


def run_convert_video_to_frames(video_path, quality=75):
    """
    Runs convert_video_to_image_sequence.py for a video
    
    Args:
        video_path (str): Path to video
        quality (int): JPEG quality
    
    Returns:
        tuple: (success, frame_count, output_dir)
    """
    print(f"\n{'='*60}")
    print(f"CONVERSION: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    try:
        # Call convert_video_to_image_sequence.py
        cmd = [
            sys.executable, 
            "convert_video_to_image_sequence.py", 
            video_path,
            None,  # output_dir will be created automatically
            str(quality)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            # Successful conversion - determine output directory
            video_name = Path(video_path).stem
            video_dir = Path(video_path).parent
            output_dir = video_dir / f"{video_name}_frames"
            
            # Check if output directory exists and contains images
            if output_dir.exists():
                frame_files = list(output_dir.glob("frame_*.jpg"))
                frame_count = len(frame_files)
                print(f"✅ Conversion successful: {frame_count} frames in {output_dir}")
                return True, frame_count, str(output_dir)
            else:
                print(f"❌ Conversion failed: Output directory not found")
                return False, 0, None
        else:
            print(f"❌ Conversion failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False, 0, None
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False, 0, None


def run_ai_processor(frames_dir, model_type='owlv2', yolo_model='yolo12x.pt', skip_frames=1):
    """
    Runs ai-processor.py for a frame directory
    
    Args:
        frames_dir (str): Path to frame directory
        model_type (str): AI model type ('owlv2' or 'yolo')
        yolo_model (str): YOLO model name
        skip_frames (int): Process only every N-th frame (default: 1 = all frames)
    
    Returns:
        tuple: (success, results_file)
    """
    print(f"\n{'='*60}")
    print(f"AI-DETECTION: {os.path.basename(frames_dir)}")
    print(f"{'='*60}")
    
    try:
        # Call ai-processor.py
        cmd = [
            sys.executable,
            "ai-processor.py",
            frames_dir,
            "--model", model_type
        ]
        
        if model_type == 'yolo':
            cmd.extend(["--yolo-model", yolo_model])
        
        if skip_frames > 1:
            cmd.extend(["--skip-frames", str(skip_frames)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            # Successful processing - find JSON file
            frames_name = os.path.basename(frames_dir)
            parent_dir = os.path.dirname(frames_dir)
            detection_results_dir = os.path.join(parent_dir, "detection_results")
            
            # Search for JSON file
            json_pattern = os.path.join(detection_results_dir, f"{frames_name}_detection_results_*.json")
            json_files = glob.glob(json_pattern)
            
            if json_files:
                results_file = json_files[0]  # First found JSON file
                print(f"✅ AI detection successful: {results_file}")
                return True, results_file
            else:
                print(f"❌ AI detection failed: JSON file not found")
                return False, None
        else:
            print(f"❌ AI detection failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error during AI detection: {e}")
        return False, None


def process_single_video(video_path, model_type='owlv2', yolo_model='yolo12x.pt', 
                        quality=75, skip_conversion=False, skip_detection=False, skip_frames=1):
    """
    Processes a single video through the complete pipeline
    
    Args:
        video_path (str): Path to video
        model_type (str): AI model type
        yolo_model (str): YOLO model name
        quality (int): JPEG quality
        skip_conversion (bool): Skip conversion
        skip_detection (bool): Skip AI detection
        skip_frames (int): Process only every N-th frame (default: 1 = all frames)
    
    Returns:
        dict: Processing statistics
    """
    stats = {
        'video_path': video_path,
        'video_name': os.path.basename(video_path),
        'conversion_success': False,
        'detection_success': False,
        'frame_count': 0,
        'frames_dir': None,
        'results_file': None,
        'error': None
    }
    
    try:
        # Step 1: Convert video to frames (if not skipped)
        if not skip_conversion:
            success, frame_count, frames_dir = run_convert_video_to_frames(video_path, quality)
            stats['conversion_success'] = success
            stats['frame_count'] = frame_count
            stats['frames_dir'] = frames_dir
            
            if not success:
                stats['error'] = "Conversion failed"
                return stats
        else:
            # Check if frame directory already exists
            video_name = Path(video_path).stem
            video_dir = Path(video_path).parent
            frames_dir = video_dir / f"{video_name}_frames"
            
            if frames_dir.exists():
                frame_files = list(frames_dir.glob("frame_*.jpg"))
                stats['frame_count'] = len(frame_files)
                stats['frames_dir'] = str(frames_dir)
                print(f"✅ Frame directory found: {len(frame_files)} frames")
            else:
                stats['error'] = "Frame directory not found and conversion skipped"
                return stats
        
        # Step 2: AI detection (if not skipped)
        if not skip_detection:
            success, results_file = run_ai_processor(stats['frames_dir'], model_type, yolo_model, skip_frames)
            stats['detection_success'] = success
            stats['results_file'] = results_file
            
            if not success:
                stats['error'] = "AI detection failed"
                return stats
        else:
            print(f"⏭️  AI detection skipped")
        
        return stats
        
    except Exception as e:
        stats['error'] = f"Unexpected error: {e}"
        return stats


def create_batch_summary(results, output_file=None):
    """
    Creates a summary of the batch processing
    
    Args:
        results (list): List of processing statistics
        output_file (str, optional): Path to output file
    """
    total_videos = len(results)
    successful_conversions = sum(1 for r in results if r['conversion_success'])
    successful_detections = sum(1 for r in results if r['detection_success'])
    total_frames = sum(r['frame_count'] for r in results)
    
    summary = {
        "batch_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_videos_processed": total_videos,
            "successful_conversions": successful_conversions,
            "successful_detections": successful_detections,
            "total_frames_processed": total_frames,
            "conversion_rate": (successful_conversions / total_videos * 100) if total_videos > 0 else 0,
            "detection_rate": (successful_detections / total_videos * 100) if total_videos > 0 else 0
        },
        "video_results": results
    }
    
    # Output summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"Videos processed:          {total_videos}")
    print(f"Successful conversions:    {successful_conversions} ({successful_conversions/total_videos*100:.1f}%)")
    print(f"Successful AI detections:  {successful_detections} ({successful_detections/total_videos*100:.1f}%)")
    print(f"Total frames processed:    {total_frames}")
    print(f"{'='*80}")
    
    # List failed videos
    failed_videos = [r for r in results if r['error']]
    if failed_videos:
        print(f"\nFailed videos:")
        for r in failed_videos:
            print(f"  - {r['video_name']}: {r['error']}")
    
    # Save JSON file
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\nBatch summary saved: {output_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Batch Video Processor - Recursive Video Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all videos in a directory:
  python batch_video_processor.py /path/to/video/directory
  
  # With YOLO model:
  python batch_video_processor.py /path/to/video/directory --model yolo --yolo-model yolo12x.pt
  
  # Process only every 5th frame (5x faster):
  python batch_video_processor.py /path/to/video/directory --skip-frames 5
  
  # Only conversion (without AI detection):
  python batch_video_processor.py /path/to/video/directory --skip-detection
  
  # Only AI detection:
  python batch_video_processor.py /path/to/video/directory --skip-conversion
        '''
    )
    
    parser.add_argument(
        'input_dir',
        help='Directory with video files (searched recursively)'
    )
    parser.add_argument(
        '--model',
        choices=['owlv2', 'yolo'],
        default='owlv2',
        help='AI model for detection (default: owlv2)'
    )
    parser.add_argument(
        '--yolo-model',
        default='yolo12x.pt',
        help='YOLO model file (default: yolo12x.pt)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=75,
        help='JPEG quality for frame extraction (1-100, default: 75)'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=1,
        help='Process only every N-th frame (default: 1 = all frames). For 5x speed: --skip-frames 5'
    )
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Skip conversion (frames must already exist)'
    )
    parser.add_argument(
        '--skip-detection',
        action='store_true',
        help='Skip AI detection (only conversion)'
    )
    parser.add_argument(
        '--output-summary',
        help='Path to batch summary file (JSON)'
    )
    
    args = parser.parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        sys.exit(EXIT_INPUT_NOT_FOUND)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Path is not a directory: {args.input_dir}")
        sys.exit(EXIT_INPUT_NOT_FOUND)
    
    # Find video files
    print(f"Searching for video files in: {args.input_dir}")
    video_files = find_video_files(args.input_dir, recursive=True)
    
    if not video_files:
        print(f"No video files found in: {args.input_dir}")
        print(f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(EXIT_NO_VIDEOS_FOUND)
    
    print(f"Found: {len(video_files)} video files")
    for i, video in enumerate(video_files, 1):
        print(f"  {i:2d}. {os.path.basename(video)}")
    
    # Start processing
    start_time = datetime.now()
    print(f"\nProcessing started: {start_time.strftime('%H:%M:%S')}")
    
    results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'#'*80}")
        print(f"VIDEO {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'#'*80}")
        
        stats = process_single_video(
            video_path=video_path,
            model_type=args.model,
            yolo_model=args.yolo_model,
            quality=args.quality,
            skip_conversion=args.skip_conversion,
            skip_detection=args.skip_detection,
            skip_frames=args.skip_frames
        )
        
        results.append(stats)
        
        # Show progress
        elapsed = datetime.now() - start_time
        print(f"\nProgress: {i}/{len(video_files)} videos processed (Time: {elapsed})")
    
    # Create summary
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\nProcessing completed: {end_time.strftime('%H:%M:%S')}")
    print(f"Total duration: {total_time}")
    
    # Batch summary
    summary_file = args.output_summary
    if not summary_file:
        # Create automatic filename
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        summary_file = f"batch_processing_summary_{timestamp}.json"
    
    create_batch_summary(results, summary_file)
    
    # Exit code based on success
    failed_count = sum(1 for r in results if r['error'])
    if failed_count == 0:
        print(f"\n✅ All videos processed successfully!")
        sys.exit(EXIT_SUCCESS)
    elif failed_count < len(results):
        print(f"\n⚠️  {failed_count} of {len(results)} videos failed")
        sys.exit(EXIT_SUCCESS)  # Partially successful
    else:
        print(f"\n❌ All videos failed")
        sys.exit(EXIT_CRITICAL_ERROR)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing cancelled by user")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        print(f"\nCritical error: {e}")
        sys.exit(EXIT_CRITICAL_ERROR)
