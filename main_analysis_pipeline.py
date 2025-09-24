#!/usr/bin/env python3
"""
Main processing pipeline for video-to-AI object detection
Combines convert_video_to_image_sequence.py and ai-processor.py into a single pipeline
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(command, description):
    """
    Executes a system command and provides detailed feedback
    
    Args:
        command (list): The command to execute as a list
        description (str): Description of the command for output
    
    Returns:
        tuple: (success, stdout, stderr)
    """
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 60)
    
    try:
        # subprocess.Popen for live output with simultaneous output collection
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_lines = []
        
        # Show live output and collect simultaneously
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                stdout_lines.append(output)
        
        # Wait until process is finished
        return_code = process.poll()
        stdout = ''.join(stdout_lines)
        
        if return_code == 0:
            print(f"\n✅ {description} completed successfully")
            return True, stdout, ""
        else:
            print(f"\n❌ {description} failed (Exit Code: {return_code})")
            return False, stdout, ""
            
    except Exception as e:
        print(f"❌ Error executing '{description}': {e}")
        return False, "", str(e)


def validate_video_file(video_path):
    """
    Validates if the video file exists and is readable
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return False
    
    if not os.path.isfile(video_path):
        print(f"❌ Error: '{video_path}' is not a file")
        return False
    
    # Check for common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wmv', '.flv']
    file_extension = Path(video_path).suffix.lower()
    
    if file_extension not in video_extensions:
        print(f"⚠️  Warning: '{file_extension}' may not be a supported video format")
        print(f"   Supported formats: {', '.join(video_extensions)}")
    
    return True


def main():
    """Main function of the processing pipeline"""
    
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Complete video-to-AI object detection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main_analysis_pipeline.py /path/to/video.mp4
  python main_analysis_pipeline.py /path/to/video.mp4 --quality 90
  python main_analysis_pipeline.py video.mp4 --output-dir /custom/output
        '''
    )
    
    parser.add_argument(
        'video_path',
        help='Path to the video file to be processed'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for frames (optional, will be created automatically)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality for frames (1-100, default: 95)'
    )
    
    args = parser.parse_args()
    
    # Start pipeline
    start_time = datetime.now()
    print("🚀 Video-to-AI object detection pipeline started")
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Validate video file
    print(f"📹 Input video: {args.video_path}")
    if not validate_video_file(args.video_path):
        sys.exit(1)
    
    # Determine script paths
    script_dir = Path(__file__).parent
    convert_script = script_dir / "convert_video_to_image_sequence.py"
    ai_script = script_dir / "ai-processor.py"
    
    # Check if required scripts exist
    if not convert_script.exists():
        print(f"❌ Error: convert_video_to_image_sequence.py not found in {script_dir}")
        sys.exit(1)
    
    if not ai_script.exists():
        print(f"❌ Error: ai-processor.py not found in {script_dir}")
        sys.exit(1)
    
    print(f"✅ All required scripts found")
    
    # STEP 1: Convert video to image sequence
    print(f"\n📋 STEP 1: Convert video to image sequence")
    
    convert_command = [
        sys.executable,
        str(convert_script),
        args.video_path
    ]
    
    # Add parameters for convert_video_to_image_sequence.py
    # Order: video_path, output_dir, quality, frame_skip
    if args.output_dir:
        # If output dir specified, set all parameters
        convert_command.extend([args.output_dir, str(args.quality), "1"])
    else:
        # No output dir specified - script automatically creates videoname_frames
        # We still need to pass parameters in the correct order
        # So: video_path (already set), output_dir (leave empty), quality, frame_skip
        # Since we want to omit output_dir, we only pass quality and frame_skip if necessary
        if args.quality != 95:
            convert_command.extend([str(args.quality), "1"])
        # If default quality: no additional parameters needed
    
    success, stdout, stderr = run_command(convert_command, "Video-to-frames conversion")
    
    if not success:
        print(f"❌ Pipeline aborted: Video conversion failed")
        sys.exit(1)
    
    # Determine output directory from conversion
    # The convert_video_to_image_sequence.py outputs the output directory
    frames_dir = None
    for line in stdout.split('\n'):
        if 'Output directory:' in line or 'Output-Verzeichnis:' in line:
            frames_dir = line.split(':')[1].strip()
            break
        elif 'Images saved to:' in line or 'Bilder gespeichert in:' in line:
            frames_dir = line.split(':')[1].strip()
            break
    
    if not frames_dir or not os.path.exists(frames_dir):
        print(f"❌ Error: Could not determine frame directory or it does not exist")
        print(f"   Detected directory: {frames_dir}")
        sys.exit(1)
    
    print(f"✅ Frames successfully created in: {frames_dir}")
    
    # STEP 2: Execute AI object detection
    print(f"\n🤖 STEP 2: AI object detection")
    
    ai_command = [
        sys.executable,
        str(ai_script),
        frames_dir
    ]
    
    success, stdout, stderr = run_command(ai_command, "AI object detection")
    
    if not success:
        print(f"❌ Pipeline aborted: AI processing failed")
        print(f"🗑️  Frame directory will be kept for debugging: {frames_dir}")
        sys.exit(1)
    
    # Successful processing
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"⏰ Start time:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ End time:       {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {total_time}")
    print(f"📹 Input video: {args.video_path}")
    print(f"📁 Frame directory: {frames_dir}")
    
    # Determine result JSON - use absolute paths
    frames_folder_name = os.path.basename(frames_dir)
    frames_parent_dir = os.path.abspath(os.path.dirname(frames_dir))
    result_json = os.path.join(frames_parent_dir, f"{frames_folder_name}_detection_results.json")
    
    if os.path.exists(result_json):
        print(f"📊 Results: {result_json}")
    
    # Frame directory is always kept
    print(f"📁 Frame directory kept: {frames_dir}")
    
    print("=" * 80)
    print("🏁 Pipeline finished")


if __name__ == "__main__":
    main()
