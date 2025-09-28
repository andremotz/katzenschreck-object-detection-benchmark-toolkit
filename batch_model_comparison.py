#!/usr/bin/env python3
"""
Batch Model Comparison Script

This script runs ai-processor.py multiple times with different YOLO models
for comprehensive model comparison and benchmarking.

Supports YOLOv8 and YOLOv12 model variants (n, s, m, l, x).
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime


def run_ai_processor(input_path, model_name, verbose=False):
    """
    Runs ai-processor.py with a specific YOLO model.
    
    Args:
        input_path (str): Path to video or frame folder
        model_name (str): YOLO model name (e.g., 'yolo12x.pt')
        verbose (bool): Show detailed output
        
    Returns:
        tuple: (success, execution_time, stdout, stderr)
    """
    script_dir = Path(__file__).parent
    ai_processor_script = script_dir / "ai-processor.py"
    
    if not ai_processor_script.exists():
        print(f"❌ Error: ai-processor.py not found in {script_dir}")
        return False, 0, "", "ai-processor.py not found"
    
    command = [
        sys.executable,
        str(ai_processor_script),
        "--model", "yolo",
        "--yolo-model", model_name,
        input_path
    ]
    
    print(f"\n🔄 Running: {model_name}")
    print(f"Command: {' '.join(command)}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        if verbose:
            # Show live output
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    stdout_lines.append(output)
            
            return_code = process.poll()
            stdout = ''.join(stdout_lines)
            stderr = ""
        else:
            # Capture output without showing
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            return_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        
        execution_time = time.time() - start_time
        
        if return_code == 0:
            print(f"✅ {model_name} completed successfully in {execution_time:.1f}s")
            return True, execution_time, stdout, stderr
        else:
            print(f"❌ {model_name} failed (Exit Code: {return_code})")
            if stderr:
                print(f"Error: {stderr.strip()}")
            return False, execution_time, stdout, stderr
            
    except subprocess.TimeoutExpired:
        print(f"❌ {model_name} timed out (30 minutes)")
        return False, time.time() - start_time, "", "Timeout"
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ {model_name} failed with exception: {e}")
        return False, execution_time, "", str(e)


def get_model_list(include_v8=True, include_v12=True, model_sizes=None):
    """
    Generates list of YOLO models to test.
    
    Args:
        include_v8 (bool): Include YOLOv8 models
        include_v12 (bool): Include YOLOv12 models
        model_sizes (list): Model sizes to include (n, s, m, l, x)
        
    Returns:
        list: List of model filenames
    """
    if model_sizes is None:
        model_sizes = ['n', 's', 'm', 'l', 'x']
    
    models = []
    
    if include_v8:
        for size in model_sizes:
            models.append(f"yolo8{size}.pt")
    
    if include_v12:
        for size in model_sizes:
            models.append(f"yolo12{size}.pt")
    
    return models


def print_summary(results):
    """
    Prints a summary of all model runs.
    
    Args:
        results (list): List of result tuples
    """
    print("\n" + "=" * 80)
    print("📊 BATCH MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"Total models tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ Successful models:")
        for model, success, exec_time, _, _ in successful:
            print(f"  • {model:<15} - {exec_time:.1f}s")
    
    if failed:
        print(f"\n❌ Failed models:")
        for model, success, exec_time, _, error in failed:
            print(f"  • {model:<15} - {error}")
    
    if successful:
        total_time = sum(r[2] for r in successful)
        avg_time = total_time / len(successful)
        print(f"\nPerformance Statistics:")
        print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average time per model: {avg_time:.1f}s")
        
        # Sort by execution time
        fastest = min(successful, key=lambda x: x[2])
        slowest = max(successful, key=lambda x: x[2])
        print(f"Fastest model: {fastest[0]} ({fastest[2]:.1f}s)")
        print(f"Slowest model: {slowest[0]} ({slowest[2]:.1f}s)")


def main():
    """
    Main function for batch model comparison.
    """
    parser = argparse.ArgumentParser(
        description='Run ai-processor.py with multiple YOLO models for comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python batch_model_comparison.py /path/to/video.mp4
  python batch_model_comparison.py /path/to/frames/ --no-v8
  python batch_model_comparison.py video.mp4 --sizes n s m --verbose
  python batch_model_comparison.py video.mp4 --only-v12 --sizes l x
        '''
    )
    
    parser.add_argument(
        'input_path',
        help='Path to video file or frame folder to process'
    )
    
    parser.add_argument(
        '--no-v8',
        action='store_true',
        help='Skip YOLOv8 models'
    )
    
    parser.add_argument(
        '--no-v12',
        action='store_true',
        help='Skip YOLOv12 models'
    )
    
    parser.add_argument(
        '--only-v8',
        action='store_true',
        help='Only run YOLOv8 models'
    )
    
    parser.add_argument(
        '--only-v12',
        action='store_true',
        help='Only run YOLOv12 models'
    )
    
    parser.add_argument(
        '--sizes',
        nargs='+',
        choices=['n', 's', 'm', 'l', 'x'],
        default=['n', 's', 'm', 'l', 'x'],
        help='Model sizes to test (default: all sizes)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output from ai-processor.py'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue with other models if one fails'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input path '{args.input_path}' does not exist.")
        sys.exit(1)
    
    # Determine which model versions to include
    include_v8 = not args.no_v8 and not args.only_v12
    include_v12 = not args.no_v12 and not args.only_v8
    
    if not include_v8 and not include_v12:
        print("❌ Error: No model versions selected. Use --only-v8 or --only-v12, or remove exclusion flags.")
        sys.exit(1)
    
    # Get list of models to test
    models = get_model_list(include_v8, include_v12, args.sizes)
    
    if not models:
        print("❌ Error: No models selected for testing.")
        sys.exit(1)
    
    # Start batch processing
    start_time = datetime.now()
    print("🚀 Starting Batch Model Comparison")
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📹 Input: {args.input_path}")
    print(f"🤖 Models to test: {len(models)}")
    print(f"📋 Model list: {', '.join(models)}")
    print("=" * 80)
    
    results = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing with {model}")
        
        success, exec_time, stdout, stderr = run_ai_processor(
            args.input_path, 
            model, 
            args.verbose
        )
        
        results.append((model, success, exec_time, stdout, stderr))
        
        if not success and not args.continue_on_error:
            print(f"\n❌ Stopping batch processing due to failure with {model}")
            print("Use --continue-on-error to continue with remaining models")
            break
    
    # Print final summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print_summary(results)
    
    print(f"\n⏰ Batch processing completed")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration}")
    print("=" * 80)
    
    # Exit with error code if any models failed
    failed_count = len([r for r in results if not r[1]])
    if failed_count > 0:
        print(f"⚠️  {failed_count} model(s) failed")
        sys.exit(1)
    else:
        print("🎉 All models completed successfully!")


if __name__ == "__main__":
    main()
