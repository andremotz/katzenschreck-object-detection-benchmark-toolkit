#!/usr/bin/env python3
"""
Script for checking JSON files and parallel 'annotated_' folders.

This script searches a given folder for JSON files and checks
whether there is a folder with the prefix 'annotated_' and the same name
(without .json extension) parallel to each JSON file.

For JSON files without annotated folders, annotate_detections.py is automatically
called to create the missing annotations.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json


def find_json_files(directory):
    """
    Finds all JSON files in a directory recursively (including subfolders).
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        list: List of JSON file paths
    """
    json_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return json_files
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        return json_files
    
    # Find all .json files recursively in directory and subfolders
    for file_path in directory_path.rglob('*.json'):
        if file_path.is_file():
            json_files.append(file_path)
    
    return json_files


def check_parallel_folders(json_files):
    """
    Checks for each JSON file whether a parallel folder with "annotated_" prefix exists.
    
    Args:
        json_files (list): List of JSON file paths
        
    Returns:
        dict: Dictionary with results of the check
    """
    results = {
        'with_folder': [],
        'without_folder': [],
        'total_json_files': len(json_files)
    }
    
    for json_file in json_files:
        # Name without .json extension
        base_name = json_file.stem
        
        # Parallel folder path with "annotated_" prefix
        folder_name = f"annotated_{base_name}"
        expected_folder = json_file.parent / folder_name
        
        # Check if the folder exists
        if expected_folder.exists() and expected_folder.is_dir():
            results['with_folder'].append({
                'json_file': str(json_file),
                'folder': str(expected_folder)
            })
        else:
            results['without_folder'].append({
                'json_file': str(json_file),
                'expected_folder': str(expected_folder)
            })
    
    return results


def run_annotation_script(json_file_path):
    """
    Executes annotate_detections.py for a JSON file.
    
    Args:
        json_file_path (Path): Path to the JSON file
        
    Returns:
        bool: True if successful, False on error
    """
    try:
        # Path to annotate_detections.py script (in the same directory)
        script_dir = Path(__file__).parent
        annotate_script = script_dir / "annotate_detections.py"
        
        if not annotate_script.exists():
            print(f"Error: annotate_detections.py not found in {script_dir}")
            return False
        
        print(f"\n  → Running annotate_detections.py for: {json_file_path.name}")
        
        # Execute the script
        result = subprocess.run(
            [sys.executable, str(annotate_script), str(json_file_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ Annotation created successfully")
            return True
        else:
            print(f"  ✗ Annotation failed (Exit Code: {result.returncode})")
            
            # Explain the exit code
            exit_code_meanings = {
                1: "JSON file not found",
                2: "JSON file could not be loaded", 
                3: "Output directory could not be created",
                4: "No frames could be annotated (frame images not found)",
                5: "Some frame annotations failed",
                6: "Frame annotation and video creation partially failed",
                7: "Only video creation failed"
            }
            
            if result.returncode in exit_code_meanings:
                print(f"    Meaning: {exit_code_meanings[result.returncode]}")
            
            if result.stderr:
                print(f"    Stderr: {result.stderr.strip()}")
            if result.stdout:
                print(f"    Stdout: {result.stdout.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: annotate_detections.py took too long")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def process_missing_annotations(results, verbose=False):
    """
    Processes JSON files without annotated folders by calling annotate_detections.py.
    
    Args:
        results (dict): Results of the check
        verbose (bool): Verbose output
        
    Returns:
        bool: True if all annotations successful, False on errors
    """
    if not results['without_folder']:
        return True
    
    print(f"\n=== AUTOMATIC ANNOTATION ===\n")
    print(f"Creating missing annotations for {len(results['without_folder'])} JSON files...\n")
    
    success_count = 0
    failed_files = []
    
    for i, item in enumerate(results['without_folder'], 1):
        json_file_path = Path(item['json_file'])
        
        print(f"[{i}/{len(results['without_folder'])}] Processing: {json_file_path.name}")
        
        success = run_annotation_script(json_file_path)
        
        if success:
            success_count += 1
        else:
            failed_files.append(json_file_path)
            # On error: abort script
            print(f"\n❌ ERROR: Annotation for {json_file_path.name} failed.")
            print(f"Script will be aborted.\n")
            return False
    
    print(f"\n=== ANNOTATION COMPLETED ===\n")
    print(f"Successful: {success_count}/{len(results['without_folder'])} annotations created")
    
    return True


def print_results(results):
    """
    Prints the results in formatted form.
    
    Args:
        results (dict): Results of the check
    """
    print(f"\n=== RESULTS ===")
    print(f"Total {results['total_json_files']} JSON files found (including subfolders).\n")
    
    print(f"JSON files WITH parallel 'annotated_' folder: {len(results['with_folder'])}")
    for item in results['with_folder']:
        json_path = Path(item['json_file'])
        folder_path = Path(item['folder'])
        print(f"  ✓ {json_path.relative_to(Path.cwd()) if json_path.is_relative_to(Path.cwd()) else json_path} → {folder_path.name}/")
    
    print(f"\nJSON files WITHOUT parallel 'annotated_' folder: {len(results['without_folder'])}")
    for item in results['without_folder']:
        json_path = Path(item['json_file'])
        expected_path = Path(item['expected_folder'])
        print(f"  ✗ {json_path.relative_to(Path.cwd()) if json_path.is_relative_to(Path.cwd()) else json_path} → {expected_path.name}/ (not present)")
    
    # Statistics
    if results['total_json_files'] > 0:
        percentage = (len(results['with_folder']) / results['total_json_files']) * 100
        print(f"\nStatistics: {percentage:.1f}% of JSON files have a parallel 'annotated_' folder.")


def main():
    """
    Main function of the script.
    """
    parser = argparse.ArgumentParser(
        description="Checks JSON files for parallel 'annotated_' folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_batch_check_for_annotated-folder_from_json-results-file.py /path/to/folder
  python main_batch_check_for_annotated-folder_from_json-results-file.py ./detection_results
  python main_batch_check_for_annotated-folder_from_json-results-file.py . --no-auto-annotate
  
The script searches for a parallel folder with 'annotated_' prefix for each JSON file.
Example: file.json → annotated_file/

By default, annotate_detections.py is automatically called for missing folders.
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory to search for JSON files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--no-auto-annotate',
        action='store_true',
        help='No automatic annotation for missing folders'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Searching directory recursively: {args.directory}")
    
    # Find JSON files
    json_files = find_json_files(args.directory)
    
    if not json_files:
        print(f"No JSON files found in '{args.directory}' (including subfolders).")
        return
    
    if args.verbose:
        print(f"Found JSON files:")
        for json_file in json_files:
            # Show relative path if possible, otherwise absolute path
            try:
                relative_path = json_file.relative_to(Path.cwd())
                print(f"  - {relative_path}")
            except ValueError:
                print(f"  - {json_file}")
    
    # Check parallel folders
    results = check_parallel_folders(json_files)
    
    # Output results
    print_results(results)
    
    # Automatic annotation for missing folders (if not disabled)
    if not args.no_auto_annotate and results['without_folder']:
        success = process_missing_annotations(results, args.verbose)
        if not success:
            sys.exit(1)
        
        # After annotation: Re-check
        print(f"\n=== RE-CHECK ===\n")
        updated_results = check_parallel_folders(json_files)
        print_results(updated_results)
    
    print(f"\n✅ Script completed successfully.")


if __name__ == "__main__":
    main()
