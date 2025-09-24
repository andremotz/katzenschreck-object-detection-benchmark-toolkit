#!/usr/bin/env python3
"""
Video Batch Checker & Converter
Checks all video files in the footage folder and verifies
if corresponding _frames subfolders exist.
Automatically converts videos without _frames folders using convert_video_to_image_sequences.py.
"""

import os
import sys
import subprocess
from pathlib import Path


def find_video_files(directory):
    """
    Findet alle Videodateien in einem Verzeichnis und seinen Unterverzeichnissen
    
    Args:
        directory (str): Pfad zum Verzeichnis
    
    Returns:
        list: Liste der gefundenen Videodateien
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.webm'}
    video_files = []
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Fehler: Verzeichnis nicht gefunden: {directory}")
        return []
    
    # Rekursiv durch alle Unterverzeichnisse gehen
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)


def check_frames_folder(video_file):
    """
    Prüft, ob für eine Videodatei ein entsprechender _frames Ordner existiert
    
    Args:
        video_file (Path): Pfad zur Videodatei
    
    Returns:
        tuple: (bool, Path or None) - (frames_folder_exists, frames_folder_path)
    """
    video_stem = video_file.stem  # Dateiname ohne Erweiterung
    video_dir = video_file.parent
    
    frames_folder_name = f"{video_stem}_frames"
    frames_folder_path = video_dir / frames_folder_name
    
    return frames_folder_path.exists(), frames_folder_path if frames_folder_path.exists() else None


def convert_video_to_frames(video_file, script_path):
    """
    Führt convert_video_to_image_sequences.py für eine Videodatei aus
    
    Args:
        video_file (Path): Pfad zur Videodatei
        script_path (Path): Pfad zum convert_video_to_image_sequences.py Script
    
    Returns:
        tuple: (success, output) - (Erfolgreich, Ausgabe des Scripts)
    """
    try:
        print(f"🔄 Konvertiere: {video_file.name}")
        print(f"   Starte convert_video_to_image_sequences.py...")
        
        # Script mit der Videodatei als Parameter ausführen
        result = subprocess.run([
            sys.executable,  # Python Interpreter
            str(script_path),
            str(video_file)
        ], capture_output=True, text=True, timeout=3600)  # 1 Stunde Timeout
        
        if result.returncode == 0:
            print(f"   ✅ Erfolgreich konvertiert!")
            return True, result.stdout
        else:
            print(f"   ❌ Fehler bei der Konvertierung!")
            print(f"   Fehlerausgabe: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout: Konvertierung dauerte zu lange (>1h)")
        return False, "Timeout erreicht"
    except Exception as e:
        print(f"   ❌ Unerwarteter Fehler: {str(e)}")
        return False, str(e)


def main():
    """Hauptfunktion"""
    
    # Standard-Pfad zum Camera_Teich-Footage Ordner - absolute Pfad verwenden
    default_footage_dir = os.path.abspath("/Volumes/documents/Camera_Teich-Footage")
    
    # Pfad zum convert_video_to_image_sequences.py Script
    script_dir = Path(__file__).parent
    converter_script = script_dir / "convert_video_to_image_sequences.py"
    
    # Prüfen ob das Converter-Script existiert
    if not converter_script.exists():
        print(f"❌ Fehler: convert_video_to_image_sequences.py nicht gefunden in {script_dir}")
        print("   Das Script muss im gleichen Verzeichnis sein!")
        return
    
    # Verzeichnis aus Kommandozeile oder Standard verwenden
    if len(sys.argv) > 1:
        footage_dir = sys.argv[1]
    else:
        footage_dir = default_footage_dir
        print(f"Kein Verzeichnis angegeben, verwende Standard-Pfad:")
        print(f"  {footage_dir}")
        print()
    
    print("=== Video Batch Checker & Converter ===")
    print(f"Scanne Verzeichnis: {footage_dir}")
    print(f"Converter Script: {converter_script}")
    print()
    
    # Alle Videodateien finden
    print("Suche nach Videodateien...")
    video_files = find_video_files(footage_dir)
    
    if not video_files:
        print("Keine Videodateien gefunden.")
        return
    
    print(f"Gefunden: {len(video_files)} Videodateien")
    print()
    
    # Für jede Videodatei prüfen, ob _frames Ordner existiert
    videos_with_frames = []
    videos_without_frames = []
    
    for video_file in video_files:
        has_frames, frames_path = check_frames_folder(video_file)
        
        if has_frames:
            videos_with_frames.append((video_file, frames_path))
        else:
            videos_without_frames.append(video_file)
    
    # Ergebnisse ausgeben
    print("=" * 60)
    print(f"ERGEBNISSE:")
    print("=" * 60)
    print()
    
    if videos_with_frames:
        print(f"✅ Videos MIT entsprechenden _frames Ordnern ({len(videos_with_frames)}):")
        print("-" * 50)
        for video_file, frames_path in videos_with_frames:
            print(f"📹 {video_file.name}")
            print(f"   📁 {frames_path}")
            print()
    else:
        print("✅ Keine Videos mit _frames Ordnern gefunden.")
        print()
    
    # Automatische Konvertierung für Videos ohne _frames Ordner
    if videos_without_frames:
        print(f"❌ Videos OHNE entsprechende _frames Ordner ({len(videos_without_frames)}):")
        print("-" * 50)
        for video_file in videos_without_frames:
            print(f"📹 {video_file.name}")
            print(f"   📂 {video_file.parent}")
        print()
        
        # Automatische Konvertierung starten
        print("🚀 Starte automatische Konvertierung...")
        print("=" * 60)
        
        successful_conversions = []
        failed_conversions = []
        
        for i, video_file in enumerate(videos_without_frames, 1):
            print(f"\n📹 Video {i}/{len(videos_without_frames)}: {video_file.name}")
            print("-" * 50)
            
            success, output = convert_video_to_frames(video_file, converter_script)
            
            if success:
                successful_conversions.append(video_file)
                # Prüfen ob _frames Ordner jetzt existiert
                has_frames_now, frames_path = check_frames_folder(video_file)
                if has_frames_now:
                    videos_with_frames.append((video_file, frames_path))
            else:
                failed_conversions.append((video_file, output))
        
        # Konvertierungs-Ergebnisse
        print("\n" + "=" * 60)
        print("KONVERTIERUNGS-ERGEBNISSE:")
        print("=" * 60)
        
        if successful_conversions:
            print(f"✅ Erfolgreich konvertiert ({len(successful_conversions)}):")
            for video_file in successful_conversions:
                print(f"   📹 {video_file.name}")
            print()
        
        if failed_conversions:
            print(f"❌ Fehlgeschlagen ({len(failed_conversions)}):")
            for video_file, error in failed_conversions:
                print(f"   📹 {video_file.name}")
                print(f"      💥 {error[:100]}...")
            print()
    
    # Zusammenfassung
    print("=" * 60)
    print("FINAL-ZUSAMMENFASSUNG:")
    print(f"  Gesamt Videodateien: {len(video_files)}")
    print(f"  Mit _frames Ordner:  {len(videos_with_frames)}")
    print(f"  Ohne _frames Ordner: {len(video_files) - len(videos_with_frames)}")
    print("=" * 60)


if __name__ == "__main__":
    print("Verwendung:")
    print("  python convert_video_batch.py [verzeichnis_pfad]")
    print()
    print("Parameter:")
    print("  verzeichnis_pfad:  Pfad zum Camera_Teich-Footage Verzeichnis")
    print("                     (optional, Standard ist der konfigurierte Pfad)")
    print()
    
    main()
