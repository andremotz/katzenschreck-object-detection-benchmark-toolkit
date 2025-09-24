#!/usr/bin/env python3
"""
Video to Image Sequence Converter
Converts a video into individual JPG images and saves them in a subfolder
"""

import cv2
import os
import sys
from pathlib import Path


def convert_video_to_images(video_path, output_dir=None, quality=95):
    """
    Konvertiert ein Video in eine Bildsequenz
    
    Args:
        video_path (str): Pfad zum Input-Video
        output_dir (str, optional): Output-Verzeichnis. Falls None, wird automatisch erstellt
        quality (int): JPEG Qualität (1-100, Standard: 95)
    
    Returns:
        tuple: (success, frame_count, output_directory)
    """
    
    # Prüfen ob Video existiert
    if not os.path.exists(video_path):
        print(f"Fehler: Video nicht gefunden: {video_path}")
        return False, 0, None
    
    # Video-Informationen
    video_name = Path(video_path).stem
    video_dir = Path(video_path).parent
    
    # Output-Verzeichnis erstellen falls nicht angegeben
    if output_dir is None:
        output_dir = video_dir / f"{video_name}_frames"
    else:
        output_dir = Path(output_dir)
    
    # Prüfen ob Output-Verzeichnis bereits existiert
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Info: Output-Verzeichnis existiert bereits und ist nicht leer: {output_dir}")
        print("Überspringe Konvertierung - Frames scheinen bereits extrahiert zu sein.")
        # Zähle vorhandene Bilder
        existing_images = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg")) + list(output_dir.glob("*.png"))
        print(f"Gefundene Bilder: {len(existing_images)}")
        return True, len(existing_images), str(output_dir)
    
    # Verzeichnis erstellen
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Video öffnen
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Fehler: Video konnte nicht geöffnet werden: {video_path}")
        return False, 0, None
    
    # Video-Eigenschaften
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video-Informationen:")
    print(f"  Datei: {video_path}")
    print(f"  Frames gesamt: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Dauer: {duration:.2f} Sekunden")
    print(f"  Output-Verzeichnis: {output_dir}")
    print()
    
    frame_count = 0
    
    # JPEG-Qualitäts-Parameter
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Filename mit führenden Nullen für richtige Sortierung
            filename = f"frame_{frame_count:06d}.jpg"
            output_path = output_dir / filename
            
            # Frame als JPG speichern
            success = cv2.imwrite(str(output_path), frame, jpeg_params)
            
            if success:
                frame_count += 1
                if frame_count % 100 == 0:  # Progress alle 100 Frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - {frame_count} Bilder gespeichert")
            else:
                print(f"Warnung: Frame {frame_count} konnte nicht gespeichert werden")
                frame_count += 1
    
    except KeyboardInterrupt:
        print("\nKonvertierung durch Benutzer abgebrochen")
    
    finally:
        cap.release()
    
    print(f"\nKonvertierung abgeschlossen!")
    print(f"  {frame_count} Bilder gespeichert in: {output_dir}")
    print(f"  Verarbeitete Frames: {frame_count} von {total_frames}")
    
    return True, frame_count, str(output_dir)


def main():
    """Hauptfunktion mit Kommandozeilen-Interface"""
    
    # Standard-Video-Pfad aus der Anfrage - absolute Pfad verwenden
    default_video = os.path.abspath("/Users/andremotz/Nextcloud/Documents/Projekte/intern - Katzenschreck/Camera_Teich-Footage/20250907PM/Camera_Teich-20250907-155316-1757253196613-7.mp4")
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = default_video
        print(f"Kein Video-Pfad angegeben, verwende Standard-Video:")
        print(f"  {video_path}")
        print()
    
    # Optionale Parameter
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else 95
    
    # Konvertierung starten
    success, frame_count, output_path = convert_video_to_images(
        video_path=video_path,
        output_dir=output_dir,
        quality=quality
    )
    
    if success:
        print(f"\n✅ Erfolgreich! {frame_count} Bilder gespeichert.")
    else:
        print(f"\n❌ Fehler bei der Konvertierung.")
        sys.exit(1)


if __name__ == "__main__":
    print("=== Video zu Bildsequenz Konverter ===")
    print()
    print("Verwendung:")
    print("  python convert_video_to_image_sequences.py [video_pfad] [output_dir] [qualität]")
    print()
    print("Parameter:")
    print("  video_pfad:  Pfad zum Input-Video (optional, Standard ist das angegebene Video)")
    print("  output_dir:  Output-Verzeichnis (optional, wird automatisch erstellt)")
    print("  qualität:    JPEG-Qualität 1-100 (optional, Standard: 95)")
    print()
    
    main()
