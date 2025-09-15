#!/usr/bin/env python3
"""
Annotierungs-Script für Detektionsergebnisse

Dieses Script liest JSON-Detektionsergebnisse ein und zeichnet Bounding-Boxes 
auf die entsprechenden Frame-Bilder.

Verwendung:
    python annotate_detections.py detection_results.json [--output-dir output_folder]

Exit-Codes:
    0 - Erfolgreich abgeschlossen
    1 - JSON-Datei nicht gefunden
    2 - JSON-Datei konnte nicht geladen werden
    3 - Ausgabeordner konnte nicht erstellt werden
    4 - Keine Frames konnten annotiert werden
    5 - Einige Frame-Annotationen fehlgeschlagen
    6 - Frame-Annotation und Video-Erstellung teilweise fehlgeschlagen
    7 - Nur Video-Erstellung fehlgeschlagen
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import glob
from datetime import datetime


def load_detection_results(json_path):
    """
    Lädt die Detektionsergebnisse aus einer JSON-Datei
    
    Args:
        json_path (str): Pfad zur JSON-Datei mit Detektionsergebnissen
    
    Returns:
        dict: Die geladenen Detektionsdaten
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"JSON-Datei erfolgreich geladen: {json_path}")
        print(f"Metadata:")
        print(f"  - Timestamp: {data['metadata']['timestamp']}")
        print(f"  - Verarbeitete Frames: {data['metadata']['processed_frames']}")
        print(f"  - Frames mit Detektionen: {data['metadata']['frames_with_detections']}")
        print(f"  - Detektions-Labels: {', '.join(data['metadata']['text_labels'])}")
        print(f"  - Anzahl Frame-Detektionen: {len(data['detections'])}")
        print()
        
        return data
    
    except FileNotFoundError:
        print(f"Fehler: JSON-Datei nicht gefunden: {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Fehler beim Parsen der JSON-Datei: {e}")
        return None
    except Exception as e:
        print(f"Unerwarteter Fehler beim Laden der JSON-Datei: {e}")
        return None


def get_label_color(label):
    """
    Bestimmt die Farbe für ein spezifisches Label
    
    Args:
        label (str): Das Label (z.B. 'cat', 'dog')
    
    Returns:
        tuple: RGB-Farbtupel
    """
    color_map = {
        'cat': (255, 165, 0),    # Orange
        'dog': (0, 255, 0),      # Grün
        'person': (255, 0, 0),   # Rot
        'bird': (0, 0, 255),     # Blau
        'default': (255, 255, 0) # Gelb als Fallback
    }
    
    return color_map.get(label.lower(), color_map['default'])


def draw_bounding_box_pil(image, detection, box_thickness=3, font_size=60):
    """
    Zeichnet eine Bounding-Box mit Label auf ein PIL-Image
    
    Args:
        image (PIL.Image): Das Eingangsbild
        detection (dict): Detektions-Daten mit Label, Confidence und Bounding-Box
        box_thickness (int): Dicke der Bounding-Box-Linien
        font_size (int): Größe der Schrift für Labels
    
    Returns:
        PIL.Image: Das annotierte Bild
    """
    # Kopie des Bildes erstellen
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Bounding-Box Koordinaten extrahieren
    bbox = detection['bounding_box']
    xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
    xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])
    
    # Label und Confidence
    label = detection['label']
    confidence = detection['confidence']
    
    # Farbe basierend auf Label bestimmen
    color = get_label_color(label)
    
    # Bounding-Box zeichnen
    for i in range(box_thickness):
        draw.rectangle([xmin - i, ymin - i, xmax + i, ymax + i], outline=color, width=1)
    
    # Text-Label erstellen
    text = f"{label}: {confidence:.2f}"
    
    # Versuche eine bessere Schrift zu laden, falls verfügbar
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            # Fallback auf Standard-Schrift
            font = ImageFont.load_default()
    
    # Text-Hintergrund-Box berechnen
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Text-Position bestimmen (oberhalb der Bounding-Box)
    text_x = xmin
    text_y = max(0, ymin - text_height - 5)
    
    # Hintergrund für Text zeichnen
    draw.rectangle([text_x, text_y, text_x + text_width + 4, text_y + text_height + 4], 
                   fill=color, outline=color)
    
    # Text zeichnen (in weißer Farbe für bessere Lesbarkeit)
    draw.text((text_x + 2, text_y + 2), text, fill=(255, 255, 255), font=font)
    
    return annotated_image


def draw_bounding_box_cv2(image, detection, box_thickness=3, font_scale=2.4):
    """
    Zeichnet eine Bounding-Box mit Label auf ein OpenCV-Image (als Alternative)
    
    Args:
        image (numpy.ndarray): Das Eingangsbild als OpenCV-Array
        detection (dict): Detektions-Daten mit Label, Confidence und Bounding-Box
        box_thickness (int): Dicke der Bounding-Box-Linien
        font_scale (float): Größe der Schrift für Labels
    
    Returns:
        numpy.ndarray: Das annotierte Bild
    """
    # Kopie des Bildes erstellen
    annotated_image = image.copy()
    
    # Bounding-Box Koordinaten extrahieren
    bbox = detection['bounding_box']
    xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
    xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])
    
    # Label und Confidence
    label = detection['label']
    confidence = detection['confidence']
    
    # Farbe basierend auf Label bestimmen (BGR für OpenCV)
    color_rgb = get_label_color(label)
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    
    # Bounding-Box zeichnen
    cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), color_bgr, box_thickness)
    
    # Text-Label erstellen
    text = f"{label}: {confidence:.2f}"
    
    # Text-Eigenschaften
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Text-Größe berechnen
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
    
    # Text-Position bestimmen (oberhalb der Bounding-Box)
    text_x = xmin
    text_y = max(text_height + 10, ymin - 5)
    
    # Hintergrund für Text zeichnen
    cv2.rectangle(annotated_image, 
                  (text_x, text_y - text_height - 5), 
                  (text_x + text_width + 5, text_y + 5), 
                  color_bgr, -1)
    
    # Text zeichnen (in weißer Farbe für bessere Lesbarkeit)
    cv2.putText(annotated_image, text, (text_x + 2, text_y), font, font_scale, (255, 255, 255), 2)
    
    return annotated_image


def process_single_frame(frame_data, output_dir, use_opencv=False):
    """
    Verarbeitet einen einzelnen Frame und speichert das annotierte Bild
    
    Args:
        frame_data (dict): Daten eines Frames mit Detektionen
        output_dir (str): Ausgabeordner für annotierte Bilder
        use_opencv (bool): Ob OpenCV statt PIL verwendet werden soll
    
    Returns:
        tuple: (success, output_path)
    """
    try:
        # Frame-Bild laden
        frame_path = frame_data['frame_path']
        
        if not os.path.exists(frame_path):
            print(f"Warnung: Frame-Bild nicht gefunden: {frame_path}")
            return False, None
        
        # Ausgabe-Dateiname erstellen
        frame_filename = frame_data['frame_filename']
        output_filename = f"annotated_{frame_filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        if use_opencv:
            # OpenCV-Verarbeitung
            image = cv2.imread(frame_path)
            
            if image is None:
                print(f"Fehler: Konnte Bild nicht laden: {frame_path}")
                return False, None
            
            # Alle Detektionen auf das Bild zeichnen
            for detection in frame_data['detections']:
                image = draw_bounding_box_cv2(image, detection)
            
            # Annotiertes Bild speichern
            cv2.imwrite(output_path, image)
        
        else:
            # PIL-Verarbeitung
            image = Image.open(frame_path)
            
            # Alle Detektionen auf das Bild zeichnen
            for detection in frame_data['detections']:
                image = draw_bounding_box_pil(image, detection)
            
            # Annotiertes Bild speichern
            image.save(output_path, quality=95)
        
        return True, output_path
    
    except Exception as e:
        print(f"Fehler beim Verarbeiten von Frame {frame_data.get('frame_number', 'unbekannt')}: {e}")
        return False, None


def create_video_from_frames(detection_data, output_dir, original_frames_dir=None):
    """
    Erstellt ein H.264-Video aus den annotierten Frames und Original-Frames
    
    Args:
        detection_data (dict): Alle Detektionsdaten
        output_dir (str): Ausgabeordner mit annotierten Frames
        original_frames_dir (str): Pfad zum Original-Frame-Ordner (optional)
    
    Returns:
        str: Pfad zum erstellten Video
    """
    try:
        print("Erstelle Video aus annotierten Frames...")
        
        # Video-Eigenschaften aus Metadaten extrahieren
        metadata = detection_data.get('metadata', {})
        
        # FPS bestimmen - verschiedene Möglichkeiten je nach Input-Typ
        fps = 30.0  # Standard-FPS als Fallback
        
        if 'video_properties' in metadata:
            # Direkte Video-Verarbeitung - FPS aus Video-Eigenschaften
            fps = metadata['video_properties'].get('fps', 30.0)
        elif original_frames_dir and 'frames_folder' in metadata:
            # Frame-Sequenz - schätze FPS basierend auf Frame-Anzahl und vermuteter Dauer
            total_frames = metadata.get('total_frames', 0)
            if total_frames > 0:
                # Annahme: Frame-Sequenzen sind meist mit 30 FPS extrahiert
                fps = 30.0
        
        print(f"Video-FPS: {fps}")
        
        # Alle annotierten Frame-Dateien finden
        annotated_pattern = os.path.join(output_dir, "annotated_frame_*.jpg")
        annotated_files = sorted(glob.glob(annotated_pattern))
        
        if not annotated_files:
            print("Keine annotierten Frames für Video-Erstellung gefunden")
            return None
        
        print(f"Gefunden: {len(annotated_files)} annotierte Frames")
        
        # Erstes Bild laden um Video-Dimensionen zu bestimmen
        first_image = cv2.imread(annotated_files[0])
        if first_image is None:
            print(f"Fehler beim Laden des ersten Frames: {annotated_files[0]}")
            return None
        
        height, width, channels = first_image.shape
        print(f"Video-Auflösung: {width}x{height}")
        
        # Output-Video-Pfad
        video_filename = "annotated_detections.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        # H.264 Video-Writer erstellen
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("Fehler: Video-Writer konnte nicht geöffnet werden")
            return None
        
        # Set mit Frame-Nummern der annotierten Frames
        annotated_frame_numbers = set()
        for file_path in annotated_files:
            filename = os.path.basename(file_path)
            # Extract frame number from "annotated_frame_XXXXXX.jpg"
            try:
                frame_num = int(filename.split('_')[2].split('.')[0])
                annotated_frame_numbers.add(frame_num)
            except (IndexError, ValueError):
                print(f"Warnung: Konnte Frame-Nummer nicht extrahieren aus {filename}")
        
        # Bestimme Frame-Range für Video
        if annotated_frame_numbers:
            min_frame = min(annotated_frame_numbers)
            max_frame = max(annotated_frame_numbers)
        else:
            print("Keine gültigen Frame-Nummern gefunden")
            video_writer.release()
            return None
        
        print(f"Frame-Range: {min_frame} bis {max_frame}")
        
        # Video-Erstellung
        frames_written = 0
        frames_skipped = 0
        
        for frame_num in range(min_frame, max_frame + 1):
            # Prüfe ob annotierter Frame existiert
            annotated_file = os.path.join(output_dir, f"annotated_frame_{frame_num:06d}.jpg")
            
            if os.path.exists(annotated_file):
                # Verwende annotierten Frame
                frame = cv2.imread(annotated_file)
                if frame is not None:
                    video_writer.write(frame)
                    frames_written += 1
                else:
                    frames_skipped += 1
            else:
                # Verwende Original-Frame falls verfügbar
                if original_frames_dir:
                    original_file = os.path.join(original_frames_dir, f"frame_{frame_num:06d}.jpg")
                    if os.path.exists(original_file):
                        frame = cv2.imread(original_file)
                        if frame is not None:
                            # Frame auf Video-Dimensionen skalieren falls nötig
                            if frame.shape[:2] != (height, width):
                                frame = cv2.resize(frame, (width, height))
                            video_writer.write(frame)
                            frames_written += 1
                        else:
                            frames_skipped += 1
                    else:
                        frames_skipped += 1
                else:
                    frames_skipped += 1
            
            # Fortschritt anzeigen
            if (frame_num - min_frame + 1) % 100 == 0:
                progress = (frame_num - min_frame + 1) / (max_frame - min_frame + 1) * 100
                print(f"Video-Erstellung: {progress:.1f}% ({frames_written} Frames geschrieben)")
        
        # Video-Writer schließen
        video_writer.release()
        
        print(f"Video erfolgreich erstellt: {video_path}")
        print(f"Frames geschrieben: {frames_written}")
        print(f"Frames übersprungen: {frames_skipped}")
        print(f"Video-Dauer: {frames_written / fps:.2f} Sekunden")
        
        return video_path
    
    except Exception as e:
        print(f"Fehler beim Erstellen des Videos: {e}")
        return None


def create_summary_image(detection_data, output_dir, max_images=9):
    """
    Erstellt ein Übersichtsbild mit den besten Detektionen
    
    Args:
        detection_data (dict): Alle Detektionsdaten
        output_dir (str): Ausgabeordner
        max_images (int): Maximale Anzahl von Bildern in der Übersicht
    
    Returns:
        str: Pfad zum erstellten Übersichtsbild
    """
    try:
        # Die besten Detektionen auswählen (höchste Confidence)
        best_detections = []
        
        for frame_data in detection_data['detections']:
            for detection in frame_data['detections']:
                best_detections.append({
                    'frame_data': frame_data,
                    'detection': detection,
                    'confidence': detection['confidence']
                })
        
        # Nach Confidence sortieren und die besten auswählen
        best_detections.sort(key=lambda x: x['confidence'], reverse=True)
        selected_detections = best_detections[:max_images]
        
        if not selected_detections:
            print("Keine Detektionen für Übersichtsbild verfügbar")
            return None
        
        # Grid-Layout berechnen
        grid_size = int(np.ceil(np.sqrt(len(selected_detections))))
        
        # Erstes Bild laden um Dimensionen zu bestimmen
        first_frame = selected_detections[0]['frame_data']
        sample_image = Image.open(first_frame['frame_path'])
        
        # Thumbnail-Größe bestimmen
        thumb_width = 300
        thumb_height = int(thumb_width * sample_image.height / sample_image.width)
        
        # Übersichtsbild erstellen
        summary_width = grid_size * thumb_width
        summary_height = grid_size * thumb_height
        summary_image = Image.new('RGB', (summary_width, summary_height), (255, 255, 255))
        
        # Bilder in Grid anordnen
        for i, det_info in enumerate(selected_detections):
            row = i // grid_size
            col = i % grid_size
            
            # Frame laden und annotieren
            frame_data = det_info['frame_data']
            image = Image.open(frame_data['frame_path'])
            
            # Nur die spezifische Detektion zeichnen
            annotated_image = draw_bounding_box_pil(image, det_info['detection'])
            
            # Auf Thumbnail-Größe skalieren
            thumbnail = annotated_image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            
            # In Übersichtsbild einfügen
            x = col * thumb_width
            y = row * thumb_height
            summary_image.paste(thumbnail, (x, y))
        
        # Übersichtsbild speichern
        summary_path = os.path.join(output_dir, "detection_summary.jpg")
        summary_image.save(summary_path, quality=90)
        
        print(f"Übersichtsbild erstellt: {summary_path}")
        return summary_path
    
    except Exception as e:
        print(f"Fehler beim Erstellen des Übersichtsbildes: {e}")
        return None


def main():
    """
    Hauptfunktion des Annotierungs-Scripts
    
    Returns:
        int: Exit-Code (0 = Erfolg, >0 = Fehler)
    """
    
    # Argument Parser einrichten
    parser = argparse.ArgumentParser(
        description='Annotiert Detektionsergebnisse auf Frame-Bildern',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Beispiele:
  # JSON-Datei mit Standard-Ausgabeordner verarbeiten:
  python annotate_detections.py detection_results/Camera_Teich-detection_results.json
  
  # JSON-Datei mit benutzerdefiniertem Ausgabeordner:
  python annotate_detections.py detection_results.json --output-dir annotated_frames
  
  # Mit OpenCV statt PIL (alternative Rendering-Engine):
  python annotate_detections.py detection_results.json --opencv
  
  # Ohne Video-Erstellung:
  python annotate_detections.py detection_results.json --no-video
  
  # Mit Original-Frame-Ordner für vollständiges Video:
  python annotate_detections.py detection_results.json --original-frames-dir /pfad/zu/original/frames
        '''
    )
    
    parser.add_argument(
        'json_file',
        help='Pfad zur JSON-Datei mit Detektionsergebnissen'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Ausgabeordner für annotierte Bilder (Standard: annotated_<json_basename>)'
    )
    
    parser.add_argument(
        '--opencv',
        action='store_true',
        help='Verwendet OpenCV statt PIL für Bild-Verarbeitung'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Erstellt kein Übersichtsbild'
    )
    
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Erstellt kein Video (standardmäßig wird ein Video erstellt)'
    )
    
    parser.add_argument(
        '--original-frames-dir',
        default=None,
        help='Pfad zum Original-Frame-Ordner (für vollständiges Video mit nicht-detektierten Frames)'
    )
    
    args = parser.parse_args()
    
    # JSON-Datei validieren
    if not os.path.exists(args.json_file):
        print(f"Fehler: JSON-Datei nicht gefunden: {args.json_file}")
        return 1  # Exit-Code 1: Eingabedatei nicht gefunden
    
    # Detektionsergebnisse laden
    detection_data = load_detection_results(args.json_file)
    if detection_data is None:
        return 2  # Exit-Code 2: JSON-Datei konnte nicht geladen werden
    
    # Ausgabeordner bestimmen
    if args.output_dir is None:
        # Erstelle Ausgabeordner parallel zum JSON-File
        json_path = Path(args.json_file)
        json_basename = json_path.stem
        json_parent_dir = json_path.parent
        args.output_dir = json_parent_dir / f"annotated_{json_basename}"
    else:
        # Benutzerdefinierter Pfad - als Path-Objekt behandeln
        args.output_dir = Path(args.output_dir)
    
    # Ausgabeordner erstellen
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Ausgabeordner: {args.output_dir}")
    except Exception as e:
        print(f"Fehler beim Erstellen des Ausgabeordners: {e}")
        return 3  # Exit-Code 3: Ausgabeordner konnte nicht erstellt werden
    
    # Verarbeitung starten
    print(f"\nStarte Annotation von {len(detection_data['detections'])} Frames...")
    print("="*60)
    
    successful_annotations = 0
    failed_annotations = 0
    
    for i, frame_data in enumerate(detection_data['detections']):
        frame_number = frame_data['frame_number']
        num_detections = len(frame_data['detections'])
        
        print(f"Frame {frame_number:6d} ({i+1:3d}/{len(detection_data['detections']):3d}) | "
              f"{num_detections} Detektion(en)", end="", flush=True)
        
        success, output_path = process_single_frame(frame_data, str(args.output_dir), args.opencv)
        
        if success:
            successful_annotations += 1
            print(f" -> {os.path.basename(output_path)}")
        else:
            failed_annotations += 1
            print(" -> Fehler!")
    
    # Prüfe ob überhaupt Frames verarbeitet wurden
    if successful_annotations == 0:
        print(f"\nFehler: Keine Frames konnten erfolgreich annotiert werden!")
        return 4  # Exit-Code 4: Keine erfolgreichen Annotationen
    
    # Übersichtsbild erstellen
    if not args.no_summary:
        print(f"\nErstelle Übersichtsbild...")
        summary_result = create_summary_image(detection_data, str(args.output_dir))
        if summary_result is None:
            print("Warnung: Übersichtsbild konnte nicht erstellt werden")
    
    # Video erstellen (standardmäßig aktiviert)
    video_creation_failed = False
    if not args.no_video:
        print(f"\nErstelle Video aus annotierten Frames...")
        
        # Original-Frame-Ordner bestimmen falls nicht angegeben
        original_frames_dir = args.original_frames_dir
        if original_frames_dir is None and 'frames_folder' in detection_data.get('metadata', {}):
            # Versuche Original-Frame-Ordner aus Metadaten zu ermitteln
            original_frames_dir = detection_data['metadata']['frames_folder']
        
        video_path = create_video_from_frames(detection_data, str(args.output_dir), original_frames_dir)
        if video_path:
            print(f"Video erfolgreich erstellt: {video_path}")
        else:
            print("Video-Erstellung fehlgeschlagen")
            video_creation_failed = True
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("ANNOTATION ABGESCHLOSSEN")
    print("="*60)
    print(f"Erfolgreich annotiert: {successful_annotations}")
    print(f"Fehlgeschlagen:        {failed_annotations}")
    print(f"Ausgabeordner:         {args.output_dir}")
    print(f"Verwendete Engine:     {'OpenCV' if args.opencv else 'PIL'}")
    
    if successful_annotations > 0:
        print(f"\nDie annotierten Bilder befinden sich in: {os.path.abspath(args.output_dir)}")
        if not args.no_video:
            video_file = os.path.join(str(args.output_dir), "annotated_detections.mp4")
            if os.path.exists(video_file):
                print(f"Das annotierte Video befindet sich in: {os.path.abspath(video_file)}")
    
    print("="*60)
    
    # Exit-Code bestimmen
    if failed_annotations > 0 and video_creation_failed:
        return 6  # Exit-Code 6: Sowohl Frame-Annotation als auch Video-Erstellung teilweise fehlgeschlagen
    elif failed_annotations > 0:
        return 5  # Exit-Code 5: Einige Frame-Annotationen fehlgeschlagen
    elif video_creation_failed:
        return 7  # Exit-Code 7: Nur Video-Erstellung fehlgeschlagen
    else:
        return 0  # Exit-Code 0: Alles erfolgreich


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
