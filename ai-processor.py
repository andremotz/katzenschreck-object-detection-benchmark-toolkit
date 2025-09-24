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

# OWLv2 imports
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warnung: ultralytics nicht installiert. YOLO-Modelle nicht verfügbar.")

# Exit-Codes für verschiedene Fehlerfälle
EXIT_SUCCESS = 0
EXIT_INPUT_NOT_FOUND = 1          # Input-Pfad existiert nicht
EXIT_UNKNOWN_INPUT_TYPE = 2       # Unbekannter Input-Typ
EXIT_VIDEO_NOT_FOUND = 3          # Video-Datei nicht gefunden
EXIT_VIDEO_OPEN_FAILED = 4        # Video konnte nicht geöffnet werden
EXIT_NO_FRAMES_FOUND = 5          # Keine Frame-Bilder gefunden
EXIT_VIDEO_PROCESSING_FAILED = 6  # Allgemeiner Video-Verarbeitungsfehler
EXIT_MODEL_LOADING_FAILED = 7     # AI-Modell konnte nicht geladen werden
EXIT_JSON_WRITE_FAILED = 8        # JSON-Datei konnte nicht geschrieben werden
EXIT_CRITICAL_ERROR = 9           # Kritischer unbekannter Fehler


def clear_memory(device):
    """
    Bereinigt GPU/MPS Memory um Out-of-Memory-Fehler zu vermeiden
    
    Args:
        device: Das verwendete Compute-Device
    """
    # Garbage Collection für Python-Objekte
    gc.collect()
    
    # PyTorch Memory Cache leeren
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()


def process_frame_with_fallback_owlv2(processor, model, device, image, text_labels):
    """
    Versucht Frame-Verarbeitung mit OWLv2 und robustem Error-Handling
    
    Args:
        processor: Der AI-Processor
        model: Das AI-Modell
        device: Primäres Device (GPU/MPS)
        image: Das zu verarbeitende Bild
        text_labels: Die zu suchenden Labels
    
    Returns:
        tuple: (success, outputs, used_device)
    """
    # Erster Versuch: GPU/MPS
    try:
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return True, outputs, device
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "should be the same" in str(e).lower():
            print(f" -> GPU-Problem! Bereinige Cache und versuche erneut...")
            clear_memory(device)
            
            # Zweiter Versuch: GPU/MPS nach Memory-Clearing
            try:
                inputs = processor(text=text_labels, images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                return True, outputs, device
            except RuntimeError as e2:
                print(f" -> Persistentes GPU-Problem, überspringe Frame: {str(e2)[:100]}...")
                return False, None, None
        else:
            print(f" -> Unbekannter GPU-Fehler, überspringe Frame: {str(e)[:100]}...")
            return False, None, None


def process_frame_with_fallback_yolo(model, device, image, target_classes):
    """
    Versucht Frame-Verarbeitung mit YOLO und robustem Error-Handling
    
    Args:
        model: Das YOLO-Modell
        device: Primäres Device (GPU/MPS/CPU)
        image: Das zu verarbeitende Bild (PIL Image)
        target_classes: Liste der zu erkennenden Klassen (z.B. ['cat', 'dog'])
    
    Returns:
        tuple: (success, detections, used_device) 
               detections: Liste von Dicts mit 'label', 'confidence', 'bounding_box'
    """
    try:
        # PIL Image zu numpy array für YOLO
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
                    
                    # Nur relevante Klassen berücksichtigen
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
        print(f" -> YOLO-Fehler, überspringe Frame: {str(e)[:100]}...")
        return False, [], None


def process_video_directly(video_path, processor, model, device, text_labels, model_type='owlv2'):
    """
    Verarbeitet ein Video direkt ohne Frame-Zwischenspeicherung
    
    Args:
        video_path (str): Pfad zum Input-Video
        processor: Der AI-Processor für Objekterkennung
        model: Das AI-Modell für Objekterkennung
        device: Das Compute-Device (CPU/GPU)
        text_labels: Liste der zu erkennenden Objekte
    
    Returns:
        tuple: (success, results_data)
    """
    
    # Prüfen ob Video existiert
    if not os.path.exists(video_path):
        print(f"Fehler: Video nicht gefunden: {video_path}")
        sys.exit(EXIT_VIDEO_NOT_FOUND)
    
    # Video öffnen
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Fehler: Video konnte nicht geöffnet werden: {video_path}")
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
    print(f"  Verarbeitung: Direkt aus Video-Stream")
    print()
    
    # JSON-Datenstruktur für Ergebnisse
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
            "model_type": model_type,
            "processing_method": "direct_video_stream"
        },
        "detections": []
    }
    
    # Zähler für Fortschrittsanzeige  
    processed_count = 0
    local_start_time = datetime.now()
    
    print(f"\nStarte Verarbeitung um {local_start_time.strftime('%H:%M:%S')}...")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # OpenCV Frame (BGR) zu PIL Image (RGB) konvertieren
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Frame-Verarbeitung zählen
            processed_count += 1
            results_data["metadata"]["processed_frames"] = processed_count
            
            # Fortschrittsanzeige
            progress_percent = (processed_count / total_frames) * 100
            elapsed_time = datetime.now() - local_start_time
            
            # Geschätzte Restzeit berechnen
            if processed_count > 0:
                avg_time_per_frame = elapsed_time.total_seconds() / processed_count
                remaining_frames = total_frames - processed_count
                estimated_remaining = remaining_frames * avg_time_per_frame
                remaining_str = str(timedelta(seconds=int(estimated_remaining)))
            else:
                remaining_str = "unbekannt"
            
            print(f"Progress: {progress_percent:5.1f}% ({processed_count:3d}/{total_frames}) | Frame {processed_count-1:6d} | Restzeit: {remaining_str}", end="", flush=True)
            
            # AI-Verarbeitung - unterschiedlich je nach Modell
            if model_type == 'yolo':
                target_classes = text_labels[0] if isinstance(text_labels[0], list) else text_labels
                success, detections, used_device = process_frame_with_fallback_yolo(
                    model, device, image, target_classes
                )
                
                if not success:
                    print(f" -> Frame-Verarbeitung fehlgeschlagen, überspringe...")
                    continue
                
                # Frame-Daten zu JSON hinzufügen
                frame_data = {
                    "frame_number": processed_count - 1,
                    "frame_timestamp": (processed_count - 1) / fps if fps > 0 else 0,
                    "image_size": {
                        "width": image.width,
                        "height": image.height
                    },
                    "detections": detections
                }
                
                # Nur zu Ergebnissen hinzufügen wenn Detektionen gefunden wurden
                if len(detections) > 0:
                    results_data["detections"].append(frame_data)
                    results_data["metadata"]["frames_with_detections"] += 1
                    print(f" -> {len(detections)} Detektion(en)!")
                else:
                    print(" -> keine Detektionen")
                    
            else:  # OWLv2
                success, outputs, used_device = process_frame_with_fallback_owlv2(
                    processor, model, device, image, text_labels
                )
                
                if not success:
                    print(f" -> Frame-Verarbeitung fehlgeschlagen, überspringe...")
                    continue
                
                # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                target_sizes = torch.tensor([(image.height, image.width)]).to(used_device)
                
                # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
                results = processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
                )
                
                # Retrieve predictions for the first image for the corresponding text queries
                result = results[0]
                boxes, scores, text_labels_result = result["boxes"], result["scores"], result["text_labels"]
                
                # Frame-Daten zu JSON hinzufügen
                frame_data = {
                    "frame_number": processed_count - 1,
                    "frame_timestamp": (processed_count - 1) / fps if fps > 0 else 0,
                    "image_size": {
                        "width": image.width,
                        "height": image.height
                    },
                    "detections": []
                }
                
                # Detektionen hinzufügen
                for box, score, text_label in zip(boxes, scores, text_labels_result):
                    detection = {
                        "label": text_label,
                        "confidence": round(score.item(), 3),
                        "bounding_box": {
                            "xmin": round(box[0].item(), 2),
                            "ymin": round(box[1].item(), 2),
                            "xmax": round(box[2].item(), 2),
                            "ymax": round(box[3].item(), 2)
                        }
                    }
                    frame_data["detections"].append(detection)
                
                # Nur zu Ergebnissen hinzufügen wenn Detektionen gefunden wurden
                if len(boxes) > 0:
                    results_data["detections"].append(frame_data)
                    results_data["metadata"]["frames_with_detections"] += 1
                    print(f" -> {len(boxes)} Detektion(en)!")
                else:
                    print(" -> keine Detektionen")
            
            # Memory-Management: Alle 10 Frames Memory leeren (aggressiver)
            if processed_count % 10 == 0:
                clear_memory(device)
                
            # Radikale Lösung: Modell alle 100 Frames neu laden bei MPS-Problemen (nur OWLv2)
            if processed_count % 100 == 0 and device.type == 'mps' and model_type != 'yolo':
                print(f"\n -> MPS-Wartung: Lade Modell neu für optimale Performance...")
                try:
                    model.cpu()  # Modell auf CPU
                    clear_memory(device)  # GPU komplett leeren
                    model.to(device)  # Modell zurück auf GPU
                    clear_memory(device)  # Nochmals bereinigen
                    print(f" -> Modell erfolgreich neu geladen")
                except Exception as e:
                    print(f" -> Modell-Neuladung fehlgeschlagen: {e}")
                    pass
    
    except KeyboardInterrupt:
        print("\nVerarbeitung durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\nFehler bei der Video-Verarbeitung: {e}")
        cap.release()
        sys.exit(EXIT_VIDEO_PROCESSING_FAILED)
    finally:
        cap.release()
    
    return True, results_data


def detect_input_type(input_path):
    """
    Erkennt automatisch ob Input ein Video oder ein Ordner mit Frames ist
    
    Args:
        input_path (str): Pfad zum Input
    
    Returns:
        str: 'video', 'frames', oder 'unknown'
    """
    path = Path(input_path)
    
    if path.is_file():
        # Prüfe Video-Dateierweiterungen
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        if path.suffix.lower() in video_extensions:
            return 'video'
    elif path.is_dir():
        # Prüfe ob Ordner Frame-Bilder enthält
        frame_pattern = os.path.join(input_path, "frame_*.jpg")
        frame_files = glob.glob(frame_pattern)
        if frame_files:
            return 'frames'
    
    return 'unknown'


def main():
    # Argument Parser einrichten
    parser = argparse.ArgumentParser(
        description='AI-basierte Objekterkennung für Videos oder Frame-Sequenzen',
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
        help='Pfad zum Video (mp4, avi, mov, etc.) oder Ordner mit Frame-Bildern (frame_*.jpg)'
    )
    parser.add_argument(
        '--model',
        choices=['owlv2', 'yolo'],
        default='owlv2',
        help='Zu verwendendes AI-Modell: owlv2 (default) oder yolo'
    )
    parser.add_argument(
        '--yolo-model',
        default='yolo12x.pt',
        help='YOLO-Modell-Datei (default: yolo11n.pt). Kann auch yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt sein'
    )
    
    args = parser.parse_args()
    input_path = args.input_path
    model_type = args.model
    yolo_model_name = args.yolo_model
    
    # Validierung des Input-Pfades
    if not os.path.exists(input_path):
        print(f"Fehler: Der angegebene Pfad '{input_path}' existiert nicht.")
        sys.exit(EXIT_INPUT_NOT_FOUND)
    
    # Input-Typ erkennen
    input_type = detect_input_type(input_path)
    
    if input_type == 'unknown':
        print(f"Fehler: Unbekannter Input-Typ. Unterstützte Formate:")
        print("  - Videos: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v")
        print("  - Frame-Ordner: Ordner mit frame_*.jpg Dateien")
        sys.exit(EXIT_UNKNOWN_INPUT_TYPE)
    
    # Modell laden je nach gewähltem Typ
    processor = None
    model = None
    
    if model_type == 'yolo':
        if not YOLO_AVAILABLE:
            print("Fehler: YOLO wurde gewählt, aber ultralytics ist nicht installiert.")
            print("Installiere mit: pip install ultralytics")
            sys.exit(EXIT_MODEL_LOADING_FAILED)
        
        print(f"Lade YOLO-Modell: {yolo_model_name}...")
        try:
            model = YOLO(yolo_model_name)
            print(f"YOLO-Modell erfolgreich geladen")
        except Exception as e:
            print(f"Fehler beim Laden des YOLO-Modells: {e}")
            sys.exit(EXIT_MODEL_LOADING_FAILED)
    else:
        print(f"Lade OWLv2-Modell...")
        try:
            processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
            print(f"OWLv2-Modell erfolgreich geladen")
        except Exception as e:
            print(f"Fehler beim Laden des OWLv2-Modells: {e}")
            sys.exit(EXIT_MODEL_LOADING_FAILED)
    
    # GPU-Unterstützung prüfen und aktivieren
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Verwendetes Device: {device}")
    
    # Nur bei OWLv2 das Modell explizit auf Device verschieben
    if model_type != 'yolo':
        model = model.to(device)
    
    # Memory bereinigen vor dem Start
    clear_memory(device)
    print(f"Memory-Cache geleert für optimale Performance")
    
    text_labels = [["cat"]]
    
    # Start-Zeit für gesamte Verarbeitung
    start_time = datetime.now()
    
    # Je nach Input-Typ unterschiedlich verarbeiten
    if input_type == 'video':
        print(f"Erkannter Input-Typ: Video")
        print(f"Verarbeite Video direkt ohne Frame-Zwischenspeicherung...")
        
        success, results_data = process_video_directly(input_path, processor, model, device, text_labels, model_type)
        if not success:
            print("Fehler bei der Video-Verarbeitung")
            sys.exit(EXIT_VIDEO_PROCESSING_FAILED)
    else:
        print(f"Erkannter Input-Typ: Frame-Sequenz")
        frames_folder = input_path
        
        print(f"Modell geladen. Verarbeite Frames aus: {frames_folder}")

        # Alle Frame-Bilder finden und sortieren
        frame_pattern = os.path.join(frames_folder, "frame_*.jpg")
        frame_files = sorted(glob.glob(frame_pattern))

        if not frame_files:
            print(f"Fehler: Keine Frame-Bilder im Ordner {frames_folder} gefunden")
            sys.exit(EXIT_NO_FRAMES_FOUND)

        print(f"Gefunden: {len(frame_files)} Frame-Bilder")

        # Alle Frames werden verarbeitet
        frames_to_process = len(frame_files)
        print(f"Frames die verarbeitet werden: {frames_to_process}")

        # JSON-Datenstruktur für Ergebnisse
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
                "model_type": model_type,
                "processing_method": "frame_sequence"
            },
            "detections": []
        }

        # Zähler für Fortschrittsanzeige
        processed_count = 0
        local_start_time = datetime.now()

        print(f"\nStarte Verarbeitung um {local_start_time.strftime('%H:%M:%S')}...")
        print("="*60)

        # Jeden Frame verarbeiten
        for i, frame_path in enumerate(frame_files):
            try:
                # Bild direkt laden
                image = Image.open(frame_path)
                
                # Frame-Nummer aus Dateiname extrahieren
                frame_filename = os.path.basename(frame_path)
                frame_number = int(frame_filename.split('_')[1].split('.')[0])
                
                # Frame-Verarbeitung zählen
                processed_count += 1
                results_data["metadata"]["processed_frames"] = processed_count
                
                # Fortschrittsanzeige
                progress_percent = (processed_count / frames_to_process) * 100
                elapsed_time = datetime.now() - local_start_time
                
                # Geschätzte Restzeit berechnen
                if processed_count > 0:
                    avg_time_per_frame = elapsed_time.total_seconds() / processed_count
                    remaining_frames = frames_to_process - processed_count
                    estimated_remaining = remaining_frames * avg_time_per_frame
                    remaining_str = str(timedelta(seconds=int(estimated_remaining)))
                else:
                    remaining_str = "unbekannt"
                
                print(f"Progress: {progress_percent:5.1f}% ({processed_count:3d}/{frames_to_process}) | Frame {frame_number:6d} | Restzeit: {remaining_str}", end="", flush=True)
                
                # AI-Verarbeitung - unterschiedlich je nach Modell
                if model_type == 'yolo':
                    target_classes = text_labels[0] if isinstance(text_labels[0], list) else text_labels
                    success, detections, used_device = process_frame_with_fallback_yolo(
                        model, device, image, target_classes
                    )
                    
                    if not success:
                        print(f" -> Frame-Verarbeitung fehlgeschlagen, überspringe...")
                        continue
                    
                    # Frame-Daten zu JSON hinzufügen
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
                    
                    # Nur zu Ergebnissen hinzufügen wenn Detektionen gefunden wurden
                    if len(detections) > 0:
                        results_data["detections"].append(frame_data)
                        results_data["metadata"]["frames_with_detections"] += 1
                        print(f" -> {len(detections)} Detektion(en)!")
                    else:
                        print(" -> keine Detektionen")
                        
                else:  # OWLv2
                    success, outputs, used_device = process_frame_with_fallback_owlv2(
                        processor, model, device, image, text_labels
                    )
                    
                    if not success:
                        print(f" -> Frame-Verarbeitung fehlgeschlagen, überspringe...")
                        continue
                    
                    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                    target_sizes = torch.tensor([(image.height, image.width)]).to(used_device)
                    
                    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
                    results = processor.post_process_grounded_object_detection(
                        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
                    )
                    
                    # Retrieve predictions for the first image for the corresponding text queries
                    result = results[0]
                    boxes, scores, text_labels_result = result["boxes"], result["scores"], result["text_labels"]
                    
                    # Frame-Daten zu JSON hinzufügen
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
                        "detections": []
                    }
                    
                    # Detektionen hinzufügen
                    for box, score, text_label in zip(boxes, scores, text_labels_result):
                        detection = {
                            "label": text_label,
                            "confidence": round(score.item(), 3),
                            "bounding_box": {
                                "xmin": round(box[0].item(), 2),
                                "ymin": round(box[1].item(), 2),
                                "xmax": round(box[2].item(), 2),
                                "ymax": round(box[3].item(), 2)
                            }
                        }
                        frame_data["detections"].append(detection)
                    
                    # Nur zu Ergebnissen hinzufügen wenn Detektionen gefunden wurden
                    if len(boxes) > 0:
                        results_data["detections"].append(frame_data)
                        results_data["metadata"]["frames_with_detections"] += 1
                        print(f" -> {len(boxes)} Detektion(en)!")
                    else:
                        print(" -> keine Detektionen")
                
                # Memory-Management: Alle 10 Frames Memory leeren (aggressiver)
                if processed_count % 10 == 0:
                    clear_memory(device)
                    
                # Radikale Lösung: Modell alle 100 Frames neu laden bei MPS-Problemen (nur OWLv2)
                if processed_count % 100 == 0 and device.type == 'mps' and model_type != 'yolo':
                    print(f"\n -> MPS-Wartung: Lade Modell neu für optimale Performance...")
                    try:
                        model.cpu()  # Modell auf CPU
                        clear_memory(device)  # GPU komplett leeren
                        model.to(device)  # Modell zurück auf GPU
                        clear_memory(device)  # Nochmals bereinigen
                        print(f" -> Modell erfolgreich neu geladen")
                    except Exception as e:
                        print(f" -> Modell-Neuladung fehlgeschlagen: {e}")
                        pass
            
            except Exception as e:
                print(f"Fehler beim Verarbeiten von {frame_path}: {e}")
                continue

    # JSON-Datei schreiben - Name basierend auf Input, Ordner basierend auf Modell
    if input_type == 'video':
        # Für Videos: Name basierend auf Video-Dateiname
        video_name = Path(input_path).stem
        output_filename = f"{video_name}_detection_results.json"
        parent_dir = Path(input_path).parent
    else:
        # Für Frame-Ordner: Name basierend auf Ordnername
        folder_name = os.path.basename(input_path)
        output_filename = f"{folder_name}_detection_results.json"
        parent_dir = os.path.dirname(input_path)
    
    # JSON-Datei im modell-spezifischen detection_results Verzeichnis speichern
    detection_results_dir = os.path.join(parent_dir, f"detection_results-{model_type}")
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
    
    print(f"Gefundene Frames:    {total_frames_count}")
    print(f"Verarbeitete Frames: {results_data['metadata']['processed_frames']}")
    print(f"Frames m. Detektion: {results_data['metadata']['frames_with_detections']}")
    print(f"Detektionsrate:      {detection_rate:.1f}%")

    # Detektions-Details zählen
    total_detections = sum(len(frame['detections']) for frame in results_data['detections'])
    cat_detections = sum(len([d for d in frame['detections'] if 'cat' in d['label']]) for frame in results_data['detections'])

    print(f"Gesamt Detektionen:  {total_detections}")
    print(f"  - Katzen:          {cat_detections}")
    print("="*60)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Ergebnisse gespeichert in: {output_path}")
        
    except Exception as e:
        print(f"Fehler beim Schreiben der JSON-Datei: {e}")
        # Fallback: Ergebnisse in der Konsole ausgeben
        print("\nFallback - Ergebnisse in der Konsole:")
        print(json.dumps(results_data, indent=2, ensure_ascii=False))
        sys.exit(EXIT_JSON_WRITE_FAILED)
    


if __name__ == "__main__":
    try:
        main()
        sys.exit(EXIT_SUCCESS)
    except KeyboardInterrupt:
        print("\nVerarbeitung durch Benutzer abgebrochen")
        sys.exit(EXIT_SUCCESS)  # Benutzer-Abbruch ist kein Fehler
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
        sys.exit(EXIT_CRITICAL_ERROR)