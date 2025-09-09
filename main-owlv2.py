import os
import glob
import json
from datetime import datetime, timedelta
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Bildsequenz-Ordner
frames_folder = "/Users/andremotz/Downloads/Camera_Teich-Footage/20250907PM/Camera_Teich-20250907-155316-1757253196613-7_frames"

# Alle Frame-Bilder finden und sortieren
frame_pattern = os.path.join(frames_folder, "frame_*.jpg")
frame_files = sorted(glob.glob(frame_pattern))

if not frame_files:
    print(f"Fehler: Keine Frame-Bilder im Ordner {frames_folder} gefunden")
    exit()

print(f"Gefunden: {len(frame_files)} Frame-Bilder")

text_labels = [["cat", "dog"]]

# Berechne wie viele Frames tatsächlich verarbeitet werden (jeden 30.)
frames_to_process = len([i for i in range(len(frame_files)) if i % 30 == 0])
print(f"Frames die verarbeitet werden (jeden 30.): {frames_to_process}")

# JSON-Datenstruktur für Ergebnisse
results_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "frames_folder": frames_folder,
        "total_frames": len(frame_files),
        "frames_to_process": frames_to_process,
        "processed_frames": 0,
        "frames_with_detections": 0,
        "detection_threshold": 0.1,
        "text_labels": text_labels[0]
    },
    "detections": []
}

# Zähler für Fortschrittsanzeige
processed_count = 0
start_time = datetime.now()

print(f"\nStarte Verarbeitung um {start_time.strftime('%H:%M:%S')}...")
print("="*60)

# Jeden Frame verarbeiten
for i, frame_path in enumerate(frame_files):
    # Nur jeden 30. Frame verarbeiten (um Performance zu verbessern)
    if i % 30 != 0:
        continue
    
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
        elapsed_time = datetime.now() - start_time
        
        # Geschätzte Restzeit berechnen
        if processed_count > 0:
            avg_time_per_frame = elapsed_time.total_seconds() / processed_count
            remaining_frames = frames_to_process - processed_count
            estimated_remaining = remaining_frames * avg_time_per_frame
            remaining_str = str(timedelta(seconds=int(estimated_remaining)))
        else:
            remaining_str = "unbekannt"
        
        print(f"Progress: {progress_percent:5.1f}% ({processed_count:3d}/{frames_to_process}) | Frame {frame_number:6d} | Restzeit: {remaining_str}", end="", flush=True)
        
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([(image.height, image.width)])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
        )
        
        # Retrieve predictions for the first image for the corresponding text queries
        result = results[0]
        boxes, scores, text_labels_result = result["boxes"], result["scores"], result["text_labels"]
        
        # Frame-Daten zu JSON hinzufügen
        frame_data = {
            "frame_number": frame_number,
            "frame_filename": frame_filename,
            "frame_path": frame_path,
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
    
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {frame_path}: {e}")
        continue

# JSON-Datei schreiben - Name basierend auf Ordnername
folder_name = os.path.basename(frames_folder)
output_filename = f"{folder_name}_detection_results.json"
output_path = os.path.join(os.getcwd(), output_filename)

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
print(f"Gefundene Frames:    {results_data['metadata']['total_frames']}")
print(f"Verarbeitete Frames: {results_data['metadata']['processed_frames']}")
print(f"Frames m. Detektion: {results_data['metadata']['frames_with_detections']}")
print(f"Detektionsrate:      {detection_rate:.1f}%")

# Detektions-Details zählen
total_detections = sum(len(frame['detections']) for frame in results_data['detections'])
cat_detections = sum(len([d for d in frame['detections'] if 'cat' in d['label']]) for frame in results_data['detections'])
dog_detections = sum(len([d for d in frame['detections'] if 'dog' in d['label']]) for frame in results_data['detections'])

print(f"Gesamt Detektionen:  {total_detections}")
print(f"  - Katzen:          {cat_detections}")
print(f"  - Hunde:           {dog_detections}")
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