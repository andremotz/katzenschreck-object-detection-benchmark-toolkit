# Hugging Face Katzendetector-Playground

Dieses Projekt verwendet das Hugging Face OWLv2 (Open-World Localization) Modell zur automatischen Erkennung von Katzen und Hunden in Videoaufnahmen. Das System analysiert Videomaterial durch Bildsequenz-Extraktion und KI-basierte Objekterkennung.

## Allgemeines Vorgehen

### 1. Video-zu-Bildsequenz Konvertierung
Das Projekt nutzt einen zweistufigen Ansatz zur effizienten Verarbeitung von Videodateien:

#### Einzelne Video-Konvertierung (`convert_video_to_image_sequence.py`)
- Konvertiert einzelne MP4-Videos in Bildsequenzen (JPG-Dateien)
- Extrahiert Frames mit konfigurierbaren Parametern (Qualität, Frame-Skip)
- Speichert Bilder in einem `_frames` Unterordner neben der Videodatei
- Fortschrittsanzeige und Performance-Optimierung

#### Batch-Verarbeitung (`convert_video_batch.py`)
- Scannt rekursiv ein Verzeichnis nach allen Videodateien
- Prüft automatisch, ob bereits `_frames` Ordner existieren
- Konvertiert automatisch alle Videos ohne entsprechende Bildsequenzen
- Übersicht und Zusammenfassung aller verarbeiteten Dateien

### 2. KI-basierte Objekterkennung (`ai-processor.py`)

#### Verfügbare AI-Modelle
Das Script unterstützt zwei verschiedene Objekterkennungsmodelle:

**OWLv2 (Standard)**
- **Modell**: `google/owlv2-base-patch16-ensemble`
- **Framework**: Hugging Face Transformers mit PyTorch
- **Erkennungskategorien**: Nur Katzen (text-basierte Erkennung)
- **Vorteile**: Flexiblere Klassendefinition, bessere Text-basierte Objekterkennung
- **Nachteile**: Langsamere Verarbeitung

**YOLO v11/v12 (Optional)**
- **Modelle**: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- **Framework**: Ultralytics YOLO
- **Erkennungskategorien**: Nur Katzen (aus 80 COCO-Klassen)
- **Vorteile**: Deutlich schnellere Verarbeitung, bessere Performance
- **Nachteile**: Feste Klassendefinition

#### Modell-Auswahl
```bash
# Standard OWLv2 verwenden
python ai-processor.py /pfad/zum/video.mp4

# YOLO verwenden (schneller)
python ai-processor.py --model yolo /pfad/zum/video.mp4

# Spezifisches YOLO-Modell verwenden
python ai-processor.py --model yolo --yolo-model yolo11l.pt /pfad/zum/video.mp4
```

#### YOLO-Modell Übersicht
| Modell | Größe | Geschwindigkeit | Genauigkeit | Verwendung |
|--------|-------|----------------|-------------|------------|
| `yolo11n.pt` | Nano | Sehr schnell | Basis | Schnelle Tests |
| `yolo11s.pt` | Small | Schnell | Gut | Produktive Nutzung |
| `yolo11m.pt` | Medium | Mittel | Sehr gut | Ausgewogene Lösung |
| `yolo11l.pt` | Large | Langsam | Ausgezeichnet | Hohe Genauigkeit |
| `yolo11x.pt` | Extra Large | Sehr langsam | Beste | Maximale Präzision |

#### Erkennungsprozess
1. **Frame-Laden**: Sequenzielle Verarbeitung der extrahierten Bildsequenz
2. **KI-Analyse**: Object Detection mit konfigurierbarem Schwellenwert (0.1)
3. **Bounding Box Extraktion**: Präzise Lokalisierung erkannter Objekte
4. **Confidence-Bewertung**: Vertrauensscore für jede Erkennung

#### Ausgabe und Dokumentation
- **JSON-Export**: Strukturierte Speicherung aller Erkennungsergebnisse
- **Metadaten**: Verarbeitungsstatistiken und Konfigurationsparameter (inklusive verwendetes Modell)
- **Detaillierte Logs**: Fortschrittsanzeige mit Zeitschätzungen
- **Performance-Metriken**: Detektionsrate und Verarbeitungsgeschwindigkeit
- **Modell-spezifische Ordner**: Getrennte Speicherung je nach verwendetem AI-Modell

#### Ordnerstruktur der Ergebnisse
```
/pfad/zum/video/
├── video.mp4
├── detection_results-owlv2/     # OWLv2 Ergebnisse
│   └── video_detection_results.json
└── detection_results-yolo/      # YOLO Ergebnisse
    └── video_detection_results.json
```

Dies ermöglicht parallele Verarbeitung mit beiden Modellen und direkten Vergleich der Ergebnisse.

### 3. Datenstruktur der Ergebnisse

Die generierten JSON-Dateien enthalten:
- **Metadaten**: Timestamp, Verarbeitungsstatistiken, Konfiguration
- **Frame-Informationen**: Bildgröße, Dateiname, Pfad
- **Erkennungsdetails**: Objekt-Label, Confidence-Score, Bounding Box Koordinaten

### 4. Systemanforderungen

#### Basis-Requirements (für OWLv2)
```
transformers>=4.30.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
scipy
opencv-python>=4.8.0
```

#### Zusätzliche Requirements für YOLO-Unterstützung
```
ultralytics>=8.0.0
numpy>=1.21.0
```

**Installation mit YOLO-Unterstützung:**
```bash
pip install -r requirements.txt
```

## Installation

Um die Scripts auf einem neuen Rechner zu installieren, folgen Sie diesen Schritten:

### Voraussetzungen
- Python 3.8+ installiert
- Git installiert
- Zugang zum Repository

### Schritt-für-Schritt Installation

1. **Repository klonen**
   ```bash
   git clone <repository-url>
   cd huggingface-catdetector-playground
   ```

2. **Virtuelle Umgebung erstellen**
   ```bash
   python -m venv venv
   ```

3. **Virtuelle Umgebung aktivieren**
   
   **Auf Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```
   
   **Auf Windows:**
   ```bash
   venv\Scripts\activate
   ```

4. **Dependencies installieren**
   ```bash
   pip install -r requirements.txt
   ```

5. **Installation überprüfen**
   ```bash
   python -c "import torch; import transformers; print('Installation erfolgreich!')"
   ```

### Zusätzliche Hinweise für GPU-Unterstützung

Für optimale Performance auf dem starken Rechner (Schritt 5 des Workflows):

```bash
# Für CUDA-fähige GPUs (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Installation überprüfen
python -c "import torch; print(f'CUDA verfügbar: {torch.cuda.is_available()}')"
```

### Pfad-Anpassungen

Nach der Installation müssen die Pfade in den Scripts an Ihre lokale Umgebung angepasst werden:

- **`convert_video_batch.py`**: Zeile 103 - Pfad zum Camera_Teich-Footage Ordner
- **`ai-processor.py`**: Zeile 14 - Pfad zum frames_folder

### Updates abrufen

Um das Projekt auf den neuesten Stand zu bringen:

```bash
git pull origin main
pip install -r requirements.txt  # Falls neue Dependencies hinzugefügt wurden
```

## Funktionales Vorgehen

Der komplette Workflow des Katzendetector-Systems folgt einem strukturierten 6-Stufen-Prozess:

### 1. Video-Download
- **Aktion**: Benutzer lädt Videomaterial (z.B. von Kameras) herunter
- **Format**: Typischerweise MP4-Dateien
- **Speicherort**: Lokaler Rechner für erste Sichtung

### 2. Manuelle Video-Sichtung
- **Aktion**: Benutzer sichtet das heruntergeladene Video manuell
- **Entscheidung**: 
  - ✅ **Katzen vorhanden** → Video behalten für weitere Verarbeitung
  - ❌ **Keine Katzen** → Video löschen (spart Speicher und Verarbeitungszeit)
- **Zweck**: Effiziente Vorselektion zur Reduzierung der Datenmenge

### 3. Video-Transfer auf Fileshare
- **Aktion**: Relevante Videos (mit Katzen) auf zentralen Fileshare übertragen
- **Ziel**: Zentrale Verfügbarkeit für die weitere automatisierte Verarbeitung
- **Organisation**: Strukturierte Ablage nach Datum/Kamera/etc.

### 4. Automatische Bildsequenz-Extraktion
- **Tool**: `convert_video_batch.py`
- **Ausführungsort**: Fileshare-Server
- **Prozess**: 
  - Rekursives Scannen aller Videos im Fileshare
  - Automatische Konvertierung in Bildsequenzen (`_frames` Ordner)
  - Vermeidung doppelter Verarbeitung bereits konvertierter Videos

### 5. KI-basierte Objekterkennung
- **Tool**: `ai-processor.py`
- **Ausführungsort**: Starker Rechner mit GPU-Unterstützung
- **Prozess**:
  - Verarbeitung der generierten Bildsequenzen
  - Wahlweise OWLv2 oder YOLO für Objekterkennung
  - Erzeugung detaillierter JSON-Erkennungsberichte in modell-spezifischen Ordnern

#### Performance-Empfehlungen:
- **Große Datenmengen**: YOLO verwenden (`--model yolo`) für deutlich schnellere Verarbeitung
- **Höchste Genauigkeit**: OWLv2 für präzisere text-basierte Katzenerkennung
- **GPU-Nutzung**: Beide Modelle profitieren von CUDA/MPS-Unterstützung
- **Modell-Vergleich**: Parallele Verarbeitung mit beiden Modellen für Qualitätsvergleich

### 6. Zentrale Zusammenführung der Ergebnisse
- **Aktion**: Alle generierten JSON-Files werden an zentraler Stelle gesammelt
- **Zweck**: 
  - Übergreifende Analyse und Statistiken
  - Zentrale Datenhaltung für weitere Auswertungen
  - Langzeit-Monitoring und Trends

## Technischer Workflow

Für die technische Umsetzung:

1. **Video-Vorbereitung**: Platzierung der selektierten MP4-Dateien im Fileshare-Verzeichnis
2. **Batch-Konvertierung**: Ausführung von `convert_video_batch.py` für automatische Bildsequenz-Extraktion
3. **KI-Analyse**: Ausführung der Objekterkennung auf leistungsstarker Hardware
   ```bash
   # Schnelle Verarbeitung mit YOLO
   python ai-processor.py --model yolo /pfad/zu/video_frames/
   
   # Präzise Verarbeitung mit OWLv2
   python ai-processor.py --model owlv2 /pfad/zu/video_frames/
   
   # Beide Modelle für Vergleich
   python ai-processor.py --model owlv2 /pfad/zu/video_frames/
   python ai-processor.py --model yolo /pfad/zu/video_frames/
   ```
4. **Ergebnis-Aggregation**: Sammlung und Zusammenführung aller JSON-Ausgabedateien aus modell-spezifischen Ordnern

Das System ist für große Videodatenmengen optimiert und bietet eine vollständig automatisierte Pipeline von der manuell vorselektierten Video-Eingabe bis zur strukturierten Erkennungsausgabe. Die YOLO-Integration ermöglicht deutlich schnellere Verarbeitung großer Datenmengen bei weiterhin hoher Erkennungsqualität.

## Detektions-Annotation Script

### Übersicht

Das `annotate_detections.py` Script liest JSON-Detektionsergebnisse von `ai-processor.py` ein und zeichnet Bounding-Boxes auf die entsprechenden Frame-Bilder.

### Funktionen

#### Hauptfunktionen:
- ✅ **JSON-Parser**: Lädt Detektionsergebnisse aus JSON-Dateien
- ✅ **Bounding-Box Zeichnung**: Zeichnet farbige Bounding-Boxes um erkannte Objekte
- ✅ **Label-Anzeige**: Zeigt Label und Confidence-Score für jede Detektion
- ✅ **Batch-Verarbeitung**: Verarbeitet alle detektierten Frames automatisch
- ✅ **Fortschrittsanzeige**: Zeigt Verarbeitungsfortschritt in Echtzeit
- ✅ **Übersichtsbild**: Erstellt automatisch ein Grid der besten Detektionen

#### Farbkodierung:
- 🟠 **Orange**: Katzen (`cat`)
- 🟢 **Grün**: Hunde (`dog`)
- 🔴 **Rot**: Personen (`person`)
- 🔵 **Blau**: Vögel (`bird`)
- 🟡 **Gelb**: Andere/Unbekannte Labels

### Verwendung

#### Grundlegende Verwendung:
```bash
python annotate_detections.py detection_results/Camera_Teich-20250907-155316-1757253196613-7_frames_detection_results.json
```

#### Mit benutzerdefiniertem Ausgabeordner:
```bash
python annotate_detections.py detection_results.json --output-dir meine_annotationen
```

#### Mit OpenCV-Rendering (Alternative zu PIL):
```bash
python annotate_detections.py detection_results.json --opencv
```

#### Ohne Übersichtsbild:
```bash
python annotate_detections.py detection_results.json --no-summary
```

### Parameter

| Parameter | Beschreibung | Standard |
|-----------|--------------|----------|
| `json_file` | Pfad zur JSON-Datei mit Detektionsergebnissen | (erforderlich) |
| `--output-dir`, `-o` | Ausgabeordner für annotierte Bilder | `annotated_<json_basename>` |
| `--opencv` | Verwendet OpenCV statt PIL für Rendering | PIL |
| `--no-summary` | Erstellt kein Übersichtsbild | Übersichtsbild erstellen |

### Ausgabe

Das Script erstellt:

1. **Annotierte Einzelbilder**: Jeder Frame mit Detektionen wird mit Bounding-Boxes versehen
   - Format: `annotated_frame_XXXXXX.jpg`
   - Enthält alle Detektionen mit Label und Confidence

2. **Übersichtsbild**: `detection_summary.jpg`
   - Grid-Layout der besten Detektionen (höchste Confidence)
   - Bis zu 9 Bilder in 3x3 Anordnung
   - Thumbnail-Größe für schnelle Übersicht

### Beispiel-Workflow

1. **Video analysieren** (mit ai-processor.py):
   ```bash
   python ai-processor.py mein_video.mp4
   # Erstellt: detection_results/mein_video_detection_results.json
   ```

2. **Detektionen annotieren**:
   ```bash
   python annotate_detections.py detection_results/mein_video_detection_results.json
   # Erstellt: annotated_mein_video_detection_results/
   ```

3. **Ergebnisse betrachten**:
   - Einzelne Frames: `annotated_mein_video_detection_results/annotated_frame_*.jpg`
   - Übersicht: `annotated_mein_video_detection_results/detection_summary.jpg`

### Technische Details

#### Unterstützte Formate:
- **Input**: JSON-Dateien von ai-processor.py
- **Output**: JPEG-Bilder mit 95% Qualität
- **Frame-Formate**: Alle von PIL/OpenCV unterstützten Formate

#### Performance:
- **PIL-Modus**: Bessere Qualität, etwas langsamer
- **OpenCV-Modus**: Schneller, für große Mengen geeignet
- **Speicher**: Effizient, lädt nur jeweils ein Bild

#### Fehlerbehandlung:
- Fehlende Frame-Bilder werden übersprungen
- Korrupte JSON-Dateien werden erkannt
- Detaillierte Fehlermeldungen für Debugging

### Anpassungen

Das Script kann leicht angepasst werden:

- **Farben ändern**: Bearbeiten Sie die `get_label_color()` Funktion
- **Text-Stil**: Anpassen von `font_size`, `box_thickness` etc.
- **Neue Labels**: Erweitern Sie die `color_map` in `get_label_color()`
- **Output-Format**: Ändern Sie `quality` Parameter beim Speichern

### Fehlerbehebung

#### Häufige Probleme:

1. **"Frame-Bild nicht gefunden"**:
   - Frame-Pfade in JSON stimmen nicht mit tatsächlichen Dateien überein
   - Frame-Bilder wurden verschoben oder gelöscht

2. **"Fehler beim Parsen der JSON-Datei"**:
   - JSON-Datei ist beschädigt oder unvollständig
   - Verwenden Sie einen JSON-Validator

3. **Schrift-Probleme**:
   - Script verwendet automatisch verfügbare System-Schriften
   - Falls nötig, installieren Sie zusätzliche Schriftarten

#### Debug-Modus:
Fügen Sie `print()` Statements hinzu, um den Verarbeitungsfortschritt zu verfolgen.

### Ergebnis-Beispiel

Nach der Ausführung erhalten Sie:
```
annotated_Camera_Teich-20250907-155316-1757253196613-7_frames_detection_results/
├── annotated_frame_000390.jpg    # Hund mit grüner Bounding-Box
├── annotated_frame_004710.jpg    # 3 Hunde-Detektionen
├── annotated_frame_004740.jpg    # 4 Katzen-Detektionen (orange)
├── ...
└── detection_summary.jpg         # Übersicht der besten Detektionen
```

Jedes Bild zeigt die ursprünglichen Frames mit:
- Farbige Bounding-Boxes um erkannte Objekte
- Label und Confidence-Score über jeder Box
- Optimierte Lesbarkeit durch Hintergrund-Boxen
