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

#### Modell und Setup
- **Verwendetes Modell**: `google/owlv2-base-patch16-ensemble`
- **Framework**: Hugging Face Transformers mit PyTorch
- **Erkennungskategorien**: Katzen und Hunde
- **Verarbeitungsoptimierung**: Analyse nur jeden 30. Frames für bessere Performance

#### Erkennungsprozess
1. **Frame-Laden**: Sequenzielle Verarbeitung der extrahierten Bildsequenz
2. **KI-Analyse**: Object Detection mit konfigurierbarem Schwellenwert (0.1)
3. **Bounding Box Extraktion**: Präzise Lokalisierung erkannter Objekte
4. **Confidence-Bewertung**: Vertrauensscore für jede Erkennung

#### Ausgabe und Dokumentation
- **JSON-Export**: Strukturierte Speicherung aller Erkennungsergebnisse
- **Metadaten**: Verarbeitungsstatistiken und Konfigurationsparameter
- **Detaillierte Logs**: Fortschrittsanzeige mit Zeitschätzungen
- **Performance-Metriken**: Detektionsrate und Verarbeitungsgeschwindigkeit

### 3. Datenstruktur der Ergebnisse

Die generierten JSON-Dateien enthalten:
- **Metadaten**: Timestamp, Verarbeitungsstatistiken, Konfiguration
- **Frame-Informationen**: Bildgröße, Dateiname, Pfad
- **Erkennungsdetails**: Objekt-Label, Confidence-Score, Bounding Box Koordinaten

### 4. Systemanforderungen

```
transformers>=4.30.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
scipy
opencv-python>=4.8.0
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
  - OWLv2-Modell analysiert jeden 30. Frame
  - Erzeugung detaillierter JSON-Erkennungsberichte

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
3. **KI-Analyse**: Anpassung der Pfade in `ai-processor.py` und Ausführung der Objekterkennung auf leistungsstarker Hardware
4. **Ergebnis-Aggregation**: Sammlung und Zusammenführung aller JSON-Ausgabedateien

Das System ist für große Videodatenmengen optimiert und bietet eine vollständig automatisierte Pipeline von der manuell vorselektierten Video-Eingabe bis zur strukturierten Erkennungsausgabe.
