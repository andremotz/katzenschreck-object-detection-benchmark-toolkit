#!/usr/bin/env python3
"""
Hauptverarbeitungs-Pipeline für Video-zu-AI-Objekterkennung
Kombiniert convert_video_to_image_sequence.py und ai-processor.py zu einer einzigen Pipeline
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(command, description):
    """
    Führt einen Systembefehl aus und gibt detaillierte Rückmeldungen
    
    Args:
        command (list): Der auszuführende Befehl als Liste
        description (str): Beschreibung des Befehls für die Ausgabe
    
    Returns:
        tuple: (success, stdout, stderr)
    """
    print(f"\n🔄 {description}")
    print(f"Befehl: {' '.join(command)}")
    print("=" * 60)
    
    try:
        # subprocess.Popen für Live-Ausgabe mit gleichzeitigem Sammeln der Ausgabe
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_lines = []
        
        # Live-Ausgabe zeigen und gleichzeitig sammeln
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                stdout_lines.append(output)
        
        # Warten bis Prozess beendet ist
        return_code = process.poll()
        stdout = ''.join(stdout_lines)
        
        if return_code == 0:
            print(f"\n✅ {description} erfolgreich abgeschlossen")
            return True, stdout, ""
        else:
            print(f"\n❌ {description} fehlgeschlagen (Exit Code: {return_code})")
            return False, stdout, ""
            
    except Exception as e:
        print(f"❌ Fehler beim Ausführen von '{description}': {e}")
        return False, "", str(e)


def validate_video_file(video_path):
    """
    Validiert, ob die Video-Datei existiert und lesbar ist
    
    Args:
        video_path (str): Pfad zur Video-Datei
    
    Returns:
        bool: True wenn gültig, False sonst
    """
    if not os.path.exists(video_path):
        print(f"❌ Fehler: Video-Datei nicht gefunden: {video_path}")
        return False
    
    if not os.path.isfile(video_path):
        print(f"❌ Fehler: '{video_path}' ist keine Datei")
        return False
    
    # Prüfe auf gängige Video-Endungen
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wmv', '.flv']
    file_extension = Path(video_path).suffix.lower()
    
    if file_extension not in video_extensions:
        print(f"⚠️  Warnung: '{file_extension}' ist möglicherweise kein unterstütztes Video-Format")
        print(f"   Unterstützte Formate: {', '.join(video_extensions)}")
    
    return True


def main():
    """Hauptfunktion der Verarbeitungs-Pipeline"""
    
    # Argument Parser konfigurieren
    parser = argparse.ArgumentParser(
        description='Komplette Video-zu-AI-Objekterkennung Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Beispiele:
  python main_processing_pipeline.py /pfad/zum/video.mp4
  python main_processing_pipeline.py /pfad/zum/video.mp4 --quality 90
  python main_processing_pipeline.py video.mp4 --output-dir /custom/output
        '''
    )
    
    parser.add_argument(
        'video_path',
        help='Pfad zur Video-Datei, die verarbeitet werden soll'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output-Verzeichnis für Frames (optional, wird automatisch erstellt)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG-Qualität für Frames (1-100, Standard: 95)'
    )
    
    
    parser.add_argument(
        '--keep-frames',
        action='store_true',
        help='Frame-Bilder nach der AI-Verarbeitung behalten (Standard: löschen)'
    )
    
    args = parser.parse_args()
    
    # Pipeline starten
    start_time = datetime.now()
    print("🚀 Video-zu-AI-Objekterkennung Pipeline gestartet")
    print(f"⏰ Startzeit: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Video-Datei validieren
    print(f"📹 Input-Video: {args.video_path}")
    if not validate_video_file(args.video_path):
        sys.exit(1)
    
    # Pfade für Skripte bestimmen
    script_dir = Path(__file__).parent
    convert_script = script_dir / "convert_video_to_image_sequence.py"
    ai_script = script_dir / "ai-processor.py"
    
    # Prüfen ob benötigte Skripte existieren
    if not convert_script.exists():
        print(f"❌ Fehler: convert_video_to_image_sequence.py nicht gefunden in {script_dir}")
        sys.exit(1)
    
    if not ai_script.exists():
        print(f"❌ Fehler: ai-processor.py nicht gefunden in {script_dir}")
        sys.exit(1)
    
    print(f"✅ Alle benötigten Skripte gefunden")
    
    # SCHRITT 1: Video zu Bildsequenz konvertieren
    print(f"\n📋 SCHRITT 1: Video zu Bildsequenz konvertieren")
    
    convert_command = [
        sys.executable,
        str(convert_script),
        args.video_path
    ]
    
    # Parameter für convert_video_to_image_sequence.py hinzufügen
    # Reihenfolge: video_path, output_dir, quality, frame_skip
    if args.output_dir:
        # Wenn Output-Dir angegeben, alle Parameter setzen
        convert_command.extend([args.output_dir, str(args.quality), "1"])
    else:
        # Kein Output-Dir angegeben - das Skript erstellt automatisch videoname_frames
        # Wir müssen trotzdem die Parameter in der richtigen Reihenfolge übergeben
        # Also: video_path (bereits gesetzt), output_dir (leer lassen), quality, frame_skip
        # Da wir output_dir weglassen wollen, übergeben wir nur quality und frame_skip wenn nötig
        if args.quality != 95:
            convert_command.extend([str(args.quality), "1"])
        # Wenn Standard-Quality: keine weiteren Parameter nötig
    
    success, stdout, stderr = run_command(convert_command, "Video-zu-Frames Konvertierung")
    
    if not success:
        print(f"❌ Pipeline abgebrochen: Video-Konvertierung fehlgeschlagen")
        sys.exit(1)
    
    # Output-Verzeichnis aus der Konvertierung ermitteln
    # Das convert_video_to_image_sequence.py gibt das Output-Verzeichnis aus
    frames_dir = None
    for line in stdout.split('\n'):
        if 'Output-Verzeichnis:' in line:
            frames_dir = line.split('Output-Verzeichnis:')[1].strip()
            break
        elif 'Bilder gespeichert in:' in line:
            frames_dir = line.split('Bilder gespeichert in:')[1].strip()
            break
    
    if not frames_dir or not os.path.exists(frames_dir):
        print(f"❌ Fehler: Konnte Frame-Verzeichnis nicht ermitteln oder es existiert nicht")
        print(f"   Ermitteltes Verzeichnis: {frames_dir}")
        sys.exit(1)
    
    print(f"✅ Frames erfolgreich erstellt in: {frames_dir}")
    
    # SCHRITT 2: AI-Objekterkennung ausführen
    print(f"\n🤖 SCHRITT 2: AI-Objekterkennung")
    
    ai_command = [
        sys.executable,
        str(ai_script),
        frames_dir
    ]
    
    success, stdout, stderr = run_command(ai_command, "AI-Objekterkennung")
    
    if not success:
        print(f"❌ Pipeline abgebrochen: AI-Verarbeitung fehlgeschlagen")
        if not args.keep_frames:
            print(f"🗑️  Frame-Verzeichnis wird beibehalten für Debugging: {frames_dir}")
        sys.exit(1)
    
    # Erfolgreiche Verarbeitung
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 80)
    print(f"⏰ Startzeit:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Endzeit:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Gesamtdauer: {total_time}")
    print(f"📹 Input-Video: {args.video_path}")
    print(f"📁 Frame-Verzeichnis: {frames_dir}")
    
    # Ergebnis-JSON ermitteln
    frames_folder_name = os.path.basename(frames_dir)
    result_json = os.path.join(os.path.dirname(frames_dir), f"{frames_folder_name}_detection_results.json")
    
    if os.path.exists(result_json):
        print(f"📊 Ergebnisse: {result_json}")
    
    # Aufräumen der Frame-Bilder (optional)
    if not args.keep_frames:
        print(f"\n🗑️  Räume Frame-Verzeichnis auf...")
        try:
            import shutil
            shutil.rmtree(frames_dir)
            print(f"✅ Frame-Verzeichnis gelöscht: {frames_dir}")
        except Exception as e:
            print(f"⚠️  Warnung: Konnte Frame-Verzeichnis nicht löschen: {e}")
            print(f"   Manuell löschen: {frames_dir}")
    else:
        print(f"📁 Frame-Verzeichnis beibehalten: {frames_dir}")
    
    print("=" * 80)
    print("🏁 Pipeline beendet")


if __name__ == "__main__":
    main()
