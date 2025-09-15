#!/usr/bin/env python3
"""
Script zum Überprüfen von JSON-Dateien und parallelen 'annotated_' Ordnern.

Dieses Script durchsucht einen gegebenen Ordner nach JSON-Dateien und prüft,
ob es parallel zu jeder JSON-Datei einen Ordner mit dem Prefix 'annotated_'
und dem gleichen Namen (ohne .json Extension) gibt.

Für JSON-Dateien ohne annotated-Ordner wird automatisch annotate_detections.py
aufgerufen, um die fehlenden Annotationen zu erstellen.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json


def find_json_files(directory):
    """
    Findet alle JSON-Dateien in einem Verzeichnis rekursiv (inklusive Subordner).
    
    Args:
        directory (str): Pfad zum Verzeichnis
        
    Returns:
        list: Liste von JSON-Dateipfaden
    """
    json_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Fehler: Verzeichnis '{directory}' existiert nicht.")
        return json_files
    
    if not directory_path.is_dir():
        print(f"Fehler: '{directory}' ist kein Verzeichnis.")
        return json_files
    
    # Alle .json Dateien rekursiv im Verzeichnis und Subordnern finden
    for file_path in directory_path.rglob('*.json'):
        if file_path.is_file():
            json_files.append(file_path)
    
    return json_files


def check_parallel_folders(json_files):
    """
    Überprüft für jede JSON-Datei, ob ein paralleler Ordner mit "annotated_" Prefix existiert.
    
    Args:
        json_files (list): Liste von JSON-Dateipfaden
        
    Returns:
        dict: Dictionary mit Ergebnissen der Überprüfung
    """
    results = {
        'with_folder': [],
        'without_folder': [],
        'total_json_files': len(json_files)
    }
    
    for json_file in json_files:
        # Name ohne .json Extension
        base_name = json_file.stem
        
        # Paralleler Ordnerpfad mit "annotated_" Prefix
        folder_name = f"annotated_{base_name}"
        expected_folder = json_file.parent / folder_name
        
        # Überprüfen ob der Ordner existiert
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
    Führt annotate_detections.py für eine JSON-Datei aus.
    
    Args:
        json_file_path (Path): Pfad zur JSON-Datei
        
    Returns:
        bool: True wenn erfolgreich, False bei Fehler
    """
    try:
        # Pfad zum annotate_detections.py Script (im gleichen Verzeichnis)
        script_dir = Path(__file__).parent
        annotate_script = script_dir / "annotate_detections.py"
        
        if not annotate_script.exists():
            print(f"Fehler: annotate_detections.py nicht gefunden in {script_dir}")
            return False
        
        print(f"\n  → Führe annotate_detections.py aus für: {json_file_path.name}")
        
        # Führe das Script aus
        result = subprocess.run(
            [sys.executable, str(annotate_script), str(json_file_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 Minuten Timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ Annotation erfolgreich erstellt")
            return True
        else:
            print(f"  ✗ Annotation fehlgeschlagen (Exit Code: {result.returncode})")
            
            # Erkläre den Exit-Code
            exit_code_meanings = {
                1: "JSON-Datei nicht gefunden",
                2: "JSON-Datei konnte nicht geladen werden", 
                3: "Ausgabeordner konnte nicht erstellt werden",
                4: "Keine Frames konnten annotiert werden (Frame-Bilder nicht gefunden)",
                5: "Einige Frame-Annotationen fehlgeschlagen",
                6: "Frame-Annotation und Video-Erstellung teilweise fehlgeschlagen",
                7: "Nur Video-Erstellung fehlgeschlagen"
            }
            
            if result.returncode in exit_code_meanings:
                print(f"    Bedeutung: {exit_code_meanings[result.returncode]}")
            
            if result.stderr:
                print(f"    Stderr: {result.stderr.strip()}")
            if result.stdout:
                print(f"    Stdout: {result.stdout.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: annotate_detections.py brauchte zu lange")
        return False
    except Exception as e:
        print(f"  ✗ Unerwarteter Fehler: {e}")
        return False


def process_missing_annotations(results, verbose=False):
    """
    Verarbeitet JSON-Dateien ohne annotated-Ordner durch Aufruf von annotate_detections.py.
    
    Args:
        results (dict): Ergebnisse der Überprüfung
        verbose (bool): Ausführliche Ausgabe
        
    Returns:
        bool: True wenn alle Annotationen erfolgreich, False bei Fehlern
    """
    if not results['without_folder']:
        return True
    
    print(f"\n=== AUTOMATISCHE ANNOTATION ===\n")
    print(f"Erstelle fehlende Annotationen für {len(results['without_folder'])} JSON-Dateien...\n")
    
    success_count = 0
    failed_files = []
    
    for i, item in enumerate(results['without_folder'], 1):
        json_file_path = Path(item['json_file'])
        
        print(f"[{i}/{len(results['without_folder'])}] Verarbeite: {json_file_path.name}")
        
        success = run_annotation_script(json_file_path)
        
        if success:
            success_count += 1
        else:
            failed_files.append(json_file_path)
            # Bei Fehler: Script abbrechen
            print(f"\n❌ FEHLER: Annotation für {json_file_path.name} fehlgeschlagen.")
            print(f"Script wird abgebrochen.\n")
            return False
    
    print(f"\n=== ANNOTATION ABGESCHLOSSEN ===\n")
    print(f"Erfolgreich: {success_count}/{len(results['without_folder'])} Annotationen erstellt")
    
    return True


def print_results(results):
    """
    Gibt die Ergebnisse formatiert aus.
    
    Args:
        results (dict): Ergebnisse der Überprüfung
    """
    print(f"\n=== ERGEBNISSE ===")
    print(f"Insgesamt {results['total_json_files']} JSON-Dateien gefunden (inklusive Subordner).\n")
    
    print(f"JSON-Dateien MIT parallelem 'annotated_' Ordner: {len(results['with_folder'])}")
    for item in results['with_folder']:
        json_path = Path(item['json_file'])
        folder_path = Path(item['folder'])
        print(f"  ✓ {json_path.relative_to(Path.cwd()) if json_path.is_relative_to(Path.cwd()) else json_path} → {folder_path.name}/")
    
    print(f"\nJSON-Dateien OHNE parallelen 'annotated_' Ordner: {len(results['without_folder'])}")
    for item in results['without_folder']:
        json_path = Path(item['json_file'])
        expected_path = Path(item['expected_folder'])
        print(f"  ✗ {json_path.relative_to(Path.cwd()) if json_path.is_relative_to(Path.cwd()) else json_path} → {expected_path.name}/ (nicht vorhanden)")
    
    # Statistik
    if results['total_json_files'] > 0:
        percentage = (len(results['with_folder']) / results['total_json_files']) * 100
        print(f"\nStatistik: {percentage:.1f}% der JSON-Dateien haben einen parallelen 'annotated_' Ordner.")


def main():
    """
    Hauptfunktion des Scripts.
    """
    parser = argparse.ArgumentParser(
        description="Überprüft JSON-Dateien auf parallele 'annotated_' Ordner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main_batch_check_for_annotated-folder_from_json-results-file.py /pfad/zum/ordner
  python main_batch_check_for_annotated-folder_from_json-results-file.py ./detection_results
  python main_batch_check_for_annotated-folder_from_json-results-file.py . --no-auto-annotate
  
Das Script sucht für jede JSON-Datei nach einem parallelen Ordner mit 'annotated_' Prefix.
Beispiel: file.json → annotated_file/

Standardmäßig wird annotate_detections.py automatisch für fehlende Ordner aufgerufen.
        """
    )
    
    parser.add_argument(
        'directory',
        help='Verzeichnis, das nach JSON-Dateien durchsucht werden soll'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Ausführliche Ausgabe'
    )
    
    parser.add_argument(
        '--no-auto-annotate',
        action='store_true',
        help='Keine automatische Annotation für fehlende Ordner'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Durchsuche Verzeichnis rekursiv: {args.directory}")
    
    # JSON-Dateien finden
    json_files = find_json_files(args.directory)
    
    if not json_files:
        print(f"Keine JSON-Dateien in '{args.directory}' (inklusive Subordner) gefunden.")
        return
    
    if args.verbose:
        print(f"Gefundene JSON-Dateien:")
        for json_file in json_files:
            # Zeige relativen Pfad wenn möglich, sonst absoluten Pfad
            try:
                relative_path = json_file.relative_to(Path.cwd())
                print(f"  - {relative_path}")
            except ValueError:
                print(f"  - {json_file}")
    
    # Parallele Ordner überprüfen
    results = check_parallel_folders(json_files)
    
    # Ergebnisse ausgeben
    print_results(results)
    
    # Automatische Annotation für fehlende Ordner (falls nicht deaktiviert)
    if not args.no_auto_annotate and results['without_folder']:
        success = process_missing_annotations(results, args.verbose)
        if not success:
            sys.exit(1)
        
        # Nach der Annotation: Erneute Überprüfung
        print(f"\n=== ERNEUTE ÜBERPRÜFUNG ===\n")
        updated_results = check_parallel_folders(json_files)
        print_results(updated_results)
    
    print(f"\n✅ Script erfolgreich abgeschlossen.")


if __name__ == "__main__":
    main()
