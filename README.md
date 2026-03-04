# DICOM PHI Scanner

Two-layer pipeline for detecting Protected Health Information (PHI) in DICOM medical imaging files. Scans both header tags and burned-in pixel text to identify PHI that must be removed before data sharing.

Built for healthcare data engineers who need to verify DICOM de-identification before inter-institutional data sharing or research use.

## Architecture

```mermaid
flowchart TB
    subgraph Input
        DCM[DICOM File]
    end

    subgraph Layer1["Layer 1: Header Tag Analysis"]
        PARSE[pydicom Tag Parser]
        HIPAA[HIPAA Safe Harbor<br/>18 Identifier Check]
        PARSE --> HIPAA
    end

    subgraph Layer2["Layer 2: Pixel PHI Detection"]
        EXTRACT[Pixel Data Extraction<br/>pydicom + Pillow]
        OCR[EasyOCR<br/>Text + Bounding Boxes]
        EXTRACT --> OCR
    end

    subgraph Output
        REPORT[Scan Report<br/>JSON + Summary]
        RECS[Remediation<br/>Recommendations]
    end

    DCM --> Layer1
    DCM -->|"BurnedInAnnotation<br/>YES or missing"| Layer2
    Layer1 -->|findings| Output
    Layer2 -->|findings| Output
```

## How It Works

### Layer 1 — Header Tag Analysis (`src/tag_scanner.py`)
Parses DICOM metadata tags against the HIPAA Safe Harbor de-identification standard. Checks ~50 tags across categories:
- **Direct identifiers** (HIGH): Patient name, ID, birth date, address, phone
- **Institutional** (HIGH): Institution name/address, physician names, accession numbers
- **Temporal** (MEDIUM): Study/series dates and times
- **Device** (MEDIUM): Station name, device serial number
- **UIDs** (MEDIUM): Study/Series/SOP Instance UIDs

Common de-identification placeholders (ANONYMOUS, REDACTED, etc.) are filtered out to reduce false positives.

### Layer 2 — Pixel PHI Detection (`src/pixel_scanner.py`)
Detects PHI burned into pixel data (common in ultrasound, CR, secondary capture):
1. Extracts pixel data to image via `pydicom` + `Pillow`
2. Runs EasyOCR to extract text with bounding box coordinates and confidence scores
3. All detected text above the confidence threshold is flagged as potential PHI

### Scanning Pipeline (`src/scanner.py`)
1. Runs header tag scan
2. Checks `BurnedInAnnotation (0028,0301)` — if YES or missing, triggers pixel scan
3. Aggregates findings, computes overall risk level (HIGH / MEDIUM / LOW), and generates remediation recommendations

## Requirements

- Python 3.10+

## Quick Start

```bash
# Install (editable mode — code changes take effect immediately)
pip install -e .

# Scan a single file (summary to screen, JSON report to file)
dicom-phi-scan path/to/file.dcm -o report.json

# Batch scan a directory
dicom-phi-scan --dir path/to/dicoms/ -o results.jsonl

# Follow symlinks (e.g. for symlinked dataset subsets)
dicom-phi-scan --dir path/to/dicoms/ -L -o results.jsonl

# Limit number of files in batch mode
dicom-phi-scan --dir path/to/dicoms/ -L -o results.jsonl --limit 50

# Force CPU for OCR (GPU/CUDA is auto-detected by default)
dicom-phi-scan path/to/file.dcm -o report.json --cpu

# Query the JSONL report for HIGH risk files
jq 'select(.risk_level == "high") | .filepath' results.jsonl
```

## Python API

```python
from src.scanner import scan_file

report = scan_file("path/to/file.dcm")
print(report.risk_level)       # Severity.HIGH / MEDIUM / LOW
print(report.total_phi_count)  # number of findings
print(report.recommendations)  # list of action items
```

## Project Structure

```
src/
├── cli.py             # CLI entry point (dicom-phi-scan)
├── models.py          # Pydantic models (ScanReport, PHITagFinding, PixelPHIFinding, BatchReport)
├── pixel_scanner.py   # Layer 2: OCR pixel text detection
├── scanner.py         # Orchestration pipeline
└── tag_scanner.py     # Layer 1: DICOM header tag analysis
```

## Design Decisions

- **Two-layer approach**: Header-only scanning misses burned-in annotations, which are common in ultrasound, CR, and secondary capture DICOM objects. Pixel analysis catches what tag scanning cannot.
- **Flag all OCR text as PHI**: Rather than attempting to classify burned-in text (which risks false negatives), all OCR-detected text is flagged as potential PHI. This conservative approach prioritizes patient privacy.
- **BurnedInAnnotation tag is checked but not trusted**: This tag is frequently missing or incorrectly set in real-world DICOM data. Pixel analysis still runs when the tag is absent.
- **Streaming batch output**: Batch scans stream per-file JSONL to disk and accumulate only lightweight stats in memory, avoiding OOM on large datasets.
- **Synthetic test data**: Real DICOM datasets from TCIA are already de-identified and don't exercise the PHI detection path. Synthetic data with planted fake PHI gives controlled, repeatable test cases.

## Stack

Python · pydicom · Pillow · EasyOCR · Pydantic

## License

MIT
