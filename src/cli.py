"""CLI entry point for DICOM PHI scanning."""

import argparse
import json
import logging
import sys

from .agent import run_agent, run_direct_scan


def main():
    parser = argparse.ArgumentParser(
        description="DICOM PHI Screening Agent — scan DICOM files for protected health information"
    )
    parser.add_argument("filepath", help="Path to DICOM file to scan")
    parser.add_argument(
        "--mode",
        choices=["agent", "direct"],
        default="agent",
        help="Scan mode: 'agent' uses Claude orchestration, 'direct' runs scans sequentially (default: agent)",
    )
    parser.add_argument(
        "--output",
        choices=["json", "summary"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.mode == "agent":
            report = run_agent(args.filepath)
        else:
            report = run_direct_scan(args.filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output == "json":
        print(report.model_dump_json(indent=2))
    else:
        _print_summary(report)


def _print_summary(report):
    """Print a human-readable summary of the scan report."""
    risk_colors = {"high": "RED", "medium": "YELLOW", "low": "GREEN"}
    risk = report.risk_level.value

    print(f"\n{'='*60}")
    print(f"DICOM PHI Scan Report")
    print(f"{'='*60}")
    print(f"File: {report.filepath}")
    print(f"Risk Level: [{risk_colors[risk]}] {risk.upper()}")
    print(f"Total PHI Findings: {report.total_phi_count}")
    print()

    if report.tag_findings:
        print(f"--- Header Tag Findings ({len(report.tag_findings)}) ---")
        for f in report.tag_findings:
            if isinstance(f, dict):
                print(f"  [{f['severity'].upper()}] {f['tag']} {f['tag_name']}: {f['value']}")
            else:
                print(f"  [{f.severity.upper()}] {f.tag} {f.tag_name}: {f.value}")
        print()

    if report.pixel_findings:
        print(f"--- Pixel PHI Findings ({len(report.pixel_findings)}) ---")
        for f in report.pixel_findings:
            if isinstance(f, dict):
                bbox = f["bbox"]
                print(f"  [{f['severity'].upper()}] \"{f['text']}\" ({f['phi_type']}) at ({bbox['x']},{bbox['y']}) {bbox['width']}x{bbox['height']}")
            else:
                print(f"  [{f.severity.upper()}] \"{f.text}\" ({f.phi_type}) at ({f.bbox.x},{f.bbox.y}) {f.bbox.width}x{f.bbox.height}")
        print()

    bia = report.burned_in_annotation_value or "NOT PRESENT"
    print(f"BurnedInAnnotation (0028,0301): {bia}")
    print()

    print("Recommendations:")
    for r in report.recommendations:
        print(f"  - {r}")
    print(f"{'='*60}\n")
