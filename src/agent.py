"""Agent orchestration layer using Claude API tool use.

Claude acts as the orchestrator: decides which tools to call (tag parser,
image extractor, OCR, classifier), aggregates results, and writes the
final de-identification report.
"""

import json
import logging

import anthropic
import pydicom

from .models import Severity, ScanReport
from .tag_scanner import scan_tags, get_burned_in_annotation
from .pixel_scanner import scan_pixels

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """\
You are a DICOM PHI Screening Agent. Your job is to analyze DICOM files for
protected health information (PHI) that must be removed before data sharing.

You have access to the following tools:

1. scan_header_tags — Scans DICOM header tags for PHI fields per HIPAA Safe Harbor
2. scan_pixel_data — Extracts pixel data, runs OCR, and classifies detected text as PHI or not
3. generate_report — Produces the final structured de-identification report

Workflow:
1. Always start with scan_header_tags to check metadata
2. Check if BurnedInAnnotation tag is YES or missing — if so, run scan_pixel_data
3. After both scans complete, call generate_report with all findings

Be thorough. PHI leakage in shared datasets is a HIPAA violation.
"""

# Tool definitions for Claude API
TOOLS = [
    {
        "name": "scan_header_tags",
        "description": "Scan DICOM header tags for PHI fields per HIPAA Safe Harbor de-identification standard. Returns a list of findings with tag name, value, severity, and HIPAA category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the DICOM file to scan",
                }
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "scan_pixel_data",
        "description": "Extract pixel data from DICOM, run OCR to find burned-in text, and classify each text region as PHI or not using AI. Returns pixel PHI findings with bounding box coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the DICOM file to scan",
                }
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "generate_report",
        "description": "Generate the final de-identification report combining header tag and pixel scan findings. Produces both JSON and human-readable summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "tag_findings_json": {
                    "type": "string",
                    "description": "JSON string of tag findings from scan_header_tags",
                },
                "pixel_findings_json": {
                    "type": "string",
                    "description": "JSON string of pixel findings from scan_pixel_data",
                },
                "burned_in_tag_present": {"type": "boolean"},
                "burned_in_tag_value": {"type": "string", "nullable": True},
            },
            "required": ["filepath", "tag_findings_json", "pixel_findings_json"],
        },
    },
]


def _handle_tool_call(
    tool_name: str,
    tool_input: dict,
    client: anthropic.Anthropic,
) -> str:
    """Execute a tool call and return the result as JSON string."""

    if tool_name == "scan_header_tags":
        filepath = tool_input["filepath"]
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        findings = scan_tags(ds)
        bia_present, bia_value = get_burned_in_annotation(ds)
        return json.dumps(
            {
                "findings": [f.model_dump() for f in findings],
                "burned_in_annotation_present": bia_present,
                "burned_in_annotation_value": bia_value,
            }
        )

    elif tool_name == "scan_pixel_data":
        filepath = tool_input["filepath"]
        ds = pydicom.dcmread(filepath)
        findings = scan_pixels(ds, client)
        return json.dumps({"findings": [f.model_dump() for f in findings]})

    elif tool_name == "generate_report":
        filepath = tool_input["filepath"]
        tag_findings = json.loads(tool_input.get("tag_findings_json", "[]"))
        pixel_findings = json.loads(tool_input.get("pixel_findings_json", "[]"))
        bia_present = tool_input.get("burned_in_tag_present", False)
        bia_value = tool_input.get("burned_in_tag_value")

        total = len(tag_findings) + len(pixel_findings)
        high_count = sum(
            1
            for f in tag_findings + pixel_findings
            if f.get("severity") == "high"
        )

        if high_count > 0:
            risk = "high"
        elif total > 0:
            risk = "medium"
        else:
            risk = "low"

        recommendations = []
        if tag_findings:
            recommendations.append(
                "Remove or redact PHI from DICOM header tags before sharing"
            )
        if pixel_findings:
            recommendations.append(
                "Redact burned-in PHI text from pixel data at identified bounding box regions"
            )
        if not bia_present:
            recommendations.append(
                "BurnedInAnnotation tag (0028,0301) is missing — add it for compliance"
            )
        if not tag_findings and not pixel_findings:
            recommendations.append("No PHI detected — file appears safe for sharing")

        report = ScanReport(
            filepath=filepath,
            tag_findings=tag_findings,
            pixel_findings=pixel_findings,
            burned_in_annotation_tag_present=bia_present,
            burned_in_annotation_value=bia_value,
            total_phi_count=total,
            risk_level=Severity(risk),
            recommendations=recommendations,
        )
        return report.model_dump_json(indent=2)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent(filepath: str, client: anthropic.Anthropic | None = None) -> ScanReport:
    """Run the DICOM PHI screening agent on a file.

    The agent uses Claude to orchestrate the scanning process, deciding
    which tools to call and aggregating results into a final report.

    Args:
        filepath: Path to the DICOM file to scan.
        client: Anthropic client instance.

    Returns:
        ScanReport with all findings and recommendations.
    """
    if client is None:
        client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": f"Scan the DICOM file at '{filepath}' for PHI. Run header tag scan first, then pixel scan if needed, then generate the final report.",
        }
    ]

    # Agent loop — let Claude drive the tool calls
    max_iterations = 10
    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=AGENT_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect all tool uses and text from the response
        tool_uses = []
        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)

        if response.stop_reason == "end_turn" or not tool_uses:
            # Agent is done — extract final report from the last generate_report call
            break

        # Add assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and build tool results
        tool_results = []
        for tool_use in tool_uses:
            logger.info(f"Agent calling tool: {tool_use.name}")
            result = _handle_tool_call(tool_use.name, tool_use.input, client)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    # Parse the final report from the last generate_report tool result
    for msg in reversed(messages):
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    try:
                        data = json.loads(item["content"])
                        if "filepath" in data and "tag_findings" in data:
                            return ScanReport(**data)
                    except (json.JSONDecodeError, KeyError):
                        continue

    # Fallback: run scans directly if agent didn't produce a report
    logger.warning("Agent did not produce a report — running scans directly")
    return run_direct_scan(filepath, client)


def run_direct_scan(
    filepath: str, client: anthropic.Anthropic | None = None
) -> ScanReport:
    """Run scans directly without agent orchestration (fallback/testing)."""
    ds = pydicom.dcmread(filepath)
    tag_findings = scan_tags(ds)
    bia_present, bia_value = get_burned_in_annotation(ds)

    pixel_findings = []
    if not bia_present or (bia_value and bia_value.upper() == "YES"):
        if client is None:
            client = anthropic.Anthropic()
        pixel_findings = scan_pixels(ds, client)

    total = len(tag_findings) + len(pixel_findings)
    high_count = sum(1 for f in tag_findings if f.severity == Severity.HIGH) + sum(
        1 for f in pixel_findings if f.severity == Severity.HIGH
    )

    recommendations = []
    if tag_findings:
        recommendations.append("Remove or redact PHI from DICOM header tags before sharing")
    if pixel_findings:
        recommendations.append(
            "Redact burned-in PHI text from pixel data at identified bounding box regions"
        )
    if not bia_present:
        recommendations.append(
            "BurnedInAnnotation tag (0028,0301) is missing — add it for compliance"
        )
    if not tag_findings and not pixel_findings:
        recommendations.append("No PHI detected — file appears safe for sharing")

    return ScanReport(
        filepath=filepath,
        tag_findings=[f.model_dump() for f in tag_findings],
        pixel_findings=[f.model_dump() for f in pixel_findings],
        burned_in_annotation_tag_present=bia_present,
        burned_in_annotation_value=bia_value,
        total_phi_count=total,
        risk_level=Severity.HIGH if high_count > 0 else (Severity.MEDIUM if total > 0 else Severity.LOW),
        recommendations=recommendations,
    )
