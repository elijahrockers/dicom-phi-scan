"""Layer 2: Pixel matrix PHI detection via OCR + Claude classification.

Extracts pixel data to images, runs pytesseract OCR for text with bounding boxes,
then sends extracted text to Claude API for PHI classification.
"""

import io
import logging

import anthropic
import pydicom
import pytesseract
from PIL import Image
from pydicom.dataset import Dataset

from .models import BoundingBox, PixelPHIFinding, Severity

logger = logging.getLogger(__name__)

# Claude system prompt for PHI classification
PHI_CLASSIFIER_PROMPT = """\
You are a HIPAA PHI classifier for medical imaging. You will receive text strings
extracted via OCR from DICOM pixel data (burned-in annotations).

For each text string, classify it as one of:
- PHI: Contains protected health information (patient names, dates of birth, MRNs,
  accession numbers, addresses, phone numbers, SSNs, or other HIPAA identifiers)
- NOT_PHI: Anatomical labels, laterality markers (L/R), technical annotations
  (kV, mA, slice thickness), equipment info, or institutional protocol names

Respond with a JSON array. Each element must have:
- "text": the original text string
- "is_phi": true or false
- "phi_type": if PHI, the type (e.g., "patient_name", "date_of_birth", "mrn",
  "accession_number", "address", "phone"). null if not PHI.
- "confidence": float 0-1

Example response:
[
  {"text": "SMITH, JOHN", "is_phi": true, "phi_type": "patient_name", "confidence": 0.95},
  {"text": "L", "is_phi": false, "phi_type": null, "confidence": 0.99},
  {"text": "120 kV", "is_phi": false, "phi_type": null, "confidence": 0.98}
]

Respond ONLY with the JSON array, no other text.
"""


def extract_image(ds: Dataset) -> Image.Image | None:
    """Extract pixel data from a DICOM dataset as a PIL Image.

    Returns None if the dataset has no pixel data.
    """
    if not hasattr(ds, "PixelData"):
        return None

    try:
        pixel_array = ds.pixel_array
    except Exception:
        logger.warning("Could not decompress pixel data — skipping pixel analysis")
        return None

    # Handle multi-frame (take first frame)
    if len(pixel_array.shape) == 3 and pixel_array.shape[2] not in (3, 4):
        pixel_array = pixel_array[0]

    # Normalize to 8-bit for OCR
    if pixel_array.max() > 255:
        pixel_array = (
            (pixel_array - pixel_array.min())
            / (pixel_array.max() - pixel_array.min())
            * 255
        ).astype("uint8")

    return Image.fromarray(pixel_array)


def run_ocr(image: Image.Image) -> list[dict]:
    """Run pytesseract OCR and return text with bounding boxes.

    Returns:
        List of dicts with keys: text, x, y, width, height, conf
    """
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    results = []
    n_boxes = len(ocr_data["text"])
    for i in range(n_boxes):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])
        if text and conf > 30:  # Filter low-confidence noise
            results.append(
                {
                    "text": text,
                    "x": ocr_data["left"][i],
                    "y": ocr_data["top"][i],
                    "width": ocr_data["width"][i],
                    "height": ocr_data["height"][i],
                    "conf": conf,
                }
            )

    return results


def classify_text_with_claude(
    texts: list[str],
    client: anthropic.Anthropic | None = None,
) -> list[dict]:
    """Send OCR-extracted text to Claude for PHI classification.

    Args:
        texts: List of text strings from OCR.
        client: Anthropic client instance. Creates one if not provided.

    Returns:
        List of classification dicts from Claude.
    """
    if not texts:
        return []

    if client is None:
        client = anthropic.Anthropic()

    text_list = "\n".join(f"- \"{t}\"" for t in texts)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following OCR-extracted texts from a DICOM image:\n\n{text_list}",
            }
        ],
        system=PHI_CLASSIFIER_PROMPT,
    )

    import json

    response_text = message.content[0].text
    return json.loads(response_text)


# Severity mapping for PHI types
_PHI_TYPE_SEVERITY: dict[str, Severity] = {
    "patient_name": Severity.HIGH,
    "date_of_birth": Severity.HIGH,
    "mrn": Severity.HIGH,
    "accession_number": Severity.HIGH,
    "address": Severity.HIGH,
    "phone": Severity.HIGH,
    "ssn": Severity.HIGH,
    "date": Severity.MEDIUM,
    "age": Severity.MEDIUM,
}


def scan_pixels(
    ds: Dataset,
    client: anthropic.Anthropic | None = None,
) -> list[PixelPHIFinding]:
    """Full pixel PHI scan: extract image → OCR → Claude classification.

    Args:
        ds: A pydicom Dataset (must include pixel data).
        client: Anthropic client instance.

    Returns:
        List of PixelPHIFinding for detected PHI.
    """
    image = extract_image(ds)
    if image is None:
        return []

    ocr_results = run_ocr(image)
    if not ocr_results:
        return []

    texts = [r["text"] for r in ocr_results]
    classifications = classify_text_with_claude(texts, client)

    findings: list[PixelPHIFinding] = []
    # Build lookup from OCR results by text
    ocr_lookup = {r["text"]: r for r in ocr_results}

    for cls in classifications:
        if cls.get("is_phi"):
            text = cls["text"]
            ocr_match = ocr_lookup.get(text, {})
            phi_type = cls.get("phi_type", "unknown")
            findings.append(
                PixelPHIFinding(
                    text=text,
                    bbox=BoundingBox(
                        x=ocr_match.get("x", 0),
                        y=ocr_match.get("y", 0),
                        width=ocr_match.get("width", 0),
                        height=ocr_match.get("height", 0),
                    ),
                    phi_type=phi_type,
                    confidence=cls.get("confidence", 0.0),
                    severity=_PHI_TYPE_SEVERITY.get(phi_type, Severity.MEDIUM),
                )
            )

    return findings
