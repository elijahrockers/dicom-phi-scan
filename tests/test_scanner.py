"""Tests for direct DICOM PHI scanning."""


import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.scanner import scan_file
from src.models import PHITagFinding, Severity, ScanReport


def _make_phi_file(tmp_path):
    """Create a minimal DICOM file with PHI header tags."""
    filepath = str(tmp_path / "test.dcm")
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "DOE^JANE"
    ds.PatientID = "MRN-12345"
    ds.InstitutionName = "Test Hospital"
    ds.BurnedInAnnotation = "NO"

    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.Rows = 16
    ds.Columns = 16
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((16, 16), dtype=np.uint8).tobytes()

    ds.save_as(filepath)
    return filepath


def test_scan_file_returns_model_instances(tmp_path):
    """Bug 5 regression: tag_findings should be model instances, not dicts."""
    filepath = _make_phi_file(tmp_path)
    report = scan_file(filepath)

    assert isinstance(report, ScanReport)
    assert len(report.tag_findings) > 0
    for f in report.tag_findings:
        assert isinstance(f, PHITagFinding)


def test_scan_file_risk_level(tmp_path):
    """Files with HIGH-severity tags should have HIGH risk."""
    filepath = _make_phi_file(tmp_path)
    report = scan_file(filepath)

    assert report.risk_level == Severity.HIGH


def test_scan_file_recommendations(tmp_path):
    """Should recommend header tag redaction when PHI is found."""
    filepath = _make_phi_file(tmp_path)
    report = scan_file(filepath)

    assert any("header tags" in r.lower() for r in report.recommendations)
