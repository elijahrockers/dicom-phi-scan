"""Create synthetic DICOM test fixtures with intentionally planted fake PHI.

These fixtures are used for testing the PHI screening pipeline. All data is
entirely synthetic — no real patient information is used.
"""

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from PIL import Image, ImageDraw, ImageFont
import os

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def create_phi_header_fixture():
    """Create a DICOM file with fake PHI in header tags only (no pixel PHI)."""

    filename = os.path.join(FIXTURES_DIR, "test_phi_header.dcm")
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # Plant fake PHI in header tags
    ds.PatientName = "DOE^JANE^M"
    ds.PatientID = "MRN-12345678"
    ds.PatientBirthDate = "19850315"
    ds.PatientSex = "F"
    ds.PatientAge = "041Y"
    ds.PatientAddress = "123 Fake Street, Houston, TX 77001"

    ds.InstitutionName = "Test Memorial Hospital"
    ds.InstitutionAddress = "456 Medical Center Blvd, Houston, TX 77030"
    ds.ReferringPhysicianName = "SMITH^ROBERT^J^DR"
    ds.PerformingPhysicianName = "JONES^ALICE^K^DR"
    ds.OperatorsName = "TECH_JOHNSON"

    ds.AccessionNumber = "ACC-2024-00789"
    ds.StudyID = "STUDY-456"
    ds.StudyDate = "20240115"
    ds.SeriesDate = "20240115"
    ds.StudyTime = "143022"
    ds.StationName = "CT_SCANNER_3"
    ds.DeviceSerialNumber = "SN-98765"

    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # BurnedInAnnotation explicitly NO
    ds.BurnedInAnnotation = "NO"

    # Minimal pixel data (small black image)
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((64, 64), dtype=np.uint8).tobytes()

    ds.save_as(filename)
    print(f"Created: {filename}")
    return filename


def create_phi_pixel_fixture():
    """Create a DICOM file with fake PHI burned into pixel data."""

    filename = os.path.join(FIXTURES_DIR, "test_phi_pixel.dcm")
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # Minimal header — already de-identified
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = "ANON-000"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"

    ds.InstitutionName = ""
    ds.AccessionNumber = ""
    ds.StudyDate = ""

    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # BurnedInAnnotation tag intentionally absent (common in real-world data)

    # Create image with burned-in fake PHI text
    width, height = 512, 512
    img = Image.new("L", (width, height), 0)  # Black background
    draw = ImageDraw.Draw(img)

    # Draw a simple "medical image" pattern (circle for anatomy)
    draw.ellipse([150, 150, 362, 362], fill=80)

    # Burn in fake PHI text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Patient name — top left (common in ultrasound, CR)
    draw.text((10, 10), "SMITH, JOHN A", fill=255, font=font)
    # MRN — top right
    draw.text((350, 10), "MRN: 9876543", fill=255, font=font)
    # Date of birth
    draw.text((10, 30), "DOB: 03/15/1990", fill=255, font=font)
    # Accession number
    draw.text((10, 50), "ACC#: A-2024-12345", fill=255, font=font)
    # Laterality marker (NOT PHI — should not be flagged)
    draw.text((480, 250), "L", fill=255, font=font)
    # Technical annotation (NOT PHI)
    draw.text((10, 490), "120 kV  250 mA", fill=255, font=font)

    pixel_array = np.array(img, dtype=np.uint8)

    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(filename)
    print(f"Created: {filename}")
    return filename


def create_clean_fixture():
    """Create a clean DICOM file with no PHI (negative test case)."""

    filename = os.path.join(FIXTURES_DIR, "test_clean.dcm")
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # All identifying fields properly de-identified
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = "ANON-000"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.InstitutionName = ""
    ds.ReferringPhysicianName = ""
    ds.AccessionNumber = ""
    ds.StudyDate = ""
    ds.BurnedInAnnotation = "NO"

    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # Clean pixel data
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((64, 64), dtype=np.uint8).tobytes()

    ds.save_as(filename)
    print(f"Created: {filename}")
    return filename


if __name__ == "__main__":
    print("Creating synthetic DICOM test fixtures...")
    create_phi_header_fixture()
    create_phi_pixel_fixture()
    create_clean_fixture()
    print("Done! All fixtures created with entirely synthetic/fake data.")
