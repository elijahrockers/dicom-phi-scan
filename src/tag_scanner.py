"""Layer 1: DICOM header tag PHI scanner.

Parses DICOM tags against the HIPAA Safe Harbor de-identification standard
(18 identifiers) and flags known PHI tags with severity classification.
"""

import pydicom
from pydicom.dataset import Dataset

from .models import PHITagFinding, Severity

# DICOM tags that may contain PHI, mapped to HIPAA category and severity.
# Based on DICOM PS3.15 Table E.1-1 and HIPAA Safe Harbor 18 identifiers.
PHI_TAGS: dict[tuple[int, int], tuple[str, str, Severity]] = {
    # Direct patient identifiers — HIGH severity
    (0x0010, 0x0010): ("PatientName", "Name", Severity.HIGH),
    (0x0010, 0x0020): ("PatientID", "Unique Identifier", Severity.HIGH),
    (0x0010, 0x0030): ("PatientBirthDate", "Date", Severity.HIGH),
    (0x0010, 0x1000): ("OtherPatientIDs", "Unique Identifier", Severity.HIGH),
    (0x0010, 0x1001): ("OtherPatientNames", "Name", Severity.HIGH),
    (0x0010, 0x1040): ("PatientAddress", "Address", Severity.HIGH),
    (0x0010, 0x2154): ("PatientTelephoneNumbers", "Phone", Severity.HIGH),
    (0x0010, 0x21F0): ("PatientReligiousPreference", "Other", Severity.MEDIUM),
    (0x0010, 0x2297): ("ResponsiblePerson", "Name", Severity.HIGH),
    # Patient demographics — MEDIUM severity
    (0x0010, 0x0040): ("PatientSex", "Demographics", Severity.LOW),
    (0x0010, 0x1010): ("PatientAge", "Age", Severity.MEDIUM),
    (0x0010, 0x1020): ("PatientSize", "Biometric", Severity.LOW),
    (0x0010, 0x1030): ("PatientWeight", "Biometric", Severity.LOW),
    (0x0010, 0x2160): ("EthnicGroup", "Demographics", Severity.MEDIUM),
    # Institutional identifiers — HIGH severity
    (0x0008, 0x0080): ("InstitutionName", "Institution", Severity.HIGH),
    (0x0008, 0x0081): ("InstitutionAddress", "Address", Severity.HIGH),
    (0x0008, 0x1040): ("InstitutionalDepartmentName", "Institution", Severity.MEDIUM),
    (0x0008, 0x0090): ("ReferringPhysicianName", "Name", Severity.HIGH),
    (0x0008, 0x1048): ("PhysiciansOfRecord", "Name", Severity.HIGH),
    (0x0008, 0x1050): ("PerformingPhysicianName", "Name", Severity.HIGH),
    (0x0008, 0x1060): ("NameOfPhysiciansReadingStudy", "Name", Severity.HIGH),
    (0x0008, 0x1070): ("OperatorsName", "Name", Severity.HIGH),
    # Study/accession identifiers — HIGH severity
    (0x0008, 0x0050): ("AccessionNumber", "Unique Identifier", Severity.HIGH),
    (0x0020, 0x0010): ("StudyID", "Unique Identifier", Severity.HIGH),
    # Dates — MEDIUM severity (dates beyond year can be identifying)
    (0x0008, 0x0020): ("StudyDate", "Date", Severity.MEDIUM),
    (0x0008, 0x0021): ("SeriesDate", "Date", Severity.MEDIUM),
    (0x0008, 0x0022): ("AcquisitionDate", "Date", Severity.MEDIUM),
    (0x0008, 0x0023): ("ContentDate", "Date", Severity.MEDIUM),
    (0x0008, 0x0030): ("StudyTime", "Date", Severity.LOW),
    (0x0008, 0x0031): ("SeriesTime", "Date", Severity.LOW),
    (0x0008, 0x0032): ("AcquisitionTime", "Date", Severity.LOW),
    (0x0008, 0x0033): ("ContentTime", "Date", Severity.LOW),
    # Device/station — MEDIUM severity (can identify location)
    (0x0008, 0x1010): ("StationName", "Device", Severity.MEDIUM),
    (0x0008, 0x1090): ("ManufacturerModelName", "Device", Severity.LOW),
    (0x0018, 0x1000): ("DeviceSerialNumber", "Device", Severity.MEDIUM),
    # Request/order identifiers
    (0x0040, 0x0006): ("ScheduledPerformingPhysicianName", "Name", Severity.HIGH),
    (0x0040, 0x0244): ("PerformedProcedureStepStartDate", "Date", Severity.MEDIUM),
    (0x0040, 0x0253): ("PerformedProcedureStepID", "Unique Identifier", Severity.MEDIUM),
    (0x0040, 0x1001): ("RequestedProcedureID", "Unique Identifier", Severity.MEDIUM),
    # UIDs that might encode site/time info
    (0x0020, 0x000D): ("StudyInstanceUID", "UID", Severity.MEDIUM),
    (0x0020, 0x000E): ("SeriesInstanceUID", "UID", Severity.MEDIUM),
    (0x0008, 0x0018): ("SOPInstanceUID", "UID", Severity.MEDIUM),
    # Burned-in annotation flag
    (0x0028, 0x0301): ("BurnedInAnnotation", "Flag", Severity.LOW),
}


def scan_tags(ds: Dataset) -> list[PHITagFinding]:
    """Scan a DICOM dataset for PHI in header tags.

    Args:
        ds: A pydicom Dataset (already loaded).

    Returns:
        List of PHITagFinding for each tag containing a non-empty value.
    """
    findings: list[PHITagFinding] = []

    for (group, elem), (tag_name, hipaa_cat, severity) in PHI_TAGS.items():
        tag = pydicom.tag.Tag(group, elem)
        if tag in ds:
            value = str(ds[tag].value).strip()
            if value and value.upper() not in ("", "NONE", "UNKNOWN"):
                findings.append(
                    PHITagFinding(
                        tag=f"({group:04X},{elem:04X})",
                        tag_name=tag_name,
                        value=value,
                        severity=severity,
                        hipaa_category=hipaa_cat,
                    )
                )

    return findings


def get_burned_in_annotation(ds: Dataset) -> tuple[bool, str | None]:
    """Check the BurnedInAnnotation (0028,0301) tag.

    Returns:
        Tuple of (tag_present, tag_value).
    """
    tag = pydicom.tag.Tag(0x0028, 0x0301)
    if tag in ds:
        return True, str(ds[tag].value)
    return False, None


def scan_file(filepath: str) -> list[PHITagFinding]:
    """Convenience function to scan a DICOM file from path."""
    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
    return scan_tags(ds)
