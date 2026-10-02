"""Installed RECORD integrity checks use only the standard library."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wheel_integrity_smoke", ROOT / "tools" / "wheel_smoke.py")
assert SPEC is not None and SPEC.loader is not None
wheel_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wheel_smoke)


@pytest.fixture
def installed_distribution(tmp_path):
    metadata_dir = tmp_path / "rope_profiler-0.1.0a1.dist-info"
    metadata_dir.mkdir()
    package_dir = tmp_path / "rope_profiler"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_bytes(b'__version__ = "0.1.0a1"\n')
    (metadata_dir / "METADATA").write_bytes(b"Metadata-Version: 2.4\nName: rope-profiler\nVersion: 0.1.0a1\n")
    rows = []
    for path in (package_dir / "__init__.py", metadata_dir / "METADATA"):
        content = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
        rows.append([str(path.relative_to(tmp_path)), f"sha256={digest}", str(len(content))])
    # RECORD cannot hash itself; generated pyc files may be absent or regenerated.
    rows.append([str((metadata_dir / "RECORD").relative_to(tmp_path)), "", ""])
    rows.append(["rope_profiler/__pycache__/__init__.cpython-312.pyc", "", ""])
    record = metadata_dir / "RECORD"

    def write_rows():
        with record.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

    write_rows()
    return importlib.metadata.Distribution.at(metadata_dir), rows, write_rows


def test_verified_count_skips_unhashed_record_and_generated_pyc(installed_distribution):
    distribution, _, _ = installed_distribution
    assert wheel_smoke.verify_installed_record(distribution) == 2


def test_same_size_byte_change_is_rejected(installed_distribution):
    distribution, rows, _ = installed_distribution
    path = Path(distribution.locate_file(rows[0][0]))
    path.write_bytes(path.read_bytes().replace(b"0.1.0a1", b"0.1.0a2"))
    with pytest.raises(AssertionError, match="hash mismatch"):
        wheel_smoke.verify_installed_record(distribution)


def test_missing_recorded_file_is_rejected(installed_distribution):
    distribution, rows, _ = installed_distribution
    Path(distribution.locate_file(rows[0][0])).unlink()
    with pytest.raises(AssertionError, match="file is missing"):
        wheel_smoke.verify_installed_record(distribution)


def test_incorrect_recorded_size_is_rejected(installed_distribution):
    distribution, rows, write_rows = installed_distribution
    rows[0][2] = str(int(rows[0][2]) + 1)
    write_rows()
    with pytest.raises(AssertionError, match="size mismatch"):
        wheel_smoke.verify_installed_record(distribution)


def test_unhashed_source_file_is_rejected(installed_distribution):
    distribution, rows, write_rows = installed_distribution
    rows[0][1] = ""
    write_rows()
    with pytest.raises(AssertionError, match="omits the hash"):
        wheel_smoke.verify_installed_record(distribution)


def test_hashed_file_without_size_is_rejected(installed_distribution):
    distribution, rows, write_rows = installed_distribution
    rows[0][2] = ""
    write_rows()
    with pytest.raises(AssertionError, match="omits the size"):
        wheel_smoke.verify_installed_record(distribution)
