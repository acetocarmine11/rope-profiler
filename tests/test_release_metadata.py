"""Validate local release contracts without importing the model runtime."""
from __future__ import annotations

import importlib.util
from importlib.metadata import version as installed_version
import io
from pathlib import Path
import tarfile
import zipfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rope_release_check", PROJECT_ROOT / "tools" / "check_release.py")
assert SPEC is not None and SPEC.loader is not None
release_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_check)


def fixture_project(tmp_path: Path, *, license_expression: str | None = None) -> Path:
    version_dir = tmp_path / "src" / "rope_profiler"
    version_dir.mkdir(parents=True)
    (version_dir / "_version.py").write_text('__version__ = "0.1.0a1"\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("Local package description.\n", encoding="utf-8")
    license_field = "" if license_expression is None else f'license = "{license_expression}"\n'
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "rope-profiler"\ndynamic = ["version"]\n'
        'readme = "README.md"\nrequires-python = ">=3.10"\n'
        + license_field
        + '[tool.setuptools.dynamic]\nversion = {attr = "rope_profiler._version.__version__"}\n',
        encoding="utf-8",
    )
    return tmp_path


def fixture_artifacts(root: Path, version: str = "0.1.0a1", *, license_expression: str | None = None,
                      requires_python: str = ">=3.10") -> Path:
    dist = root / "dist"
    dist.mkdir()
    text = f"Metadata-Version: 2.4\nName: rope-profiler\nVersion: {version}\nRequires-Python: {requires_python}\n"
    if license_expression is not None:
        text += f"License-Expression: {license_expression}\n"
    with zipfile.ZipFile(dist / f"rope_profiler-{version}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"rope_profiler-{version}.dist-info/METADATA", text)
    encoded = text.encode("utf-8")
    with tarfile.open(dist / f"rope_profiler-{version}.tar.gz", "w:gz") as sdist:
        info = tarfile.TarInfo(f"rope_profiler-{version}/PKG-INFO")
        info.size = len(encoded)
        sdist.addfile(info, io.BytesIO(encoded))
    return dist


def test_live_project_uses_one_dynamic_version_source():
    report = release_check.check_project(PROJECT_ROOT)
    assert report["project"] == "rope-profiler"
    assert report["version"] == installed_version("rope-profiler")
    assert report["publishes"] is False


def test_unlicensed_local_project_can_be_validated(tmp_path):
    report = release_check.check_project(fixture_project(tmp_path))
    assert report["license_required"] is False
    assert report["license_expression"] is None


def test_release_requires_license_expression(tmp_path):
    root = fixture_project(tmp_path)
    (root / "LICENSE").write_text("Local license text.\n", encoding="utf-8")
    with pytest.raises(release_check.ReleaseCheckError, match="SPDX expression"):
        release_check.check_project(root, require_license=True)


@pytest.mark.parametrize("contents", [None, "   \n"])
def test_release_requires_nonempty_license_file(tmp_path, contents):
    root = fixture_project(tmp_path, license_expression="MIT")
    if contents is not None:
        (root / "LICENSE").write_text(contents, encoding="utf-8")
    with pytest.raises(release_check.ReleaseCheckError, match="license file|LICENSE/LICENCE"):
        release_check.check_project(root, require_license=True)


def test_release_validates_spdx_expression(tmp_path):
    root = fixture_project(tmp_path, license_expression="Definitely-Not-An-SPDX-License")
    (root / "LICENSE").write_text("Local license text.\n", encoding="utf-8")
    with pytest.raises(release_check.ReleaseCheckError, match="invalid project.license"):
        release_check.check_project(root, require_license=True)


def test_source_and_built_metadata_agree(tmp_path):
    root = fixture_project(tmp_path, license_expression="MIT")
    (root / "LICENSE").write_text("Local license text.\n", encoding="utf-8")
    dist = fixture_artifacts(root, license_expression="MIT")
    report = release_check.check_project(root, require_license=True, dist_dir=dist)
    assert len(report["artifacts"]) == 2
    assert report["license_files"] == ["LICENSE"]
    assert report["license_expression"] == "MIT"


def test_stale_distribution_version_is_rejected(tmp_path):
    root = fixture_project(tmp_path)
    dist = fixture_artifacts(root, version="0.1.0a0")
    with pytest.raises(release_check.ReleaseCheckError, match="Version"):
        release_check.check_project(root, dist_dir=dist)


def test_python_requirement_order_can_be_normalized_by_build_backend(tmp_path):
    root = fixture_project(tmp_path)
    path = root / "pyproject.toml"
    path.write_text(path.read_text().replace('requires-python = ">=3.10"',
                                             'requires-python = ">=3.10,<3.13"'), encoding="utf-8")
    dist = fixture_artifacts(root, requires_python="<3.13,>=3.10")
    assert len(release_check.check_project(root, dist_dir=dist)["artifacts"]) == 2


def test_different_python_requirement_is_rejected(tmp_path):
    root = fixture_project(tmp_path)
    dist = fixture_artifacts(root, requires_python=">=3.11")
    with pytest.raises(release_check.ReleaseCheckError, match="Requires-Python"):
        release_check.check_project(root, dist_dir=dist)


def test_license_metadata_must_match_source(tmp_path):
    root = fixture_project(tmp_path, license_expression="MIT")
    (root / "LICENSE").write_text("Local license text.\n", encoding="utf-8")
    dist = fixture_artifacts(root, license_expression="Apache-2.0")
    with pytest.raises(release_check.ReleaseCheckError, match="License-Expression"):
        release_check.check_project(root, require_license=True, dist_dir=dist)


def test_static_and_dynamic_versions_cannot_coexist(tmp_path):
    root = fixture_project(tmp_path)
    path = root / "pyproject.toml"
    path.write_text(path.read_text().replace('dynamic = ["version"]', 'dynamic = ["version"]\nversion = "0.1.0a0"'), encoding="utf-8")
    with pytest.raises(release_check.ReleaseCheckError, match="must be dynamic"):
        release_check.check_project(root)


def test_version_check_does_not_execute_source(tmp_path):
    root = fixture_project(tmp_path)
    (root / "src" / "rope_profiler" / "_version.py").write_text(
        '__version__ = "0.1.0a1"\nraise RuntimeError("must not execute")\n', encoding="utf-8")
    assert release_check.check_project(root)["version"] == "0.1.0a1"
