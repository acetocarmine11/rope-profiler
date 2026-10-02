"""A release candidate retains one verified pair of prebuilt artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_bundle_tool", ROOT / "tools" / "prepare_release.py")
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    files = {
        "pyproject.toml": '[project]\nname="rope-profiler"\ndynamic=["version"]\nreadme="README.md"\nrequires-python=">=3.10,<3.13"\n[tool.setuptools.dynamic]\nversion={attr="rope_profiler._version.__version__"}\n',
        "README.md": "A local package.\n",
        "MANIFEST.in": "recursive-include tools *.py\n",
        "src/rope_profiler/_version.py": '__version__ = "0.1.0a1"\n',
        "src/rope_profiler/__init__.py": "from ._version import __version__\n",
        "src/rope_profiler/score.py": "answer = 42\n",
        "docs/RELEASING.md": "Local release instructions.\n",
        "tools/check_input.py": "print('local')\n",
        ".github/workflows/ci.yml": "name: Local checks\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def build_fixture(root, *, version="0.1.0a1", license_expression=None, omit_sdist=(),
                  generated_setup_cfg=b"[egg_info]\ntag_build =\ntag_date = 0\n",
                  wheel_readme=None, sdist_readme=None,
                  wheel_requirements=(), sdist_requirements=None,
                  wheel_extras=(), sdist_extras=None):
    dist = root / "dist"
    dist.mkdir()
    metadata = f"Metadata-Version: 2.4\nName: rope-profiler\nVersion: {version}\nRequires-Python: <3.13,>=3.10\n"
    if license_expression:
        metadata += f"License-Expression: {license_expression}\nLicense-File: LICENSE\n"
    current_readme = (root / "README.md").read_text()
    def metadata_text(readme, requirements, extras):
        return (metadata + "".join(f"Requires-Dist: {value}\n" for value in requirements)
                + "".join(f"Provides-Extra: {value}\n" for value in extras)
                + "\n" + (current_readme if readme is None else readme))
    wheel_metadata = metadata_text(wheel_readme, wheel_requirements, wheel_extras)
    sdist_metadata = metadata_text(sdist_readme, wheel_requirements if sdist_requirements is None else sdist_requirements,
                                  wheel_extras if sdist_extras is None else sdist_extras)
    wheel = dist / f"rope_profiler-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"rope_profiler-{version}.dist-info/METADATA", wheel_metadata)
        for path in (root / "src" / "rope_profiler").rglob("*.py"):
            archive.writestr(path.relative_to(root / "src").as_posix(), path.read_bytes())
        if license_expression:
            archive.writestr(f"rope_profiler-{version}.dist-info/licenses/LICENSE", (root / "LICENSE").read_bytes())
    sdist = dist / f"rope_profiler-{version}.tar.gz"
    inputs = [(path.relative_to(root).as_posix(), path.read_bytes())
              for path in root.rglob("*") if path.is_file() and dist not in path.parents
              and path.relative_to(root).as_posix() not in omit_sdist]
    inputs.extend([("PKG-INFO", sdist_metadata.encode()), ("setup.cfg", generated_setup_cfg)])
    with tarfile.open(sdist, "w:gz") as archive:
        for name, content in inputs:
            entry = tarfile.TarInfo(f"rope_profiler-{version}/{name}")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return dist


def test_draft_bundle_preserves_artifact_bytes_and_records_source_identity(project, tmp_path):
    dist = build_fixture(project)
    output = tmp_path / "candidate"
    manifest = bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    assert manifest["schema"] == bundle.SCHEMA
    assert manifest["version"] == "0.1.0a1"
    assert manifest["mode"] == "draft" and manifest["license"]["verified"] is False
    assert set(manifest["source_files"]) >= {"README.md", "pyproject.toml", "MANIFEST.in", "docs/RELEASING.md", "tools/check_input.py", ".github/workflows/ci.yml"}
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name, details in manifest["artifacts"].items():
        assert (output / name).read_bytes() == (dist / name).read_bytes()
        assert details["sha256"] == hashlib.sha256((dist / name).read_bytes()).hexdigest()
        assert f"{details['sha256']}  {name}\n" in (output / "SHA256SUMS").read_text()


def test_identical_candidate_is_idempotent_and_verify_only_does_not_write(project, tmp_path):
    dist = build_fixture(project)
    output = tmp_path / "candidate"
    first = bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    snapshot = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert bundle.prepare_release(project, dist_dir=dist, output_dir=output) == first
    assert bundle.prepare_release(project, output_dir=output, verify_only=True) == first
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()} == snapshot


def test_existing_different_directory_is_never_overwritten(project, tmp_path):
    dist = build_fixture(project)
    output = tmp_path / "candidate"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_bytes(b"previous user content")
    with pytest.raises(bundle.ReleaseBundleError, match="directory conflicts"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    assert sentinel.read_bytes() == b"previous user content"
    assert list(output.iterdir()) == [sentinel]


def test_existing_artifact_byte_conflict_is_preserved(project, tmp_path):
    dist = build_fixture(project)
    output = tmp_path / "candidate"
    manifest = bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    artifact = output / next(iter(manifest["artifacts"]))
    artifact.write_bytes(b"user's different candidate")
    with pytest.raises(bundle.ReleaseBundleError, match="artifact bytes conflict"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    assert artifact.read_bytes() == b"user's different candidate"


def test_old_version_is_rejected(project, tmp_path):
    dist = build_fixture(project, version="0.1.0a0")
    with pytest.raises(bundle.check_release.ReleaseCheckError, match="Version"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_artifact_filename_version_must_match_metadata_and_source(project, tmp_path, suffix):
    dist = build_fixture(project)
    artifact = next(path for path in dist.iterdir() if path.name.endswith(suffix))
    artifact.rename(artifact.with_name(artifact.name.replace("0.1.0a1", "0.1.0a0")))
    with pytest.raises(bundle.ReleaseBundleError, match="filename disagrees"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_source_changed_since_wheel_build_is_rejected(project, tmp_path):
    dist = build_fixture(project)
    (project / "src/rope_profiler/score.py").write_text("answer = 43\n")
    with pytest.raises(bundle.ReleaseBundleError, match="wheel source bytes differ"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_stale_metadata_readme_is_rejected_even_when_sources_match(project, tmp_path, kind):
    dist = build_fixture(project, **{f"{kind}_readme": "A former package description.\n"})
    with pytest.raises(bundle.ReleaseBundleError, match=f"{kind} metadata README payload differs"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_wheel_and_sdist_dependency_mismatch_is_rejected(project, tmp_path):
    dist = build_fixture(project, wheel_requirements=("torch>=2.4",), sdist_requirements=("torch>=2.8",))
    with pytest.raises(bundle.ReleaseBundleError, match="Requires-Dist declarations differ"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_wheel_and_sdist_extra_mismatch_is_rejected(project, tmp_path):
    dist = build_fixture(project, wheel_extras=("trainer",), sdist_extras=("wandb",))
    with pytest.raises(bundle.ReleaseBundleError, match="Provides-Extra declarations differ"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_dependency_and_extra_header_order_does_not_change_identity(project, tmp_path):
    dist = build_fixture(project, wheel_requirements=("torch>=2.4,<3", 'wandb>=0.19; extra == "wandb"'),
                         sdist_requirements=('wandb>=0.19; extra == "wandb"', "torch<3,>=2.4"),
                         wheel_extras=("trainer", "wandb"), sdist_extras=("wandb", "trainer"))
    manifest = bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")
    assert manifest["version"] == "0.1.0a1"


def test_new_source_file_missing_from_wheel_is_rejected(project, tmp_path):
    dist = build_fixture(project)
    (project / "src/rope_profiler/new.py").write_text("new = True\n")
    with pytest.raises(bundle.ReleaseBundleError, match="wheel source file set differs"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


@pytest.mark.parametrize("name", ["README.md", "pyproject.toml", "MANIFEST.in", "docs/RELEASING.md", "tools/check_input.py", ".github/workflows/ci.yml"])
def test_changed_public_build_input_is_rejected(project, tmp_path, name):
    dist = build_fixture(project)
    with (project / name).open("a") as stream:
        stream.write("\n# changed after build\n")
    with pytest.raises(bundle.ReleaseBundleError, match="sdist source bytes differ"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_sdist_cannot_omit_current_public_input(project, tmp_path):
    dist = build_fixture(project, omit_sdist={"docs/RELEASING.md"})
    with pytest.raises(bundle.ReleaseBundleError, match="sdist is missing"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_unmatched_generated_setup_cfg_cannot_add_build_configuration(project, tmp_path):
    dist = build_fixture(project, generated_setup_cfg=b"[options]\npackage_dir = changed\n")
    with pytest.raises(bundle.ReleaseBundleError, match="setup.cfg has unexpected"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_multiple_wheels_are_rejected(project, tmp_path):
    dist = build_fixture(project)
    shutil.copyfile(next(dist.glob("*.whl")), dist / "another.whl")
    with pytest.raises(bundle.check_release.ReleaseCheckError, match="exactly one wheel"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate")


def test_verify_only_rejects_changed_source(project, tmp_path):
    dist = build_fixture(project)
    output = tmp_path / "candidate"
    bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    (project / "src/rope_profiler/score.py").write_text("answer = 43\n")
    with pytest.raises(bundle.ReleaseBundleError, match="wheel source bytes differ"):
        bundle.prepare_release(project, output_dir=output, verify_only=True)


@pytest.mark.parametrize("name", ["manifest.json", "SHA256SUMS"])
def test_verify_only_rejects_changed_manifest_or_sums(project, tmp_path, name):
    dist = build_fixture(project)
    output = tmp_path / "candidate"
    bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    with (output / name).open("a") as stream:
        stream.write("\n")
    with pytest.raises(bundle.ReleaseBundleError, match="manifest conflicts|SHA256SUMS conflicts"):
        bundle.prepare_release(project, output_dir=output, verify_only=True)


def test_verify_only_requires_exact_mode(project, tmp_path):
    dist = build_fixture(project)
    output = tmp_path / "candidate"
    bundle.prepare_release(project, dist_dir=dist, output_dir=output)
    with pytest.raises(bundle.ReleaseBundleError, match="schema or mode"):
        bundle.prepare_release(project, output_dir=output, verify_only=True, mode="release")


def test_release_mode_requires_license_checks(project, tmp_path):
    dist = build_fixture(project)
    with pytest.raises(bundle.check_release.ReleaseCheckError, match="SPDX expression"):
        bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate", mode="release")


def test_release_mode_preserves_verified_license_in_wheel_and_sdist(project, tmp_path):
    pyproject = project / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('readme="README.md"', 'readme="README.md"\nlicense="MIT"'))
    (project / "LICENSE").write_text("A fixture license, for metadata validation only.\n")
    dist = build_fixture(project, license_expression="MIT")
    manifest = bundle.prepare_release(project, dist_dir=dist, output_dir=tmp_path / "candidate", mode="release")
    assert manifest["license"] == {"verified": True, "expression": "MIT", "files": ["LICENSE"]}
    assert "LICENSE" in manifest["source_files"]
