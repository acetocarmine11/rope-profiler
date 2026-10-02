#!/usr/bin/env python3
"""Validate local packaging inputs/artifacts; never upload or publish anything.

Default checks support an unpublished package whose license decision is pending.
Use --require-license before release to require a LICENSE file and a valid SPDX
expression in project.license. This flag verifies metadata, not legal ownership.
"""
from __future__ import annotations

import argparse
import ast
from email.parser import Parser
import json
from pathlib import Path
import re
import tarfile
from typing import Any
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: supplied by the dev extra.
    import tomli as tomllib


class ReleaseCheckError(ValueError):
    """A local project or distribution does not satisfy the release contract."""


def _literal_version(path: Path) -> str:
    if not path.is_file():
        raise ReleaseCheckError(f"version source does not exist: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets, value = [statement.target], statement.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str) or not value.value:
                raise ReleaseCheckError("__version__ must be a nonempty string literal")
            return value.value
    raise ReleaseCheckError("version source must define a literal __version__")


def _license_files(project_root: Path, project: dict[str, Any]) -> list[Path]:
    patterns = project.get("license-files", ["LICENSE", "LICENSE.*", "LICENCE", "LICENCE.*"])
    if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
        raise ReleaseCheckError("project.license-files must be a list of file patterns")
    found: set[Path] = set()
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ReleaseCheckError("license-file patterns must remain within the project directory")
        matches = [path for path in project_root.glob(pattern) if path.is_file()]
        if "license-files" in project and not matches:
            raise ReleaseCheckError(f"license-file pattern does not match a file: {pattern}")
        found.update(matches)
    return sorted(found)


def _check_license(project_root: Path, project: dict[str, Any]) -> tuple[str, list[Path]]:
    expression = project.get("license")
    if not isinstance(expression, str) or not expression.strip():
        raise ReleaseCheckError("release requires a project.license SPDX expression string")
    try:
        from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
    except ImportError as exc:
        raise ReleaseCheckError("license expression validation requires packaging>=24.2 (install the dev extra)") from exc
    try:
        canonical = canonicalize_license_expression(expression)
    except InvalidLicenseExpression as exc:
        raise ReleaseCheckError(f"invalid project.license SPDX expression: {expression}") from exc
    files = _license_files(project_root, project)
    if not files:
        raise ReleaseCheckError("release requires at least one LICENSE/LICENCE file (or project.license-files entry)")
    for path in files:
        if not path.read_text(encoding="utf-8").strip():
            raise ReleaseCheckError(f"license file is empty: {path}")
    return canonical, files


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _check_metadata(text: str, project: dict[str, Any], version: str, artifact: Path,
                    license_expression: str | None = None) -> None:
    metadata = Parser().parsestr(text)
    for field, expected in (("Name", project["name"]), ("Version", version),
                            ("Requires-Python", project["requires-python"])):
        actual = metadata.get(field)
        if field == "Name":
            equal = _normalize_name(actual or "") == _normalize_name(expected)
        elif field == "Requires-Python":
            try:
                from packaging.specifiers import InvalidSpecifier, SpecifierSet
            except ImportError as exc:
                raise ReleaseCheckError("distribution validation requires packaging (install the dev extra)") from exc
            try:
                equal = actual is not None and SpecifierSet(actual) == SpecifierSet(expected)
            except InvalidSpecifier as exc:
                raise ReleaseCheckError(f"{artifact.name}: invalid Requires-Python specifier") from exc
        else:
            equal = actual == expected
        if not equal:
            raise ReleaseCheckError(f"{artifact.name}: {field} is {actual!r}; expected {expected!r}")
    if license_expression is not None and metadata.get("License-Expression") != license_expression:
        raise ReleaseCheckError(f"{artifact.name}: License-Expression does not match project.license")


def check_project(project_root: Path, *, require_license: bool = False,
                  dist_dir: Path | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = project_root / "pyproject.toml"
    if not config_path.is_file():
        raise ReleaseCheckError(f"pyproject.toml does not exist: {config_path}")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project = config.get("project", {})
    if project.get("name") != "rope-profiler":
        raise ReleaseCheckError("project.name must be rope-profiler")
    if "version" in project or "version" not in project.get("dynamic", []):
        raise ReleaseCheckError("project version must be dynamic and must not also be specified statically")
    attr = config.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {}).get("attr")
    if attr != "rope_profiler._version.__version__":
        raise ReleaseCheckError("dynamic version must use rope_profiler._version.__version__")
    version = _literal_version(project_root / "src" / "rope_profiler" / "_version.py")
    try:
        from packaging.version import InvalidVersion, Version
        Version(version)
    except ImportError:
        pass  # Source-only validation can operate before the dev extra is installed.
    except InvalidVersion as exc:
        raise ReleaseCheckError(f"invalid package version: {version}") from exc
    if not isinstance(project.get("requires-python"), str) or not project["requires-python"]:
        raise ReleaseCheckError("project.requires-python must declare the supported Python versions")
    readme = project.get("readme")
    readme_file = readme if isinstance(readme, str) else readme.get("file") if isinstance(readme, dict) else None
    if not readme_file or not (project_root / readme_file).is_file():
        raise ReleaseCheckError("project.readme must point to an existing README file")

    license_expression = None
    license_files: list[Path] = []
    if require_license:
        license_expression, license_files = _check_license(project_root, project)
    artifacts: list[str] = []
    if dist_dir is not None:
        dist_dir = dist_dir.resolve()
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ReleaseCheckError("dist directory must contain exactly one wheel and one .tar.gz source distribution")
        with zipfile.ZipFile(wheels[0]) as archive:
            entries = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise ReleaseCheckError("wheel must contain exactly one .dist-info/METADATA")
            _check_metadata(archive.read(entries[0]).decode("utf-8"), project, version, wheels[0], license_expression)
        with tarfile.open(sdists[0], mode="r:gz") as archive:
            entries = [entry for entry in archive.getmembers() if entry.name.count("/") == 1 and entry.name.endswith("/PKG-INFO") and entry.isfile()]
            if len(entries) != 1:
                raise ReleaseCheckError("source distribution must contain exactly one top-level PKG-INFO")
            stream = archive.extractfile(entries[0])
            if stream is None:
                raise ReleaseCheckError("cannot read source distribution metadata")
            _check_metadata(stream.read().decode("utf-8"), project, version, sdists[0], license_expression)
        artifacts = [str(path) for path in [*wheels, *sdists]]
    return {
        "project": project["name"], "version": version,
        "requires_python": project["requires-python"],
        "license_required": require_license,
        "license_expression": license_expression,
        "license_files": [str(path.relative_to(project_root)) for path in license_files],
        "artifacts": artifacts, "publishes": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--require-license", action="store_true")
    arguments = parser.parse_args()
    try:
        report = check_project(arguments.project_root, require_license=arguments.require_license,
                               dist_dir=arguments.dist_dir)
    except (ReleaseCheckError, OSError, SyntaxError, tomllib.TOMLDecodeError, zipfile.BadZipFile, tarfile.TarError) as exc:
        parser.exit(1, f"Local release check failed: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
