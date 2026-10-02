# Release procedure

The initial public version is `0.1.0`, with matching source tag `v0.1.0`.
This guide describes how to prepare and publish a fixed candidate. A source
push, tag, or successful build does not establish that PyPI contains the
package; verify the published index metadata and installed files separately.

## Prepare one candidate

1. Confirm the project license and attribution. Include the complete license
   file and a `project.license` SPDX expression. The repository's release checks
   require both before preparing an upload candidate. These metadata checks do
   not verify copyright ownership.
2. Set `src/rope_profiler/_version.py`, the changelog, README version links, and
   project documentation URL to the intended release. A plain `0.1.0` identifies
   the first public version; support and performance claims still follow the
   documented validation evidence.
3. Run the tests and documented example. Local CPU results, actual GPU
   measurements, and online CI results have separate evidence statuses.
4. Build into a clean version-specific directory, check the package metadata,
   and freeze the exact files with their source fingerprints:

   ```bash
   python -m pytest -q
   python examples/train_monitor.py --regions --full-distances
   python tools/check_release.py --require-license
   python -m build --outdir dist/0.1.0
   python -m twine check dist/0.1.0/*
   python tools/prepare_release.py --dist-dir dist/0.1.0 \
     --output-dir release-candidates/0.1.0 --mode release
   python tools/prepare_release.py --output-dir release-candidates/0.1.0 \
     --mode release --verify-only
   ```

   `prepare_release.py` copies the wheel/sdist unchanged and records their
   hashes, size, version, and source-file hashes in `manifest.json` and
   `SHA256SUMS`. It never builds, uploads, commits, or tags. A conflicting output
   directory is rejected; identical repeated preparation is safe. Verification
   writes no files. Draft mode is available for local review while a license
   decision remains pending, and its bundles cannot be used for publication.
5. Install the candidate wheel in a fresh environment outside the checkout and
   run `python -I /path/to/tools/wheel_smoke.py --expected-version 0.1.0`.
   This checks installed RECORD hashes/sizes, import location, version, core
   scores, and configuration recovery. Keep the results with the candidate
   checksums. Local environments, task caches, and run logs are excluded from
   the distribution.

## Build the candidate from the public source tag

Publish the reviewed source to the standalone
[`acetocarmine11/rope-profiler`](https://github.com/acetocarmine11/rope-profiler)
repository, create its matching tag, and run `release.yml` with
`release_tag=v0.1.0`. The workflow checks CPU compatibility, tag/version, and
license, builds once, tests the wheel as a consumer, and retains the complete
`release-candidate` artifact.

Dispatch the build from the release tag so the run's commit matches the
candidate source:

```bash
gh workflow run release.yml --ref v0.1.0 -f release_tag=v0.1.0
```

The upload workflow requires a completed, successful `release.yml` run from
that exact repository and commit.

Save the successful run ID and download that artifact before its 90-day
retention period expires. The files from this workflow run become the upload
candidate. Verify them and attach those same files and checksums to the matching
GitHub Release. Preserve earlier local and development candidates without
changing their contents.

## Configure account access once

Configure PyPI and TestPyPI Trusted Publishers using these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `rope-profiler` |
| GitHub owner | `acetocarmine11` |
| GitHub repository | `rope-profiler` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` on PyPI; `testpypi` on TestPyPI |

For a first project upload, configure a pending publisher. Account login and
any required account verification occur directly on the index website. No
password or API token is needed in the repository or chat. A project-page 404
does not reserve its name. See the official
[PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

Create the matching GitHub environments and protect them according to the
release policy. Only the publishing job has `id-token: write`; the retrieval job
uses read-only repository/actions permissions.

## Publish the candidate

For an optional TestPyPI upload, run `publish.yml` with `release_tag=v0.1.0`, the successful release workflow's
`candidate_run_id`, and `index=testpypi`. The workflow downloads that existing
candidate, verifies its manifest/checksums against the tag's source and license,
and stages only the wheel/sdist. It never rebuilds a package.

To publish to PyPI, use the verified candidate with `index=pypi`. When using
both indexes, keep the **same tag and candidate run ID** so they receive identical
package bytes. A release can use the local and GitHub consumer checks before
its first production upload. Supply dependencies from the intended production environment
when checking TestPyPI; the test index need not contain them all. Install the
specific uploaded version without resolving dependencies against TestPyPI.

```bash
gh workflow run publish.yml --ref v0.1.0 -f release_tag=v0.1.0 \
  -f candidate_run_id=BUILD_RUN_ID -f index=pypi
```

After each upload, read the index metadata independently, compare artifact
hashes with `SHA256SUMS`, install the exact version in a fresh environment, and
run the consumer smoke check. Preserve the workflow run URLs and executed
results as release evidence. If a published version needs a change, increment
the version and prepare a new candidate.
