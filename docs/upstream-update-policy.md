# Upstream Update Policy

llama.cpp changes several times per day. `llama.cpp-m` does not mirror every
commit. It publishes curated, immutable compatibility releases that pin
checkpoints proven by the complete platform and API gate.

## Repository Boundary

The wrapper and upstream project have separate identities:

- `llama.cpp-m` has its own semantic version.
- Every wrapper release maps to exactly one upstream tag and commit.
- The package coordinate is `mcpplibs:llamacpp`; consumers use
  `import llamacpp;`.
- mcpp-index points only to immutable `llama.cpp-m` releases, never a moving
  upstream branch.

`llama.cpp-m` owns the vendored source, `mcpp.toml`, module interface, generated
exports, API snapshots, import tools, runtime tests, CI, and release mapping.
mcpp-index owns only the Form A package descriptor, archive identity, workspace
registration, and cold consumer tests.

Consumer builds do not fetch upstream source. The release archive contains the
verified source under `third_party/llama.cpp/` and does not use a Git submodule.

## Version Model

The initial wrapper release is `0.1.0`, based on upstream `b10069`.

| Change | Required wrapper version change |
|---|---|
| Wrapper fix on the same upstream checkpoint | Patch |
| New upstream checkpoint or additive API/model/backend support | Minor |
| Module/feature rename or intentional public API removal | Major |

This policy applies before `1.0.0`: an upstream update cannot silently break a
published module contract.

Every release records:

- wrapper version;
- upstream repository, tag, and resolved commit;
- official archive URL and SHA-256;
- supported platforms and backends;
- material API, model, and backend changes.

Published tags, archives, and index entries are immutable.

## Cadence And Triggers

The scheduled workflow checks upstream weekly. It writes an informational
report only. It does not edit source, create a branch or issue, publish a
release, or change mcpp-index.

An update candidate is justified only by one or more of these triggers:

- security, crash, memory-safety, or inference-correctness repair;
- important model architecture or quantization support;
- a supported-user backend improvement;
- public API required by a consumer;
- mcpp, compiler, operating-system, or SDK compatibility repair;
- a checkpoint at least 30 days old when a newer checkpoint passes the complete
  release gate.

Routine upstream releases are limited to at most one per month. Security and
critical compatibility repairs are exempt. Commit count is informational and
never a release trigger by itself.

The weekly report can be run locally:

```bash
python3 tools/check_upstream.py
```

Approved trigger reasons can be attached explicitly. Age-based candidacy also
requires an explicit completed release gate:

```bash
python3 tools/check_upstream.py --trigger security-correctness
python3 tools/check_upstream.py --release-gate-passed
```

## Deterministic Import

`upstream.lock` stores the upstream repository, tag, resolved commit, official
archive URL, archive SHA-256, and initial import timestamp.

An update branch is named `update/upstream-<tag>`. The import procedure is:

1. Select a checkpoint for a documented policy trigger.
2. Resolve the official tag to its commit and record the archive SHA-256.
3. Import the official archive into `third_party/llama.cpp/`.
4. Regenerate exports and the source/API snapshot.
5. Inspect and resolve every API drift item.
6. Repeat check mode and require no generated or vendored drift.

Useful commands are:

```bash
python3 tools/import_upstream.py --lock upstream.lock --verify-tag
python3 tools/import_upstream.py --lock upstream.lock --check --verify-tag
python3 tools/gen_exports.py --check
python3 tools/audit_snapshot.py --check snapshots/<upstream-tag>.json
```

Vendored source remains patch-free by default. An unavoidable downstream patch
must be stored separately, documented in release notes, and covered by a
regression test.

Transient TLS, connection, timeout, rate-limit, and server failures use bounded
retries. Stable HTTP client errors, digest mismatches, source drift, compiler
failures, and test failures fail immediately.

## API Drift

The audit classifies:

- added declarations;
- removed declarations;
- changed signatures or types;
- macro changes;
- declarations needing a manual module-boundary decision.

Additions may be exported after compilation and audit. Removals and signature
changes require an explicit compatibility decision and release-note entry; they
cannot be accepted by regenerating the snapshot alone.

Macros remain outside the named-module export mechanism. Stable public value
macros may receive typed `export inline constexpr` replacements. Other macros
remain in the explicit skipped report with a reason.

## Verification Gate

Every compatibility change must pass:

- official tag and archive SHA-256 verification;
- deterministic import, export generation, and API snapshot checks;
- manifest parsing with the pinned mcpp version;
- cold package materialization;
- module import, compile, link, and runtime assertions;
- pinned GGUF load, decode, and sample.

The required platform behavior is:

| Platform | Required behavior |
|---|---|
| Linux x86_64 | CPU build and inference |
| Linux ARM64 | Released ARM64 mcpp, static musl helper, CPU inference |
| Windows x86_64 | CPU build and inference |
| macOS ARM64 | CPU build and inference |
| macOS ARM64 with Metal | Registration, allocation, offload, decode, sample |

Linux ARM64 must prove the build helper and consumer are ARM64 static ELF files
without `PT_INTERP`, and that compilation selects ARM rather than x86 sources.
Metal must prove device registration, buffer allocation, embedded shader use,
and positive layer offload. Compilation is not backend coverage.

A small pinned model covers portable regression CI. A release claiming a new
model architecture must add targeted evidence for that architecture. Large
models may run only during release-candidate or scheduled extended validation,
but their identity, command, backend, and observed result must be recorded.

No release claims all upstream models merely because the API compiled.

## Release Flow

1. Record the update trigger and checkpoint.
2. Import deterministically and resolve API drift.
3. Run the complete CI matrix and model-specific validation.
4. Update the release mapping and evidence.
5. Verify the tag, version, lock, vendored source, exports, and snapshot:

   ```bash
   python3 tools/check_release.py --tag v<version>
   ```

6. Create an immutable wrapper tag and release only after all gates pass.
7. Open a separate mcpp-index PR that appends the wrapper archive URL and
   SHA-256 and proves cold local-index consumer behavior.

The release workflow refuses to mutate an existing release and does not build a
replacement source archive. Form A consumes the GitHub tag archive pinned by
its SHA-256.

## Failure Ownership

A failing candidate remains unreleased and the last published package remains
current. Record the failure as one of:

- upstream regression;
- wrapper/API adaptation;
- mcpp runtime or toolchain defect;
- platform backend limitation;
- model-specific incompatibility;
- external download or CI infrastructure failure.

No failure mutates an existing tag, release asset, or index entry. A wrapper-only
repair on the same checkpoint receives a patch version. A repair that changes
the checkpoint receives a minor version.
