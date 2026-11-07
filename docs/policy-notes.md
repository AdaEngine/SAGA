# Foundation policy and verification

Date: 2026-11-06. SAGA-3 adds a current runtime policy without rewriting SAGA-2 history.

## Baseline preservation

`baseline/saga-2` contains byte-for-byte original test_docs.py, test_naga_scope.py,
verify_docs.py and verification-results.json. hashes.json pins them. Original
Package.swift rejection assertions remain in the fixture, and their scope guard is
executed by the current tests. Existing content tests still validate pinned source hashes,
reproducible 750-unit inventory, 59 native flags (reference audit), roadmap DAG, links,
accepted ADR, historical decisions/report, and required outputs.

The current Package.swift assertion deliberately uses `runtime-policy.json` instead.
`tools/verify_docs.py` runs the current tests and audit and writes only
`.verification/docs-current.json`. It never rewrites historical verification-results.json.
The archived original validator is not a command to run against today's runtime checkout.

## Runtime boundary policy

`python3 -B docs/tools/runtime_policy.py` inspects `swift package dump-package`:
exact module edges, library products, tools-version 6.2 and language mode 6; zero package
and product dependencies; no custom source paths, resources, linker/build settings,
plugins, system or binary targets. Imports use a per-module allowlist; Metal is isolated
to SAGAMetal. All actual Sources/Tests files are walked independent of git ignore, except noncompiled
Finder .DS_Store metadata (symlinks are still rejected).
Unsafe Sendable/concurrency and C entrypoint/link escapes are rejected. Negative tests
inject dependencies, graph changes, unsafe settings/imports, C files and Cargo metadata.
This is a regression guard, not an adversarial Swift parser/security sandbox. Dependency
policy changes require explicit review; browser-owned internals are not SAGA dependencies.

## Measured compiler inventory

`compiler-feature-inventory.json` is generated from the pinned read-only reference by
`python3 -B docs/tools/generate_compiler_inventory.py /path/to/wgpu --check`.
It enumerates public type/enum declarations, public fields (including options), argh CLI arguments, constants
(including capability bitflags), source units, and all input/expected-output fixture files
with hashes. The 16 accepted scope components have separate owners and source locations.
All implementation statuses remain not-started/not-run: contracts do not implement Naga.
There is no claim of atomic semantic completeness, arbitrary all-to-all support, expanded
fixture case count, serialization compatibility, or executed translation. Other private options, enum variants, implicit validation rules and supported capability combinations
still need semantic expansion in SAGA-5/6/11. Source hashes pin auxiliary arena/naga-types/
serialization sources even when they have no separate lexical item.

## Commands and CI

Offline/current checks: `python3 -B docs/tools/verify_docs.py`.
Full local reference audit: `SAGA_REFERENCE=/path/to/wgpu python3 -B docs/tools/verify_docs.py`.
Without SAGA_REFERENCE the reference-dependent tests explicitly skip; local content/hash
and policy tests still run. Skips do not prove reference freshness.

CI builds/tests macOS with exact Swift 6.2.4 and 6.3.2 releases, and compiles all libraries
for Wasm with the exact Swift 6.3.2 toolchain and matching SDK. The pinned SDK checksum is
from the official release record at https://www.swift.org/api/v1/install/releases.json,
saved in scripts/swift-web-sdk.json. SDK installation verifies the compiler version and
uses SwiftPM's `--checksum` verification. No toolchain/SDK is installed by the docs audit.

The workflow YAML is authored configuration, not a successful hosted CI run. Wasm
compilation is not browser execution or WebGPU conformance. `check-swift-test-log.py`
requires a nonzero successful Swift Testing summary even when XCTest prints zero tests.
There are no runnable products here, and no Play recipe is registered.
