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
exact module edges, products, tools-version 6.2 and language mode 6; zero SwiftPM package
and external product dependencies; no custom source paths, resources, plugins, system or binary targets.
The graph includes the native backends authorized by the [native backend ADR](adr/add-vulkan-and-direct3d-native-backends.md)
and the standalone `SAGABenchmark` executable. The only linker settings are `m` on Linux for
`SAGAShaderIR` constant folding and `winmm` on Windows for the benchmark timer. Both the evaluated
settings (including platform conditions) and exact manifest declarations are checked; extra links
and unsafe flags still fail.

Imports use per-module allowlists with per-file additions for platform integration fixtures.
The scanner checks import declarations, not comment prose containing the words "import libraries".
All actual Sources/Tests files are walked independent of git ignore, except noncompiled
Finder .DS_Store metadata (symlinks are still rejected).

The native C ABI necessarily uses dynamic system entry points and pointer casts. The exact existing
source files, allowed escape patterns and reasons are recorded in `runtime-policy.json` under
`native_abi_exceptions`. A changed file digest, a copied file, or a new escape elsewhere fails;
there is no backend-wide exemption for unsafe casts or unchecked concurrency. Web reactor exports
retain their separate, exact ABI rule. The digest inventory is a review gate, **not proof of pointer
or concurrency safety**. In particular, the existing D3D12 process-global debug device remains a
concurrency risk for separate source review, not a safety claim granted by passing this audit.
Do not regenerate the inventory automatically after source changes.

The exact seven files in `Benchmarks/` are a separate opt-in comparison harness, not a SwiftPM
runtime dependency. Its Cargo manifest and Rust source are digest-pinned; other Rust/Cargo files
remain forbidden, including in the runtime tree. The docs audit does not build or fetch Rust
packages or mutate the read-only sibling checkout. See `Benchmarks/README.md` for the opt-in comparison recipe.

The pre-existing tracked root outputs `comp.spv` and `task.spv` (commit `47fc200`) are
also stored under exact filename and digest checks. They are not SwiftPM resources and have
no runtime reference. Their storage is not a provenance clearance or shader-conformance result;
unlisted shader binaries remain rejected. No shader output is executed by this audit.

Negative tests inject dependencies, graph changes, altered linker platform conditions, unsafe
settings/imports, changed/copied ABI files, C files and unlisted Cargo metadata. This is a regression
guard, not an adversarial Swift parser/security sandbox. Dependency policy changes require explicit
review; browser-owned internals are not SAGA dependencies.

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

Use Python **3.11 or newer** (`tomllib` is required by the configured corpus tools).
Offline/current checks: `python3 -B docs/tools/verify_docs.py`.
Full local reference audit: `SAGA_REFERENCE=/path/to/wgpu python3 -B docs/tools/verify_docs.py`.
Without SAGA_REFERENCE the reference-dependent tests explicitly skip; local content/hash
and policy tests still run. Skips do not prove reference freshness.

CI selects Python 3.12 explicitly and uses a commit-pinned Swiftly-based `setup-swift` v3.
The previous v2 action rejected both required Swift releases before any build: its static
release list stopped at 6.2.1. Keep the exact compiler checks and the matching Wasm SDK;
do not substitute a floating Swift release to make setup pass.

CI builds/tests macOS with exact Swift 6.2.4 and 6.3.2 releases, and compiles all libraries
for Wasm with the exact Swift 6.3.2 toolchain and matching SDK. The pinned SDK checksum is
from the official release record at https://www.swift.org/api/v1/install/releases.json,
saved in scripts/swift-web-sdk.json. SDK installation verifies the compiler version and
uses SwiftPM's `--checksum` verification. No toolchain/SDK is installed by the docs audit.

The workflow YAML is authored configuration, not a successful hosted CI run. Wasm
compilation is not browser execution or WebGPU conformance. `check-swift-test-log.py`
requires a nonzero successful Swift Testing summary even when XCTest prints zero tests.
Runnable native and browser diagnostics have separate build and execution recipes; this docs
audit neither executes them nor proves their GPU behavior.
