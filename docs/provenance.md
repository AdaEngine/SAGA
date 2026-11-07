# Provenance and license audit

Audit date: 2026-11-06. Documentation-only SAGA-2. Technical licensing inventory, not legal advice or blanket redistribution clearance.

## Verified inputs and proposed pins

| Input | Verified revision / evidence | Role |
|---|---|---|
| Local wgpu | `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`; `git -C ../wgpu rev-parse HEAD`, clean porcelain at inspection | Read-only implementation reference, not runtime or normative authority |
| GPUWeb WebGPU + WGSL | **proposed pin** `4e829b310ba7381034b93b974eb1a6aabfb464f7`; GitHub commit API and pinned raw source fetched successfully | Same repo revision for both specs; Editor's Draft, not a final Recommendation assertion |
| Independent GPUWeb CTS | **proposed pin** `d116bcccd1833369fa3715a6df503c0dd9b0d7fa`; GitHub commit API + complete recursive tree fetched | Independent suite to route into Swift later, no tests run |
| wgpu's CTS reference | `b009414b0d1c9c57af3ff0b37477c0bcf736428c` from local `cts_runner/revision.txt` | Distinct reference pin; remote existence/content not verified here, not selected for SAGA |

Verification sources: [GPUWeb commit metadata](sources/gpuweb-revision.json), [CTS commit metadata](sources/cts-revision.json), [CTS recursive tree](sources/cts-tree.json), [download URLs and SHA-256](sources/fetch.json), [all source fingerprints](source-hashes.json). These are point-in-time pins, not claims that a moving branch will still point there.

- [Pinned WebGPU source](https://github.com/gpuweb/gpuweb/blob/4e829b310ba7381034b93b974eb1a6aabfb464f7/spec/index.bs), [local bytes](sources/webgpu-spec.bs); readable named anchors use <https://gpuweb.github.io/gpuweb/>.
- [Pinned WGSL source](https://github.com/gpuweb/gpuweb/blob/4e829b310ba7381034b93b974eb1a6aabfb464f7/wgsl/index.bs), [local bytes](sources/wgsl-spec.bs); readable named anchors use <https://gpuweb.github.io/gpuweb/wgsl/>.
- [Pinned CTS](https://github.com/gpuweb/cts/tree/d116bcccd1833369fa3715a6df503c0dd9b0d7fa), [README](sources/cts-README.md). The CTS/spec pair is verified to exist, not proven mutually aligned or compatible by execution. Resolve revision skew before freezing release queries.

The required `webgpu-specs` skill was read. Its helper invokes Cargo metadata and writes into wgpu/target; it was intentionally **not run** to respect read-only scope. Equivalent immutable source downloads were saved only under saga/docs/sources. No Rust command/test/CTS was run.

## Materials actually retained

| Material | Observed license / attribution | Action and unresolved obligation |
|---|---|---|
| GPUWeb `.bs` snapshots, generated inventory derived from them | W3C Software and Document License; repository distinguishes software contributions under GPU for the Web 3-Clause BSD | [Complete repository notice](sources/gpuweb-LICENSE.md) retained alongside unmodified snapshots. Preserve pre-existing notices; generated inventory is an adapted extraction, not spec text replacement |
| CTS README, tree metadata, license text | BSD 3-Clause; Copyright 2019 WebGPU CTS Contributors | [Full CTS notice](sources/cts-LICENSE.txt) retained. No CTS test code vendored/executed; future test copies retain notices/disclaimer in source/binary docs; no endorsement |
| wgpu MIT/Apache notices and CTS pin text | Reference repository dual MIT OR Apache-2.0 at root; Copyright 2026 gfx-rs developers in MIT text | [MIT](sources/wgpu-LICENSE.MIT), [Apache-2.0](sources/wgpu-LICENSE.APACHE) retained unmodified. No Rust runtime/code/tests copied. Future adaptations require applicable file notices, attribution, license selection and Apache NOTICE/patent review if chosen |
| Original SAGA prose/scripts | Newly authored docs and extraction/checking scripts | SAGA project outbound license **unknown / not selected**. No license grant or release choice made here |

Attribution / change notice: This documentation includes material copied from or derived from WebGPU and WebGPU Shading Language at the pinned GPUWeb URLs above. Copyright © World Wide Web Consortium; see [W3C notice and terms](sources/gpuweb-LICENSE.md). Spec source snapshots are unmodified; inventory and explanatory methodology are newly generated/adapted, not official W3C text. CTS README/license are unchanged upstream snapshots; metadata is selected API response data.

## Reference test inventory (paths checked locally)

At the fixed wgpu revision: `tests/tests/wgpu-gpu`, `tests/tests/wgpu-validation`, `tests/tests/wgpu-compile`, `tests/tests/wgpu-dependency`, `tests/tests/wgpu_surface.rs`, `naga/tests/in`, `naga/tests/out`, `cts_runner`, `deno_webgpu`, `wgpu-types/src/features.rs` exist. Older names in reference testing prose do not necessarily exist in this checkout. Rust GPU/validation tests may inform Swift test design but must be rewritten/adapted; Naga snapshots are compiler-specific, not WGSL truth. Each adapted fixture must record original path/revision/copyright/license, modifications and normative assertion IDs. Per-fixture exceptions and transitive asset licenses are **unknown until inspected**, not covered by a root-license shortcut.

## Native parity audit

[Native manifest](native-inventory.json) enumerates 59 literal native feature flags in `wgpu-types/src/features.rs` at the fixed revision. This is an initial feature register, **not all native API methods/types or full parity**. `FeaturesWebGPU` and the current spec can differ by revision; subgroup/atomic/extension overlap needs semantic mapping. Native adapter enumeration, polling, surfaces, tracing/caches, raw handles, passthrough shader and unsafe extensions need a later API-level parity inventory if approved. Status remains **proposed**, denominator unknown until user scope approval.

## Release-blocking open audit items

- Select SAGA outbound license explicitly; do not infer it from reference.
- Audit each copied/adapted fixture and every future JS host/backend/compiler dependency, including transitive licenses, NOTICE and redistribution/patent conditions.
- Preserve source hashes, attribution and change notices as pins evolve; regenerate inventory with reviewed ID migration.
- Verify CTS expanded enumeration and spec/CTS alignment; current recursive tree verifies directories, not the number of parameterized cases or conformance.
- No runtime dependency/link graph exists yet, so absence of prohibited runtime dependencies is currently a design constraint, not a completed binary audit. The [accepted platform ADR](adr/use-metal-and-browser-webgpu.md) separates prohibited SAGA project dependencies from browser-owned WebGPU internals. Local snapshot/hash verification is not a fresh network request or release license clearance.

[Conformance](conformance.md) · [Decision](adr/implement-independent-webgpu.md) · [Verification](verification.md).
