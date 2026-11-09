# SAGA-5 frontend coverage / provenance audit

## Delivered boundary

This is the **bounded audit harness**, not a compiler implementation or parity result.
The [accepted full-Naga ADR](adr/port-full-naga-on-swift.md) remains unchanged.
WGSL / GLSL / SPIR-V frontends and shared semantic/capability validation remain required.
Outputs/transforms/tooling remain in the parent scope; no all-to-all capability support is inferred.

Read-only reference: `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`.
Pinned WGSL spec: `4e829b310ba7381034b93b974eb1a6aabfb464f7`, existing
`docs/sources/wgsl-spec.bs` hash recorded in manifest metadata. No downloads or Rust execution.

## Denominator (overlapping inventory units, NOT semantic assertions)

| Kind | Count | Meaning |
| --- | ---: | --- |
| Compiler sources | 157 | Every source hash in immutable compiler inventory, including shared IR/proc/common/arena dependencies |
| SAGA-5 lexical inventory entries | 557 | All original SAGA-5-owned entries; IDs preserved with `inventory/` prefix |
| Fixture / container files | 1425 | Every pinned file in `naga/tests`, not just `tests/in` |
| Inline test anchors | 348 | Lexical `#[test]` function anchors in frontend/validator/test source; line-qualified to distinguish same names in nested Rust modules |
| Validation error anchors | 283 | Public error variant labels; trigger predicates/subcases are **not** expanded assertions |
| Named/capability obligations | 64 | 18 explicit rule families + 46 capability flags, each requiring positive/negative evidence before `tested` |
| Fuzz references | 7 | Pinned fuzz source/configuration metadata only; not executed fuzz runs |
| **Total records** | **2841** | Cannot be summed into percentage of semantic or full-Naga parity |

Fixture roles are **268 input**, **192 config**, **951 output**, **2 negative-input**,
**8 inline-test-container**, **4 upstream-auxiliary (Slang)**. Slang is not an invented
SAGA input frontend. Inputs: 183 WGSL, 49 GLSL, 36 SPIR-V assembly.
Expected outputs are denominator context, never frontend successes.
SPIR-V assembly requires an audited assembler-to-binary mapping before it can count as
binary execution; this harness currently refuses `tested` on assembly text.

Negative coverage includes `naga/tests/naga/{deep-expression,deep-template}.wgsl`,
`wgsl_errors.rs`, `validation.rs`, `spirv_capabilities.rs`, frontend parser tests and
inline validators. Source hashes retain macro-generated cases even where lexical test
anchors do not expand them. Macro invocations, arrays, conditional compilation, overloads,
capability combinations, diagnostic predicates and parameter subcases **still need an
expanded reviewed semantic mapping**. This file inventory is complete for the stated
pinned membership; it is deliberately not a claim of a complete normative assertion census.

## Independent membership and drift checks

[frontend_coverage.py](tools/frontend_coverage.py) derives expected membership from the pinned
Git tree and original immutable inventory, **not** the manifest being audited. Git object
bytes are loaded with one `cat-file --batch`, compared to read-only worktree bytes, then
SHA-256 hashed. It rejects dirty reference/HEAD drift, original inventory/scope hash drift,
missing/duplicate/extra denominator records, classification/hash/metadata drift, unknown
fields/status, and altered provenance. `--generate` refuses to overwrite existing manifests
or imported result references. Normal checking writes nothing.

The [manifest](frontend-coverage.json) has no native execution imports: **0 tested**,
**621 not-implemented**, **2220 not-run**, `compiler_parity: false`.
This is an inventory-audit PASS, not 2841 passing shaders.

Named requirements explicitly cover stage, type, uniformity, address-space, capability,
layout, expression typing, control flow, invalid raw IR handles, constants/overrides,
bindings, source/binary diagnostics, grammar, preprocessing, binary structure and budgets.
Metadata preserves actual WGSL/GLSL/SPIR-V option fields, 46 capability names/bit expressions,
validation flags, stage masks, subgroup operation sets, partial GLSL440 vs450/460,
and all TOML config paths. These are metadata, not executed option combinations.

## Status and trusted result-import contract

- `not-implemented`: no implemented feature/rule evidence imported.
- `not-run`: reference unit identified, no compatible native execution imported.
- `unsupported`: explicit nonempty reason; remains in denominator, does not count as pass.
- `tested`: requires actual compatible execution result files; never means complete parity.

To import genuine native results, update only a record's `status`, `reason`, `result_files`.
Immutable ID/source/hash/role/format/options metadata must not change. Result JSON under
`docs/evidence/saga-5/coverage/` must include:

- schema1, exact reference revision, `record_id`, `reference_sha256`, compatible `format`;
- kind `native-frontend-semantic`, actual `swift test` command, zero exit status,
  nonzero `tests_executed`, exact passing case name and matching count in hashed log;
- explicit stage, every format option field including defaults, known capabilities list, substantive assertion descriptions;
- polarity and matching expected/actual accept or reject; negative results also require
  typed diagnostic code and a valid UTF8 / binary-word / raw-IR-handle location range;
- current compiler implementation source paths/hashes to invalidate stale execution evidence.

Named/capability requirements need separate positive AND negative result files. Config,
expected output, source containers and auxiliary units cannot become shader executions.
Paths reject traversal/symlinks, artifacts are bounded to16MiB, missing logs/zero-test logs,
changed source hashes, demo/lexer-only result kinds and fake `covered` statuses fail.

This importer is a **trusted integrator evidence consistency check**, not remote attestation
or a malicious-code sandbox. It cannot establish that forged logs/metadata were actually
produced by a semantic compiler, nor infer exact semantic assertions from prose. The
integrator must review real test bodies, validator witness usage, diagnostic expectations,
options and capability combinations. Synthetic positive controls live only in temporary
unittest data and are NOT imported as project shader results. No API in this harness grants
full compiler parity: `compiler_parity: true` is always rejected, even after some tests pass.

## Provenance / licensing

[Per-path provenance](frontend-fixture-provenance.json) records all1425 fixture/container
paths, revisions, SHA-256, role, no-copy/no-modification status and per-file audit obligation.
**No shader/test source bytes were copied.** Root MIT/Apache notice hashes are references,
not a blanket fixture license clearance. Every row is `not-audited-for-copy` with empty
`copied_to`; the audit rejects root-license shortcuts and unaudited copies. Any future
copy requires an explicit separately reviewed schema/policy extension with original
copyright/license/notice/attribution/modifications and normative assertion mapping.
Original [provenance](provenance.md), compiler inventory and frozen baseline are unchanged.

## Reproduce / actual checks

From `/Users/vlad-prusakov/Developer/saga`:

```sh
python3 -B -m unittest discover -s docs/tools -p test_frontend_coverage.py -v
python3 -B docs/tools/frontend_coverage.py --reference /Users/vlad-prusakov/Developer/wgpu
/usr/bin/swift build --target SAGACompiler --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
```

**35 audit unit/integration controls PASS**, zero skips. Positive/negative controls include
dropped/duplicate denominator, fake coverage without results, pin/hash/option drift, unknown
status/fields, config/output misclassification, absent provenance, license shortcuts,
unaudited copy, wrong actual result/format/identity, missing typed negative diagnostic,
zero tests/fabricated count, stale logs/implementation, parser demo and false parity.
Real pinned-tree audit PASS; actual native strict SAGACompiler build PASS. The build is a
separate compatibility check, not shader execution. No Swift shader tests, fuzz run,
Wasm/browser/Metal/CTS execution were performed by this child.

[Tests](evidence/saga-5/coverage/tests.log) · [Audit](evidence/saga-5/coverage/audit.json) ·
[Build](evidence/saga-5/coverage/native-build.log) ·
[Commands/results](evidence/saga-5/coverage/checks.json) ·
[Immutable fingerprints](evidence/saga-5/coverage/preservation.json).
[Initial test-first red](evidence/saga-5/coverage/red.log) is the expected missing-module
failure before implementation, not an outstanding final failure.

## Configured SPIR-V execution

The [configured corpus runner](tools/verify_configured_frontend_corpus.py) can assemble the
36 pinned `.spvasm` inputs using a local SPIRV-Tools executable. It runs the same source
and output arguments as the pinned Naga snapshot test and records the executable version,
arguments and hashes for the executable, source and assembled binary. The Swift runner
checks the mapped binary and applies the fixture's frontend options before semantic validation.

```sh
python3 -B docs/tools/verify_configured_frontend_corpus.py \
  --reference ../wgpu --output .verification/frontend-configured \
  --spirv-as "$(command -v spirv-as)"
```

Without an assembler or when assembly is unavailable, the corresponding rows stay in the
denominator as unsupported. Read `results.json` for the individual parse and validation outcomes.
This run is separate from the coverage manifest's evidence importer.

The default assembler invocation may produce a newer SPIR-V header than the version
recorded in a fixture's leading comments. Keep that result in the configured corpus.
For version diagnostics, assemble a separate binary with `spirv-as --target-env` and
validate it under the corresponding universal and client profiles.

SPIRV-Tools v2026.4.rc2-0-gef96ed76 produced and validated the following
separate assemblies of the unchanged pinned fixtures. The SAGA results
use each fixture's configured frontend options and capabilities.

| Input | Target environment | Khronos validation | SAGA |
| --- | --- | --- | --- |
| `8151-barrier-reorder.spvasm` | SPIR-V 1.3 | Universal 1.3 and Vulkan 1.1 pass | Import and semantic validation pass |
| `bitcast-f16.spvasm`, `spec-constants.spvasm`, `f16-spv.spvasm` | SPIR-V 1.0 | Universal 1.0 and Vulkan 1.0 pass | Import and semantic validation pass |
| `sign-adjusted-ops.spvasm` | SPIR-V 1.0, selected for the version probe | Universal 1.0 and Vulkan 1.0 pass | Import and semantic validation pass |
| `shadow.spvasm` | SPIR-V 1.0 | Universal 1.0 passes. Vulkan 1.0 requires `Block` on the `StorageBuffer` runtime-array struct (`VUID-StandaloneSpirv-OpTypeRuntimeArray-04680`). | Import rejects the block decoration |
| `subgroup-operations-s.spvasm` | SPIR-V 1.3 | Universal 1.3 passes. Vulkan 1.1 requires a memory class bit with acquire-release ordering (`VUID-StandaloneSpirv-MemorySemantics-10870`). | Import rejects the barrier's empty memory class mask |
| `atomic_compare_exchange.spvasm` | SPIR-V 1.5 and 1.6 | Universal 1.5 and 1.6 and Vulkan 1.3 reject the compare-exchange instruction. Both masks set `WorkgroupMemory` with relaxed order (`VUID-StandaloneSpirv-MemorySemantics-10871` under Vulkan). | Import rejects the atomic instruction (`SAGA6002`) |
| `degrees.spvasm`, `inv-hyperbolic-trig-functions.spvasm` | SPIR-V 1.0 and 1.6 | Universal 1.0 and 1.6, Vulkan 1.0 and 1.3 reject the fragment entry point because it declares neither origin mode | Import rejects the missing `OriginUpperLeft` (`SAGA7001`) |

Under universal SPIR-V 1.6, validation rejects `8151-barrier-reorder`,
`bitcast-f16` and `spec-constants` for omitted entry interface variables.
It rejects `f16-spv`, `shadow` and `sign-adjusted-ops` because `BufferBlock`
is available through SPIR-V 1.3. The subgroup barrier has subgroup execution
scope, workgroup memory scope and no memory class bits. SAGA's subgroup barrier
IR couples subgroup execution with memory synchronization.

The compare-exchange source SHA-256 is
`6bcb1dce8db05a387b37281aaf5a4da04a44347a219f263af5d7f95f3a4e4b79`.
The SPIR-V 1.5 and 1.6 binary hashes are
`9e9ffd0ded3262d9ad98d8ddbd12bed845ba3d9d4539c0a5f5808246c5d07ebe` and
`784451c6a823cde3d430afbd399c46818ec3c4179b74952ae7dd589d2792fa50`.
A scratch control set both semantics operands to zero and kept Workgroup scope.
It passed universal and Vulkan validation. SAGA generated WGSL, MSL, HLSL and GLSL
and built a WebGPU compute artifact from the control.
[Vulkan memory semantics rule](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-StandaloneSpirv-MemorySemantics-10871).

The fragment origin results apply to these source and binary SHA-256 hashes at the pinned revision.

| Input under `naga/tests/in/spv/` | Pinned source | SPIR-V 1.0 binary | SPIR-V 1.6 binary |
| --- | --- | --- | --- |
| `degrees.spvasm` | `5676d4bef44bebba2da8cffb29b32fd3c969d9b69e6d4106fcd96fff77fc3df7` | `5f8ab641f6ecf7b4a035bba8efd19b523f87cba02547ad796fe5e230d0fa0c17` | `f11574154ea225ea86aa27b163b2e7ed172da9a0caea936a1d12971d650559b5` |
| `inv-hyperbolic-trig-functions.spvasm` | `7041107675db9130f565336d822a1f5e2cbc1697279055984f0be8ee41c29e62` | `8e81bcb1cfc7cc5de2149da0c8bfffce036d3ae71a7a9b2aa61039aba58df1aa` | `3bd8e1f09b87e95da8dcb4aef60fbc9ff5fb8b7e18dd06a397362f68c00c899a` |

In scratch SPIR-V 1.0 controls, adding only `OriginUpperLeft` let both shaders pass
Universal and Vulkan validation, SAGA import, and SAGA semantic validation. Controls
with only `OriginLowerLeft` passed Universal validation and were rejected by Vulkan
(`VUID-StandaloneSpirv-OriginLowerLeft-04653`) and SAGA (`SAGA6002`).
The configured corpus retains the original inputs and their pinned acceptance expectations.
[SPIR-V 1.0 validation rules](https://registry.khronos.org/SPIR-V/specs/1.0/SPIRV.html),
[Vulkan SPIR-V environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).

No board/workflow edits, dependencies, commits, publication or Rust writes. The full
SAGA-5 frontend/semantic DoD remains OPEN. This documentation/audit-only child adds no
application target and needs no Play configuration.
