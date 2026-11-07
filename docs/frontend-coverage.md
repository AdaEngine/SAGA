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

No board/workflow edits, dependencies, commits, publication or Rust writes. The full
SAGA-5 frontend/semantic DoD remains OPEN. This documentation/audit-only child adds no
application target and needs no Play configuration.
