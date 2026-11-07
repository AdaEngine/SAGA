# SAGA-15 blocked: concurrent same-session source collision (stream68c3)

Status: **incomplete / blocked**, not full SAGA-15 success. 2026-11-06T23:29Z.

The inherited request was executing in two overlapping model streams of the SAME session `session-dc0e5ed5-1cb5-4b1b-8877-4707990c1d64`. This stream observed and preserved actual decoder/lowering work already present at23:23Z, fixed the macOS14-unavailable UTF8 string initializer with byte round-trip validation, added malformed decoration/mode/UTF8 and AccessChain/composite/canonical-loop regression tests. A separate overlapping stream modified the same decoder, extended SPIRVFunction.swift, then at23:27:16 wrote FunctionLowering.swift defining incompatible duplicate SPVBlockRecord, SPVValue and SPVFunctionLowerer. At23:27:57 that other stream also reported the collision in this same session history and stopped. Its report was independently observed through parent history. This is not a speculative API gap: strict native build demonstrably fails on duplicate types.

## Preservation

No deletion, rename, cleanup, task/workflow mutation, commits/publication or shared API edits. Both competing source variants remain. All three current SPIRV source files and current tests are snapshotted alongside this report as conflict-stream-68c3-*.txt. Never choose/remove a variant based solely on file timestamps. The decoder currently contains both `localPointerTypes` and `spillPointerTypes`, proving overlapping API expectations; retain/reconcile the corresponding decoder with the chosen lowering.

## Actual verification

- Original expected missing API RED: red.log (historical in this evidence directory).
- Focused strict test attempt in this stream first failed due unowned concurrent ShaderSemanticValidationTests.swift missing `try`; no edits to that file.
- `regressions-red.log`:10 tests actually ran, one negative assertion failed (missing fragment OriginUpperLeft accepted). Fixed that entry mode check; other old regressions passed.
- `access-red.log`:13 tests ran,12 passed, one assertion failed: AccessChain emitted `.access` for constant struct member, not `.accessIndex`. This is not final passing evidence. CompositeConstruct/Extract tests ran successfully on current partial lowering.
- `loop-red.log`: test compilation prevented by duplicate SPVBlockRecord/SPVValue/SPVFunctionLowerer declarations from concurrent sources. No canonical loop success claimed.
- Final exact strict build: `/usr/bin/swift build --target SAGACompiler --scratch-path .build-spirv -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete` exit1; [full log](conflict-build-stream-68c3.log). No successful final verification evidence ID.

## Implemented partial surface (not accepted full parity)

Public `SPIRVFrontend:BoundedShaderFrontend`, Module=IRModule, init(options:SPIRVFrontendOptions=.init(),limits:ShaderLimits=.default). Actual header/endian/version/schema/wordcount/order checking, sparse ID definitions (UInt32.max bound needs no bound-sized allocation), type/constant/global declarations, functions with parameters/calls, loads/stores, scalar/vector arithmetic, selection and phi spills in SPIRVFunction.swift. Some AccessChain/construct/extract and canonical loop work appeared concurrently and is NOT final verified. Output is raw shared IR, not validated witness. Unknown opcodes fail typed ShaderFailure.diagnostics with original binary-word span. strictCapabilities=false explicitly unsupported; IO/coordinate wrappers unavailable and rejected rather than silently ignored.

## Precise residuals

1. Reconcile duplicate source variants and mixed decoder fields under ONE active executor; both streams must be idle first. Preserve snapshots; no broad untracked cleanup.
2. Full supported reference capability/options/opcode mapping and validated expression/SSA dominance/evaluation semantics. Current type storage is not proof of source validity.
3. General structured loops/switch, CFG dominance/spilling and arbitrary phi assignment; forward call topological mapping; unreachable constructs.
4. Input/output entrypoint wrapper lowering, Position coordinate conversion and all stage/builtin/decorations; row/column-major matrix layouts, robust layout computation. strict=false remains rejected.
5. Image/sampler/texture operations, atomics/barriers/subgroups/ray/extensions/GLSL.std.450, specialization constants/overrides, complete conversions/math/composites, memory operand variants, full debug/source metadata.
6. Complete reference fixture and malformed/fuzz/semantic matrix and provenance. Existing deterministic malformed400-case fixture mutation test is not exhaustive fuzz or reference parity. No GPU/browser/CTS or emitted shader execution.

## Parent communication

One collision report attempt to parent `session-e26a2eeb-8007-4c5e-a65a-5eccd466c9e6` returned `message_chain_limit` (eight deliveries). It was not delivered. Do not retry or bypass that cap. Parent independently read same-session collision report through history at23:28Z. This durable report and typed blocked session outcome are the handoff; no claim of delivery receipt. Do not accept_result/mark SAGA-5 or SAGA-15 done from these files.

Reference `/Users/vlad-prusakov/Developer/wgpu` remains READ ONLY at pinned a55352911bb26a1b33bc8d5fc03bdebb0a967f5a. Grammar metadata attribution/hash in grammar-reference.json. Independently authored word fixtures, no Rust runtime/FFI/dependencies/Cargo builds. Library has no runnable target/Play configuration.
