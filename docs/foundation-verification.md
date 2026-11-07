# SAGA-3 foundation verification

Date: 2026-11-06. This record is finalized with the results below after implementation;
no GPU, CTS, real shader translation or browser execution is claimed.

## Evidence boundaries

- SwiftPM package: tools-version 6.2, Swift language mode 6. Actual local native compilers are
  Apple Swift 6.3.3 (Xcode 26.6) and upstream Swift 6.3.2. Swift 6.2.x compiler unavailable locally;
  the authored CI matrix includes 6.2.4 but has not run on hosted CI.
- Native contracts are tested using actor test backends, not Metal execution.
- Wasm build uses upstream Swift 6.3.2 plus matching installed SDK; all library modules compile
  to wasm32-unknown-wasip1 objects. No executable, JS host integration or browser test exists yet.
- Package and sources have no external/FFI/binary implementation dependency. Rust reference
  remains pinned/read-only. Compiler inventory is source-level measurement, not parity evidence.
- Historical SAGA-2 source snapshots, inventory/report and docs-only guard remain preserved under
  the separate immutable fixture. Current documentation evidence is written to .verification.

## Final check results

| Check | Result |
| --- | --- |
| Apple Swift 6.3.3 native tests | 28 tests passed, 0 failures |
| Upstream Swift 6.3.2 native tests | 28 tests passed, 0 failures |
| Compile-time contracts | 6 passed: 1 positive control, 5 expected compiler rejections |
| Docs/current policy/pinned-reference audit | 23 passed, 0 skipped |
| Native strict build | passed, warnings as errors |
| Matched Swift 6.3.2 Wasm strict build | passed for all 11 library targets |

28 is the Swift Testing function denominator; parameterized functions additionally execute
18 compiler-negative pairs and 2 overlapping-teardown scenarios, not 18 translated shaders.
The inventory records 1418 lexical items, 1415 fixture files and 157 source hashes, all with
not-started/not-run compiler status.

[Apple test log](evidence/saga-3/native-swift-6.3.3-tests.log) ·
[Upstream test log](evidence/saga-3/native-swift-6.3.2-tests.log) ·
[Compile-time checks](evidence/saga-3/typecheck-contracts.log) ·
[Docs/reference audit](evidence/saga-3/docs-reference-audit.log) ·
[Native build](evidence/saga-3/native-build.log) ·
[Wasm build](evidence/saga-3/wasm-build.log) ·
[Pre-fix teardown failure](evidence/saga-3/teardown-regression-red.log).


Final native tests, typecheck checks, docs/reference audit and Wasm/native builds are recorded in
[the machine-readable result](foundation-results.json). The corresponding runtime verification
IDs are supplied in SAGA-3's completion note and session outcome. Build/test logs are linked there.

The Swift Testing denominator is authoritative; XCTest may print zero compatibility tests.
Compile-time checks deliberately require five invalid programs to fail, alongside a positive control.
Reference audit is run with SAGA_REFERENCE explicitly set; offline CI explicitly skips reference-
dependent freshness checks, not silently counts them as pass.

## Red/green checks and review

Descriptor tests initially failed with absent production modules before implementation.
Async error-scope regression failed with a lost error before the pending-scope fix.
Independent review reproduced a two-resource overlapping close/lose teardown race; both gated
regression cases failed when the pre-fix teardown count was restored temporarily. The fixed
source was restored immediately and then passed final verification. See
[public API review](public-api-review.md) for fixes and trusted boundary assumptions.

## Not executed / not implemented

Hosted CI runs, Swift 6.2.x local compilation, Metal GPU dispatch/readback/render/presentation,
real browser WebGPU or JS-host tests, semantic compiler translation, binary interchange compatibility,
expanded compiler fixture execution, differential shader tests and WebGPU CTS. Those are downstream
gates, not missing SAGA-3 translator implementations disguised as successes. No app or CLI product
was added, so no runnable launch configuration was registered. No commits or publication performed.
