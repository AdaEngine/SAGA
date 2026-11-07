# SAGA-4 Metal backend diagnostic verification

Date: 2026-11-06. Checkout: `/Users/vlad-prusakov/Developer/saga`.
Scope: **real adapter → device → queue → compute → readback**, not full WebGPU/WGSL conformance.
[Backend/retirement ADR](adr/select-initial-backends.md) · [machine results](metal-results.json).

## Delivered

- OS Metal adapter discovery and a real MTLDevice/MTLCommandQueue; typed no-adapter error.
- Actor-owned native buffers, zero initialization, staged uploads, MSL compute pipeline,
  GPU blit copies and immutable read snapshots. No result-generating CPU fallback.
- Portable optional command SPI and shared Core validation/pinning/queue/error-scope path.
  Existing browser hosts remain valid; unimplemented commands throw unsupported.
- Typed bounded GPU completion timeout. Core becomes terminal while native in-flight objects
  remain quarantined until actual callback. Owner teardown clears buffer/pipeline registries;
  a single immutable native diagnostic pipeline cache lives with its backend.
- Native executable and a headless macOS app wrapping the exact same built Mach-O executable.
- Exact dependency/target/import policies evolved for the single named diagnostic executable;
  regression checks still reject arbitrary executables, external dependencies, foreign code,
  native imports into Core and concurrency escapes. Frozen SAGA-2 hashes remain intact.

## Actual environment

- Apple M3 Pro, 18 GPU cores, Metal 4 reported by system_profiler; actual dispatch succeeded.
- macOS 26.6.1 (25G76), arm64.
- Native Apple Swift 6.3.3; tools-version 6.2, language mode 6, complete concurrency,
  warnings-as-errors. This is not a local Swift 6.2.x compiler verification.
- Web compilation: upstream Swift 6.3.2 + matching swift-6.3.2-RELEASE_wasm SDK.
- Read-only Rust reference HEAD `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`, status clean.

## Acceptance evidence

| Gate | Result | Durable log |
| --- | --- | --- |
| Native strict build | PASS | [build](evidence/saga-4/native-build.log) |
| Swift Testing (not XCTest compatibility zero) | **39 tests PASS**, no disabled tests on this host | [tests](evidence/saga-4/native-tests-final.log) |
| Same suite with Metal API Validation | **39 PASS** | [validation tests](evidence/saga-4/metal-validation-tests.log) |
| GPU boundary sizes | every element checked for 1, 63, 64, 65, 4099 | same test logs |
| Large actual GPU dispatch/readback | **1,048,576 elements; 0 mismatches** | [run](evidence/saga-4/gpu-run-final.log) |
| Same large run with Metal API Validation | PASS, no validation errors | [validation run](evidence/saga-4/gpu-api-validation.log) |
| Compile-time contracts | **6 PASS** (1 positive, 5 expected compiler rejections) | [contracts](evidence/saga-4/typecheck-contracts.log) |
| Docs/policy/pin/reference reproduction | **24 PASS, 0 skipped** | [audit](evidence/saga-4/docs-audit.log) |
| All 11 library modules in Wasm | PASS through four library roots | [Wasm](evidence/saga-4/wasm-build.log) |
| Dependency/module policy | 13 targets, **0 external dependencies** | [policy](evidence/saga-4/dependency-policy.log) |
| Native linked library inspection | only OS/Swift libraries; no foreign runtime | [otool](evidence/saga-4/linked-libraries.log) |
| Actual packaged .app launch | 4099 elements, 0 mismatches | [app launch](evidence/saga-4/app-launch.log) |
| CLI negative paths | missing adapter, zero count, excessive count, zero timeout → expected exit 1 | [negatives](evidence/saga-4/cli-negative.log) |

The test-first [red log](evidence/saga-4/red.log) records the expected missing command/pipeline
API compile failure before implementation. It is not a remaining failing check.

### Safety cases exercised

Negative command tests validate flags, aliases, zero/oversize counts, foreign/stale resource
identity, UInt64.max/overflow-safe ranges, alignment and read/write/copy usage before backend
entry. Native tests additionally cover unknown pipeline failure without queue poisoning,
zeroed buffers, read-before-write snapshot ordering, repeated owner teardown, cancellation,
and in-flight destruction. Existing error-scope/order/loss/overlapping-close regressions pass.

A real Metal command's callback is deliberately gated by an internal test-only seam: the
completion deadline expires, Core.close returns, owner registries are empty but the batch
still retains both native buffers; releasing the callback then retires that batch. This tests
the timeout path without submitting a deliberately infinite shader. It is **not** evidence
of actual GPU removal, driver reset, hardware hang or recovery. Completion-latch tests also
cover early completion, late completion and actual timer expiry with no callback.

## Reproduce

Native:

```
/usr/bin/swift build --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
/usr/bin/swift test --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
/usr/bin/swift run --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete SAGAMetalDiagnostic --count 1048576
MTL_DEBUG_LAYER=1 .build-apple/debug/SAGAMetalDiagnostic --count 1048576
SAGA_REFERENCE=/Users/vlad-prusakov/Developer/wgpu python3 -B docs/tools/verify_docs.py
```

Matched Wasm library compilation (not browser execution; the native CLI is intentionally
not linked into a web app):

```
for target in SAGA SAGACompiler SAGAMetal SAGAWebGPU; do
  /Users/vlad-prusakov/Library/Developer/Toolchains/swift-6.3.2-RELEASE.xctoolchain/usr/bin/swift build --target "$target" --scratch-path .build-wasm --swift-sdk swift-6.3.2-RELEASE_wasm -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
done
```

## Verified Play recipe

Configuration `8186ee3f-420e-4762-a53e-bab7141e8cc0`, target `SAGAMetalDiagnostic`, macOS,
checkout above, working directory `.`. The recipe is persisted for this chat through
session.launch.configure; [portable recipe record](metal-launch-recipe.json).

Build commands, both checked with the exact executables:

```
/usr/bin/swift build --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
/usr/bin/python3 scripts/package-metal-diagnostic.py
```

App path: `.build-play/SAGAMetalDiagnostic.app`; bundle ID `team.saga.MetalDiagnostic`.
It is intentionally **headless**: Play runs 4099 elements and appends success/failure to
`.build-play/saga-metal-diagnostic.log`, not a graphics window. The app was actually opened
with `/usr/bin/open -n -W .build-play/SAGAMetalDiagnostic.app`; new log bytes were checked.
No screenshot is applicable to a headless compute diagnostic. App launch success alone
must not be interpreted as compute success: read the report. Requires a Mac/Metal device,
Swift and system Python; no package install, network or CPU fallback.

Runtime metadata limitation: adding returned verification IDs to the launch card was rejected
as `launch_configuration_invalid` even though tools returned current-turn successful IDs;
session history showed the underlying persisted tool events lack that augmented field.
The **same verified recipe was saved successfully without optional ID metadata**, with build/run
logs archived here. This does not change any build/GPU check or fabricate a launch-run record.

## Explicit limits and follow-up

MSL here is a fixed backend diagnostic, **not WGSL/WebGPU/independent compiler conformance**.
The command ABI is narrow (UInt32 input/output/count), not general bind groups/encoders.
Mapping/unmapping, textures/render/presentation, full Naga, browser JS execution and CTS are
not implemented by this child. Browser-owned internals remain outside project dependencies.
Hosted CI and local compiler 6.2.x are not verified. The CI YAML selects all library roots
for web compilation and is configuration, not an executed hosted run.

GPU completion is bounded; a synchronous OS driver operation (e.g. MSL compilation) is not
force-cancelled by that deadline. A permanently missing callback retains native quarantine
until process termination; no claim of driver-hang recovery is made. Negative no-adapter
was injected intentionally (this machine actually has a GPU). SAGA-1 stays open. No Rust
checkout changes, commits or publication. Earlier approval timeouts are historical and resolved.
