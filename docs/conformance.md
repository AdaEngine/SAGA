# Conformance baseline and coverage methodology

Date: 2026-11-06. Scope: SAGA-2 documentation, not a working WebGPU implementation.
[Roadmap](roadmap.md) · [Inventory register](inventory.md) · [Machine manifest](inventory.json) · [Native register](native-inventory.json) · [Provenance/pins](provenance.md).

## Authorities and boundaries

**Known unresolved query-resolve divergence (2026-11-10):** the pinned
WebGPU snapshot requires `firstQuery` to be less than the query-set count even
when `queryCount` is zero. Pinned wgpu checks only that the end of the range
does not exceed the count, and SAGA currently follows that behavior. The
browser host skips the zero-count native call, so this case is accepted through
SAGA even though it does not satisfy the pinned WebGPU rule. The existing
`EmptyQueryResolveTests` and browser diagnostic cover the current behavior;
they are not conformance evidence for this case. This mismatch remains open.

**Known unresolved 3D copy difference (2026-11-10):** SAGA rejects copies
whose source and destination depth-slice ranges overlap within the same 3D mip.
The pinned wgpu range check treats a 3D copy as one array layer and can accept
distinct starting Z coordinates even when the depth ranges overlap. SAGA
supports disjoint slice ranges, including through a browser intermediate.
The pinned WebGPU snapshot identifies a 3D subresource by mip level and requires
copy subresource sets to be disjoint; Chromium rejects a direct same-mip copy
even for disjoint Z slices. SAGA's intermediate therefore implements the wgpu
behavior as an extension, not evidence of strict WebGPU conformance. The
overlapping case needs separate specification and backend-behavior review.

1. **Normative WebGPU API/algorithms** and **WGSL language/validation/execution semantics** at the proposed immutable GPUWeb pin determine behavior. Editor's Draft revisions are proposed engineering baselines, not a W3C certification or final-standard claim. Living readable anchors may move: pinned source snapshots are authoritative for this baseline.
2. **Independent GPUWeb CTS** is a separate conformance authority. Its pinned README says tests are normative and mismatches are bugs. A mismatch requires triage against the pinned spec, recording an upstream issue/resolution; do not silently change Swift behavior or weaken tests to copy a Rust result.
3. **Native wgpu additions** are a separate parity product decision, not WebGPU requirements. `FeaturesWGPU` contains features overlapping newer spec capabilities (for example subgroups); classify by actual pinned normative semantics, never by bit/name alone.
4. **Rust/Naga tests and snapshots** are implementation-specific references to adapt with license review. Their assertions, diagnostics, IR and generated-shader spelling are not independent CTS or a substitute for the specification.
5. Browser/DOM integration, external images/video textures, secure-context/origin checks, color conversions and JS conversions stay explicitly represented; native API adaptation cannot erase them while claiming complete browser WebGPU conformance. The platform direction is [Apple/Metal plus browser WebGPU](adr/use-metal-and-browser-webgpu.md); precise host/profile applicability still needs review.

## Initial measurable denominator

[Inventory](inventory.json) currently contains **750 coarse review units**:

- **139 WebIDL declarations** (interfaces/mixins, namespaces, dictionaries, enums; partial GPUDevice declarations merged; member text preserved).
- **150 explicit anchored WebGPU headings**, including validation, errors, memory/resource states and operations.
- **461 explicit anchored WGSL headings**, including lexical/grammar, typing, builtins, memory, uniformity and execution.
- Separately **59 native-wgpu feature flags** in [native-inventory.json](native-inventory.json); parity proposed, all not-started/not-run. Not part of the 750.

Implemented coarse units: **0/750**. Verified coarse units: **0/750**. This is a tracking baseline, NOT “0% of all normative assertions”: headings overlap declarations and contain nonnormative examples/introductory material; there is no weighting or summation into a WebGPU completion percentage. Every unit has stable ID, pinned source, proposed owner, planned test-source directory, implementation status, verification status and evidence array.

**Normative assertion denominator: unknown. Expanded CTS parameter-case denominator: unknown.** Source-tree file count is not case count. No tests have run. No CTS/Swift/GPU/build pass is claimed.

### Refinement before implementation claims

For every coarse unit, a reviewer must extract atomic requirements with child IDs (`API-GPUBuffer::mapAsync::alignment`, for example) and record: exact source anchor/algorithm step and revision, positive behavior, invalid-input boundary, conditional feature/stage/profile, public Swift API symbol, test IDs and ownership. IDs are immutable once referenced; splitting retains parent IDs and records supersession rather than reusing IDs. Add typedef/includes/inherited mixin members and inline HTML headings not captured by this heading extractor. WGSL builtins require overload/type/stage/input-domain cases, not one parser smoke test per builtin.

Freeze a versioned assertion manifest after review. Every mandatory assertion must be represented once in the assertion denominator (parents are rollups, not extra assertions). Classify every coarse unit `mandatory`, `conditional`, `informative`, or `native-only` with reviewer/rationale. Current section `review-required` is intentionally not an approval/exclusion. Optional features contribute their negotiation/rejection obligations even when not advertised; execution cases enter the applicable matrix when advertised. Never advertise unsupported capability.

### Metrics on a frozen manifest

For one spec revision, runtime build and approved matrix cell:

- `A`: required applicable atomic assertions. `I`: A whose implementation is complete and reviewed. Implementation coverage = `|I| / |A|`.
- `V`: A with passing mapped tests **on Swift**, required evidence layers satisfied and no unresolved counterexample. Verified assertion coverage = `|V| / |A|`.
- `T`: all expanded canonical CTS parameter cases for the frozen query and CTS revision. Enumerate through the approved SAGA-10 runner, storing exact query/case list + SHA-256, feature/limit snapshot, enumeration command and host version. Do not count source files, wildcards or test groups as T.
- Partition T exactly once into `pass`, `fail`, `skip`, `unsupported`, `not-run`; infrastructure crashes/timeouts are `fail` with category and logs, not removed cases. Record expected-fail as a fail annotation, not a pass. Total = sum of those disjoint counts. Gross pass rate = pass / T.
- Separately publish approved `not-applicable` exclusions with assertion IDs, exact case IDs, feature/profile/platform rationale, owner, approving ADR and review date. Effective denominator = T minus approved exclusions. Always publish both raw and effective counts/rates; never shrink raw T or relabel unknown/unsupported as excluded automatically. Zero denominator = `unknown / N/A`, not 100%.
- Optional-capability early returns inside CTS can appear as pass without exercising a feature; pair run evidence with capability/limit snapshots and conditional applicability. Report advertised-feature coverage separately. A pass on one GPU/backend is not a pass on all cells.
- Line/branch coverage, parser acceptance count, native shader compilation, local unit tests and coarse unit completion are supplemental metrics, never CTS coverage.

### Evidence record and state machine

Implementation: `not-started → partial → complete`; verification: `not-run → pass | fail | skip | unsupported`. Regressions reopen complete/pass. Unit-only success cannot mark a GPU requirement pass.

For each execution retain spec/WGSL/CTS SHAs, assertion IDs, expanded CTS IDs, Swift checkout/build digest, bridge version/routing marker, OS/backend/GPU/driver, compiler/language mode, advertised features/limits, seed, command, timestamp, actual result/log artifact and reviewer. Exclusions and skips require reason/owner/expiry. Hardware unavailable is blocked/not-run; CPU fallback cannot satisfy GPU execution. Archive raw per-case output before aggregation. Multi-run flakiness is shown, not hidden by the best run.

## Subsystem acceptance matrix

All rows **not-started / not-run**, test sources below are planned sources, not performed checks. Individual ID links resolve into the baseline register.

| Subsystem | Inventory reference | Workstream | Test source and required future checks |
|---|---|---|---|
| Adapter/device/features/limits; async conversion | [initialization](inventory.md#WEBGPU-SEC-initialization), [asynchrony](inventory.md#WEBGPU-SEC-asynchrony) | SAGA-3/4/9 | CTS `src/webgpu/api/validation`; negotiation/limit negatives, promise ordering, cross-device misuse |
| Buffers/map/unmap/destroy/zero-init | [buffers](inventory.md#WEBGPU-SEC-buffers), [mapping](inventory.md#WEBGPU-SEC-buffer-mapping) | SAGA-7 | CTS API validation/operation; sizes/overflow/alignment/state; GPU readback; destroy while work pending |
| Textures/views/formats/external textures/samplers | [textures](inventory.md#WEBGPU-SEC-textures), [samplers](inventory.md#WEBGPU-SEC-samplers) | SAGA-7/9 | CTS API validation/operation; aspect/mip/sample/usage/format and external-image lifetime/color tests |
| Binding/layout/dynamic offsets and shader reflection | [bindings](inventory.md#WEBGPU-SEC-bindings) | SAGA-6/8 | CTS API validation; stage visibility, auto layouts, binding-size/alignment, cross-stage matching |
| Commands/copies/queues/hazards | [encoding](inventory.md#WEBGPU-SEC-command-encoding), [queues](inventory.md#WEBGPU-SEC-queues) | SAGA-7 | CTS operation/validation; copies, bounds, ordering, debug state, invalidity and resubmission |
| Compute/render pipelines, passes, bundles and draw | [pipelines](inventory.md#WEBGPU-SEC-pipelines), [render passes](inventory.md#WEBGPU-SEC-render-passes), [bundles](inventory.md#WEBGPU-SEC-bundles) | SAGA-8 | CTS operation/validation; async errors, vertex/index/indirect, depth/stencil/blend/MSAA; GPU images/readback |
| Queries/timestamps/errors/loss/presentation | [queries](inventory.md#WEBGPU-SEC-queries), [errors](inventory.md#WEBGPU-SEC-errors-and-debugging), [canvas](inventory.md#WEBGPU-SEC-canvas-rendering) | SAGA-9 | CTS + native presentation tests; fault injection, cancellation and resize/live visual evidence |
| WGSL lexer/parser/directives/AST/diagnostics | [text](inventory.md#WGSL-SEC-textual-structure), [module](inventory.md#WGSL-SEC-wgsl-module) | SAGA-5 | CTS `src/webgpu/shader`; independently authored/adapted positive/negative fixtures; bounded fuzzing |
| WGSL resolution/types/constants/control flow/stages | [types](inventory.md#WGSL-SEC-types), [entry points](inventory.md#WGSL-SEC-entry-points) | SAGA-5 | CTS shader validation/execution; abstract numeric conversion, overflow, overrides, vertex/fragment/compute stage restrictions |
| WGSL memory/layout/uniformity and builtins | [memory](inventory.md#WGSL-SEC-memory), [uniformity](inventory.md#WGSL-SEC-uniformity), [builtins](inventory.md#WGSL-SEC-builtin-functions) | SAGA-5/6 | CTS shader; host-shareable layout, aliasing, atomics/barriers, subgroup/derivative restrictions, FP tolerances |
| Independent IR/native codegen/robustness | [execution](inventory.md#WGSL-SEC-execution), [memory model](inventory.md#WGSL-SEC-memory-model) | SAGA-6 | Swift unit/golden/native compilation + real GPU semantic comparisons; no direct normative IR format |
| JS-host bridge and proof-of-routing | [GPU API](inventory.md#API-GPU) | SAGA-10 | Independent CTS enumeration, Promise/error/typed-array/map contracts, bridge sentinel/counters |

## CTS routing gate

SAGA-10 must install a JS WebGPU facade whose every relevant API path reaches SAGA Swift API/validation. Apple native execution targets Metal; web execution targets browser WebGPU. Exact CTS host/interop and tested browser versions remain follow-on decisions. Deno/wgpu runner is reference only, not the bridge implementation.

Before accepting even one CTS result: identify Swift runtime build in bridge handshake; trace calls into Swift with per-run correlation ID; deliberately inject a Swift-side validation sentinel and observe the expected JS error; remove/disable bridge and confirm smoke tests fail rather than falling back to browser `navigator.gpu`; inspect SAGA project and linked dependencies for Dawn/wgpu/Rust Naga. Browser-owned internals are not project dependencies; the intended browser backend is allowed, while bypassing the SAGA facade is not. Record typed-array copying/lifetime, Promise resolution/rejection and scheduling semantics. Stock browser CTS evaluates that browser's GPU implementation, not SAGA. A SAGA-routed web run is a distinct profile using the browser compiler/GPU implementation: it cannot prove SAGA independent Swift compiler/codegen or native Metal correctness. Archive native and web evidence separately. Additional compiler-format scope remains gated on human input, not inferred from browser WGSL support.

## Security and concurrency gates

- WGSL is untrusted: bounded source size, nesting, recursion/work in parser/type resolver/constant evaluation, IR/codegen complexity and compiler timeouts. Reject unsupported syntax with diagnostics; never regex-translate or let native shader compiler substitute for WGSL validation.
- Pre-backend validation must reject overflow, invalid usage/layout/state/cross-device handles. Zero initialization and robust OOB access must survive optimizer lowering and indirect workloads; use GPU tests plus fuzz/stress. HAL behavior is not a safe validation boundary.
- Define queue ordering versus CPU actor scheduling, ownership across pending GPU completion, mapping cancellation, exactly-once continuations, loss/destroy races and shutdown. Strict Swift 6.2-compatible concurrency; do not blanket-mark unchecked Sendable. Concurrency ADR remains proposed in SAGA-3.
- Native unsafe/passthrough extensions require separate explicit risk acceptance and never contaminate safe normative API guarantees.

## Verification for this baseline

Run `python3 docs/tools/verify_docs.py` from saga. It checks local links/explicit inventory anchors, source hashes, IDs and source token existence, declared CTS tree paths, reference paths/revision, DAG and stable generated inventory. It does not execute Swift/Rust/GPU/CTS. [Verification report](verification.md) records actual results and limitations.
