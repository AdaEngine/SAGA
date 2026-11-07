# Define resource ownership and concurrency contracts

- Status: accepted for SAGA-3 foundation, not a stable WebGPU API or runtime conformance claim
- Date: 2026-11-06
- Owner: SAGA / yadev; implementation authority delegated by the user in SAGA-3

## Context

The approved [platform decision](use-metal-and-browser-webgpu.md) selects Apple/Metal and
Swift web builds through browser WebGPU. The later [full Naga decision](port-full-naga-on-swift.md)
supersedes its old open compiler-format gate. Rust is read-only reference at
`a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`. Neither raw platform handles nor foreign
shader/runtime implementations may cross the independent Swift boundary.

Swift concurrency must protect validation, resource lifetime and ordering even when an
async backend suspends. SAGA-3 supplies testable contracts before GPU commands and complete
shader IR are implemented. Sendable alone does not guarantee immutability or queue order.

## Decision

Use immutable Sendable descriptors/errors/IDs and actor-owned mutable device state.
The [foundation guide](../foundation.md) specifies the module DAG and safety contract.

- `Device.make` negotiates requested features against immutable backend capabilities.
  Process-local device identities and monotonically increasing resource serials never reuse
  identifiers; exhaustion throws rather than wrapping. IDs are names, not security capabilities.
- `Device.createBuffer` validates flags, limits and mapped alignment **before** backend calls.
  Allocation starts pending; after suspension it rechecks cancellation and device loss.
  Failure rolls back the backend allocation before leaving the operation. Mapped allocation
  deliberately throws unsupported until the mapping API exists.
- Explicit ownership is required: resource owners call `destroyBuffer`, device owners await
  `close`. No asynchronous work is hidden in deinit. Backend implementations additionally
  own their native/JS objects and must release them on final teardown. There are no raw
  pointers or public Metal/Vulkan/browser handles in API/Core/compiler contracts.
- Destroy invalidates new uses immediately and is idempotent for retired local serials.
  Accepted submissions pin unique resources. Physical destruction waits for completion of
  all accepted uses. Retired entries are removed, without an unbounded tombstone table.
- `submitResourceUse` is a foundation lifecycle harness, **not** a WebGPU command submission
  API. Actor acceptance defines total order. Each task waits for its predecessor's completion,
  not merely backend enqueue. Errors retire pins and do not poison following submissions.
  A backend device-loss error makes the device terminal; explicit loss does the same.
- Cancellation before acceptance creates no work. Cancelling allocation rolls it back after
  backend return. Cancelling a ticket waiter does not cancel accepted GPU work or release
  resources early. Ticket wait cancellation is cooperative at entry/exit, not prompt while
  the backend is outstanding. Backend operations must eventually return even on device loss.
- Error scopes form an explicit device-global LIFO stack. Each async operation retains its
  **acceptance-time** scope tokens. Popping detaches the scope from future operations, waits
  for outstanding operations, and returns its first matching error. Later errors matching
  an already-full scope do not escape to an outer scope. Device loss is uncaptured and
  terminal. Independent tasks must coordinate push/pop order; no implicit task-local stack.
- `lose` marks terminal state before suspension and keeps the entire teardown batch counted
  as pending. Every `close` waits for that batch, pending allocations, accepted submissions,
  and pending destroys. This includes overlapping loss/close callers. No submission may
  reuse a destroyed resource or publish an allocation after loss.
- Trusted `BackendDevice` implementations must zero-initialize allocations, roll back partial
  failure, tolerate idempotent destroy, keep IDs device-namespaced, and only complete submit
  after native/JS use retires. The protocol is a trusted SPI, not a sandbox: callers bypassing
  `Device` take responsibility for validation. BrowserGPUHost is explicit injection without
  direct browser fallback; JS integration is future work.
- Compiler interfaces separate raw `ShaderIRModule`, frontend parsing, `IRValidator`, private
  `ValidatedModule` construction, transforms/interchange and code generation. Codegen accepts
  only validated witnesses. A transformed or decoded raw IR requires validation again.
  Compiler implementors are trusted: modules must be immutable/value-semantic; external
  mutable classes cannot be made safe merely by conforming to Sendable. Full typed IR/schema,
  capabilities, reflection/layout/robustness and complete format options are SAGA-5/6 work.
- `IndependentShaderCompiler` and `MetalBackend.makeDevice` fail unsupported, never return a
  dummy successful translation/device. Generic pipeline tests use visibly synthetic test IR
  and artifacts, not shaders. Binary SPIR-V containers are separate from text containers;
  container checks are not semantic shader validation. Interchange is an interface only,
  not a claim of compatibility with upstream binary fixtures.

Tools-version **6.2** and language mode **6** are distinct from toolchain patch versions.
Swift 6 strict concurrency is mandatory, warnings are errors in verification/CI. There is
no `.v6_2` language mode and no unreviewed unchecked Sendable, unsafe isolation escape, FFI,
external package or system/binary target. The [runtime policy](../runtime-policy.json) checks
that exact module graph and import allowlists.

## Alternatives

1. Lock-protected shared device objects: shorter synchronous API, but difficult cancellation,
   retirement and callback reentrancy; rejected for foundation state ownership.
2. MainActor-only runtime: convenient for browser UI, but would constrain native queues and
   compiler concurrency. JS-specific executor confinement belongs inside the future host.
3. Actor per resource: stronger local isolation but makes atomic multi-resource submission
   validation/pinning cumbersome. One device actor owns resource state; adapters own handles.

## Consequences

Actor reentrancy is explicit and tested; conservative completion-order serialization favors
correctness over GPU throughput. Multi-queue scheduling, real commands, full WebGPU error timing,
map/unmap, backend-specific deferred retirement and optimized ownership can evolve behind these
contracts after SAGA-4/7 evidence. Browser host executor/interop and full IR schema need separate
ADRs. The foundation feature/limit model contains only two feature names and buffer-size limit;
it is not the final normative inventory. No native extension or extra runtime backend is selected.

## Verification

[Public API review](../public-api-review.md) and [foundation verification](../foundation-verification.md)
record concrete tests and toolchains. Required negative checks include invalid descriptors,
foreign/stale IDs, destroy in flight, allocation cancellation/loss rollback, queue errors/order,
async scope pop and late-error routing, overlapping teardown, unsupported translators and dependency
policy regressions. GPU, browser execution and CTS remain separate future gates; no success here
is claimed for them. Historical SAGA-2 documentation evidence remains immutable.

## Addendum — 2026-11-06 SAGA-4 concrete command backend

The original foundation result above is historical. SAGA-4 implements Metal factory/commands
and refines terminal timeout vs physical retirement and owner teardown in
[initial backend ADR](select-initial-backends.md). Successful submissions still imply GPU
completion; terminal timeout uses backend-owned in-flight quarantine until a real callback.
The optional command SPI carries no native handles; the independent compiler remains unsupported.
