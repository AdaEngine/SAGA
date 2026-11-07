# Use immutable cross-format shader IR and bounded frontend contracts

- Status: accepted for the bounded SAGA-12 shared contract; **not** full Naga or SAGA-5 acceptance
- Date: 2026-11-06
- Scope: [full-Naga decision](port-full-naga-on-swift.md), [ownership](define-resource-and-concurrency-model.md), [manifest](../naga-scope.json)
- Reference: read-only wgpu `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`

## Context

WGSL, GLSL and SPIR-V must lower to the same independent Swift value model. A WGSL-only
AST cannot represent all upstream inputs. Existing generic `ShaderIRModule`,
`ShaderFrontend`, `IRValidator` and the **private** `ValidatedModule` initializer remain
source compatible. Parsing raw IR never grants a validation witness. No Rust runtime,
FFI, external package or platform handles are added.

The pinned `naga/src/ir/mod.rs`, `naga-types/src/lib.rs`, `naga/src/diagnostic_filter.rs`,
`naga/src/valid/mod.rs` and frontend option declarations were inspected. New Swift code
and contract tests are independently authored, with type/operation vocabulary mapped to
that reference. There are no copied shader fixtures in this child. Existing upstream
[MIT](../sources/wgpu-LICENSE.MIT) and [Apache-2.0](../sources/wgpu-LICENSE.APACHE) notices
remain retained; this ADR does not select the SAGA project's outbound license.

## Decision

Use public immutable `Sendable` structs/enums and typed table-relative handles in
`Sources/SAGAShaderIR`. All concrete module storage is value semantic; there are no
mutable class payloads, actors, pointers or unchecked Sendable escapes.

### Tables and ownership

```swift
IRHandle<Node: Sendable>(index: Int) throws
IRHandle<Node: Sendable>(rawValue: UInt32) throws
IRTable<Element: Sendable>(_ values: [Element] = [])
IRTable<Element>.handle(at: Int) throws -> IRHandle<Element>
IRTable<Element>.value(for: IRHandle<Element>) throws -> Element
IRTable<Element>[handle] -> Element?  // dangling index yields nil, never a trap
IRTableBuilder<Element>.append(_: Element, budget: inout ShaderBudget) throws -> IRHandle<Element>
IRTableBuilder<Element>.freeze() -> IRTable<Element>
```

Handles check integer representability, NOT arena identity. They are not capabilities.
A handle can be constructed for forward declarations; table lookup checks bounds.
Use checked `rawValue:` conversion for SPIR-V IDs, especially on 32-bit Wasm. No
public unsafe integer subscript exists. `values` permits iteration; consumers must not
index it with unvalidated integers. Freezing a builder retains an immutable snapshot.
The builder charges one node and one work unit **before** allocation. Payload lists,
strings and their elements must also be charged by the frontend before constructing them.
Raw array/table initializers are trusted assembly APIs, not memory quotas.

Reference contexts:

- `IRType` handles always refer to `IRModule.types`.
- Constants, overrides, globals, functions and diagnostic filters refer to module tables.
- Constant/override/global initializers and entrypoint workgroup/mesh override expressions
  refer to `IRModule.globalExpressions`.
- Each function owns separate `localVariables`, `expressions`, `statements`, `blocks` tables.
  Expression operands, local initializers, named expressions and statement operands refer to
  that function's expressions. `functionArgument(UInt32)` indexes that function's arguments.
- `IRBlock.statements` indexes its function's statement table. Control flow references blocks
  by handle, **not recursively stored Swift values**. A cyclic malformed graph can be stored
  without recursive destruction/equality; a bounded walker rejects reachable cycles.
- Entry points reference an `IRFunction` in the module function table rather than embedding
  another function. The validator must enforce entrypoint/function call restrictions.
- All concrete types expose explicit public initializers and immutable `let` fields.

`IRFunction.walkBlocks(budget:visit:)` is an iterative source-order structural utility.
It checks reachable block/statement indices, active-path cycles, work and depth. It does
**not** validate expression operands, type IDs, uniformity, block semantics, unreachable
blocks, type equivalence, resource layout or capability permissions. A successful walk
must never be promoted to a semantic-validation witness.

### Data and operation coverage

The representational model includes the following pinned families. Counts below are
**declaration mapping**, not semantic tests or implementation parity:

| Pinned family | Swift representation | Variant count |
| --- | --- | --- |
| TypeInner | IRTypeKind | 14 |
| Expression | IRExpressionKind | 35 |
| Statement | IRStatementKind | 24 |
| MathFunction | IRMathFunction | 79 |
| BuiltIn | IRBuiltIn | 50 |
| StorageFormat | IRStorageFormat | 41 |
| ShaderStage | IRStage | 9 |
| validation capabilities | ShaderCapabilities | 46 named bits |

This includes image queries/sampling/storage/atomics, memory barriers/decorations,
subgroups and gather modes, ray-query and ray-pipeline operations, mesh/task metadata,
cooperative matrices/load/store/multiply-add, abstract scalars, pointers/value pointers,
binding arrays, constants/overrides, special/predeclared types and documentation targets.
There is no catch-all operation string that silently accepts unknown opcodes.

`IRExpression.type` explicitly records the frontend's resolved type. It is an assertion
by a trusted producer, not checked truth; the validator must recompute/check it. `f16Bits`
preserves the exact 16-bit payload without requiring a third-party half-float runtime.
NaN, infinity, invalid scalar widths, zero array sizes, unsupported capability combinations,
invalid flags, malformed expression DAGs and declaration cycles remain representable raw
IR and must be rejected by semantic validation. Swift `Equatable` is structural equality,
not language type equivalence or shader semantic equivalence.

Expression evaluation timing is preserved explicitly: `emit` lists expression handles;
calls/atomics/subgroups/proceed are statements with explicit result-expression handles.
Frontends must lower SPIR-V phi/control flow to structured blocks/local spills, preserve
emit/scope ordering, and not turn effectful calls into freely movable expressions.

### Diagnostics and ranges

`ShaderDiagnostic(severity:message:span:)` is preserved. It maps to code `.unspecified`
and `.source(span)`. The additional initializer is:

```swift
ShaderDiagnostic(severity: .error, code: .invalidBinary,
                 message: "Invalid operand", location: .binary(wordSpan))
```

`ShaderDiagnosticCode` has stable SAGA numeric strings and a typed category. This is a
shared diagnostic taxonomy, not a one-to-one port of every frontend's detailed error enum.
`ShaderLocation.source(SourceSpan)` uses half-open original UTF-8 byte offsets.
`SourceSpan.validate(in:)` checks scalar boundaries and source bounds, not grapheme
boundaries. `ShaderLocation.binary(BinaryWordSpan)` uses half-open original 32-bit word
indices **including the SPIR-V header**. Its optional byte conversion returns distinct
`BinaryByteSpan`; it cannot be mistaken for a text coordinate. Multiplication overflow
throws `ShaderFailure.invalidSpan`. Binary diagnostics have `span == nil`.
Preprocessing must preserve a source map; macro expansion locations must not be invented.
`IRDiagnosticRule` models derivative-uniformity/unknown/user filter rules separately from
emitted SAGA codes. Filter severity has off/info/warning/error; actual reporting is a
frontend/validator responsibility.

### Bounded work and frontend dispatch

`ShaderLimits(sourceBytes:words:tokens:nodes:depth:work:) throws` rejects negative values
and depth greater than 512. Default: 4 MiB text, 1,048,576 words/tokens/nodes, depth 128,
16,777,216 work units. Zero is a valid deny-all limit. Exhaustion throws
`ShaderFailure.budgetExceeded(resource)`; invalid bookkeeping throws `.invalidBudget`.
Arithmetic uses checked remaining-budget subtraction rather than overflow-prone addition.

`ShaderBudget` is an invocation-local value. `consume` and depth methods check task
cancellation. `checkInput` charges source UTF-8 bytes including GLSL define keys/values,
SPIR-V words, and preflight work. Text counting stops at the first limit. Tokens,
expansion, nodes, traversal, identifier hashing/comparison and recursion must be charged
by the actual frontend. `withDepth` restores previous nesting even on errors/cancellation.
Budget copies are independent branches, not shared global quotas; failure may leave work
charged and the failed invocation must be discarded. This is cooperative accounting, not
preemption of arbitrary client code or a bound on already allocated caller input storage.

New frontends conform to `BoundedShaderFrontend` and explicitly declare
`typealias Module = IRModule`. Implement `limits` and
`parse(_:budget: inout ShaderBudget) throws -> IRModule`. Its default legacy `parse(_:)`
preflights input, checks format, forwards the SAME budget and checks cancellation on exit.
Do not implement a second legacy parse that bypasses this path. Legacy synthetic
foundation frontends remain compatible; the generic protocol is a trusted SPI, not a sandbox.

`WGSLFrontendOptions` carries parseDocComments/capabilities; `GLSLFrontendOptions` carries
IRStage/defines; `SPIRVFrontendOptions` carries adjustCoordinateSpace/strictCapabilities.
Options express requested policy, not evidence it is implemented. Upstream's SPIR-V
`block_ctx_dump_prefix` debug filesystem side effect is deliberately not implemented.
No output path is silently ignored. Add an explicitly authorized tooling API if needed.
Legacy `ShaderStage` and `ShaderInput.glsl` retain their three-stage ABI. `IRStage` and GLSL
options represent all nine reference stages; additional frontend entry APIs must carry those
options rather than silently collapsing mesh/ray/task stages to compute. Existing foundation
exhaustive switches are unchanged.

### Construction example

This contract example constructs raw void compute IR, **not parsed shader success**:

```swift
import SAGAShaderIR

var budget = ShaderBudget()
var statementBuilder = IRTableBuilder<IRStatement>()
let ret = try statementBuilder.append(IRStatement(kind: .returnValue(nil)), budget: &budget)
var blockBuilder = IRTableBuilder<IRBlock>()
let body = try blockBuilder.append(IRBlock(statements: [ret]), budget: &budget)
var functionBuilder = IRTableBuilder<IRFunction>()
let function = try functionBuilder.append(
    IRFunction(name: "main", statements: statementBuilder.freeze(),
               blocks: blockBuilder.freeze(), body: body), budget: &budget)
let module = IRModule(functions: functionBuilder.freeze(), entryPoints: [
    IREntryPoint(name: "main", stage: .compute, function: function,
                 workgroupSize: IRWorkgroupSize(x: 1, y: 1, z: 1))
])
// Pass module through the real semantic validator before any code generator.
```

For a typed literal: create `IRType(kind: .scalar(.u32))` in module.types, get its handle,
then `IRExpression(type: u32, kind: .literal(.u32(7)))` in the appropriate expression table.
All three frontend workers can use the same structures; no IR source format discriminator
is required for semantics. Keep their source maps/options separately.

## Alternatives and consequences

- Recursive Swift block enums are convenient but malicious depth can overflow stack during
  traversal, equality or release. Flat tables avoid implicit recursion; bounded explicit
  traversal is slightly more verbose.
- Mutable shared reference arenas were rejected: Sendable alone does not make a validated
  witness immutable. Local value-semantic builders preserve safe parallel frontend ownership.
- A fully generic `Any`/opcode-string representation would shorten code but defer all shape
  checking. Typed payloads make unsupported constructs explicit.
- Frozen tables do not intern types. Validator/type equivalence, compaction and transforms
  remain separate. Decode/transform returns raw IR and must be validated again.

## Actual verification and remaining scope

Native: Apple Swift 6.3.3; tools-version 6.2, language mode 6, warnings-as-errors and
strict-concurrency=complete. **11 new contract tests passed**, plus six adjacent foundation
compiler tests (**17 test functions**, including 18 parameter pairs proving unimplemented
translations still fail). The XCTest compatibility runner's zero is not the Swift Testing
count. Test-first missing-API failure was recorded before implementation; fixture Unicode
normalization and explicit bounded-protocol associated-type conformance were fixed and rerun.
A short deterministic boundary sweep tests invalid indices/limits/UTF-8 offsets, not shader
parser fuzz conformance. Cycle/dangling block checks, ordering/work/depth limits, cancellation,
builder snapshots and binary-word overflow have focused tests.

Exact native commands:

```
/usr/bin/swift build --target SAGACompiler --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
/usr/bin/swift test --filter FrontendContractsTests --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
```

The six existing compiler checks were also run through a combined regex filter. The
compiler dependency closure builds with matched upstream Swift 6.3.2 and the installed
`swift-6.3.2-RELEASE_wasm` SDK; **compile-only**, not Wasm execution/browser/WebGPU evidence.
The construction example above also passed direct strict swiftc typechecking.
Native compiler 6.2.x, complete project suite, GPU, browser, CTS, hosted CI, real frontend
fixtures, full semantic validation and interoperability were not executed by this child.
This library-only result has no runnable app or Play recipe.

Current-turn final evidence in child session `session-a9cf17ae-fcab-47fc-a5fd-c5f52b739285`:
`verification-10` final native strict compiler build; `verification-8` 17 tests;
`verification-9` matched Wasm compiler build. Logs are tool-owned durable artifacts, not
new evidence files written outside this child's allowlist. Their tool-output directory is:

```
/Users/vlad-prusakov/.sloppy/.sloppy/tool-outputs/3ee4b71c2247295ba03153e9195d77a638ae733c189c9e900353376fe8463484/efa21aa1aa9d2eac24bf717d05e161420f6120bd68f51d4b0be1eb75ac16a348/
```

- Native build stdout: `3283493b-c7b6-4fda-846a-9b14cab33ed0.log`
- Final tests stdout: `ef9188ae-02a5-4395-be29-3298dc8842bc.log`
- Final tests build stderr: `cddfb1f6-9cef-4687-90d6-21fcf6d3c172.log`
- Final Wasm build stdout: `13302203-98f6-4c45-98e5-4f51ccf21a40.log`
- Initial expected red stderr: `848e0b49-e70f-4e1e-8cea-76a30a10d695.log`

**Exact remaining work:** real WGSL lexing/parsing/resolution/lowering; GLSL preprocessing,
versions/extensions/stages/defines and lowering; SPIR-V opcode decoding/ID rules/phi spills/
control-flow reconstruction; all reference semantic/capability/layout/uniformity rules;
per-frontend detailed diagnostic cases/source maps/filter behavior; type interning/equivalence,
constant evaluation, robustness, transforms, reflection, all outputs, CLI and upstream
serialization/binary interchange compatibility. No parent task or workflow result is closed
by these contracts. There are no intentionally omitted core TypeInner/Expression/Statement
families in this model, but a represented variant is not an implemented semantic construct.

Final additional checks: the existing dependency/module audit passed (13 targets, zero external
packages); its tool stdout is `1cc88ad9-87c9-4298-87c3-2758105b57c1.log` in the same directory.
Expected compile-time rejections for immutable module mutation, wrong-kind table handle and
private witness construction passed; log `0a12d9c2-2019-462c-8340-e0175c6ba681.log`.
Selected unowned boundary hashes (validator, compiler, codegen, existing compiler tests,
package and frozen inventory) matched pre-edit hashes. Rust status remained clean at the
pinned revision. Only this child's assigned files were edited; no board/workflow mutation.

Coordinator delivery limitation: both explicit `messages.send` attempts to
`yadev/session-e26a2eeb-8007-4c5e-a65a-5eccd466c9e6` were rejected by
`message_chain_limit` (eight-delivery automatic chain cap). No receipt or coordinator read
is claimed. Final child session output and this ADR contain the actual handoff; do not
bypass the message gate. This communication limit is unrelated to build/test approval.
