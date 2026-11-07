# Vector arithmetic/select/access and sequential matrix-constant audit

Read-only Naga pin a55352911bb26a1b33bc8d5fc03bdebb0a967f5a. Independently authored Swift implementation/test literals; no upstream shader source copies or Rust implementation dependency. Same original SAGA-5 session, inactive former frontend children, no new executor/run/board/policy/commit/publication/reference mutation.

## Normative/pinned overload audit before edits

Pinned WGSL `docs/sources/wgsl-spec.bs` [arithmetic](https://gpuweb.github.io/gpuweb/wgsl/#arithmetic-expr), [vector access](https://gpuweb.github.io/gpuweb/wgsl/#vector-access-expr), [select](https://gpuweb.github.io/gpuweb/wgsl/#select-builtin), bitwise and shift expression rules:

- Component-wise numeric vectors support +,-,*,/,%; matching integer vectors support &/|/^ and shifts. Bool vectors have &/| and equality, not xor/arithmetic/ordering. Scalar/vector mixed shapes are normative only for +,-,*,/,%, both directions. No bitwise scalar splat, scalar shift count for vector, scalar-vector comparison or scalar logical vector coercion was invented.
- Exact concrete consensus is required; only abstract numeric promotions permit a different scalar leaf kind. Shapes remain exact. Concrete i32/u32 arithmetic wraps32; AbstractInt is checked (bounded Int64 profile); f32 operations round in Float at every node as previously audited. Selected arithmetic reuses the typed scalar operation semantics; it does not force vectors through one scalar value or rewrite source.
- Shift RHS is a same-width u32 vector; abstract counts are checked/materialized. A count32 is invalid for concrete i32/u32 but valid for this bounded AbstractInt64 profile. Initial negative expectation incorrectly used an abstract vector with32; it was corrected to concrete i32 and a positive AbstractInt shift32 regression was added. No implementation check was weakened to pass that mistaken expectation. The inherited abstract count64 ceiling is explicit supported-profile behavior, not arbitrary-precision shift parity.
- `select` accepts same scalar or vector value type and scalar bool condition, or same-width vectors with same-width bool condition. BOTH value arguments are eagerly evaluated; this is not logical short-circuit.
- Swizzle names contain1..4 letters exclusively xyzw or rgba, unmixed, all within vector width. Repetitions and reordered letters are allowed. A single component is scalar, multiple components a vector. Indices must be scalar integer and within0..N-1 when constant, including in skipped RHS. Negative, huge, wrong-type and mixed/non-ASCII swizzles reject with original UTF8 spans.

Pinned Naga lower/mod.rs binary_op_splat handles Add/Subtract/Divide/Modulo explicitly; multiplication has typed scalar/vector semantics in the IR. Its select path resolves consensus scalar and shape; constant_evaluator.rs Access/AccessIndex/Swizzle extracts checked evaluated values. [Arithmetic snippets](normative-arithmetic-audit.log) · [select/access snippets](normative-select-access-audit.log).

## Independent typed implementation

`VectorOperations.swift` resolves overload shape and leaf type before creating a flat binary AST node. Implicit conversions are recorded per operand in its original shape and validated before ordinary lazy evaluation; scalar expansion happens only after selecting the normative arithmetic overload. Component evaluation uses at most4 typed leaves and the existing audited Float/integer rules. Vector comparison no longer creates artificial scalar AST nodes while evaluating binary components. Unary vector behavior from the previous slice is retained.

Postfix swizzle and index are explicit flat nodes. Index grammar has separate bracket-delimited operator grouping, and bounds are a mandatory creation-time obligation using the base TYPE, without unnecessarily evaluating an unselected base value. The new code does not perform text substitution or treat unknown properties as strings to rewrite.

Required per-component integer checks still run before ordinary short-circuit: denominator zero, signed MIN/-1 (only needs the skipped numerator when divisor is-1), shift count and left-shift overflow. Whole-RHS implicit conversion/syntax/name/type checks and scalar clamp obligations remain. Eager select converts/checks both values. Tests distinguish safe skipped numerator/explicit float conversion from invalid denominator, shift and implicit representation failures.

Runtime uses of compile-time values continue through existing scalar literals and compose. A bounded helper can inspect PURE already-lowered literal/compose/swizzle/access nodes for contextual abstract materialization, never a load/effectful call. This allows4294967295 extracted from a vector to materialize to u32, without globali32 or raw validator relaxation. Dynamic runtime indexing/type rules are NOT declared complete. A new frontend literal-vector-index bounds guard rejects source cases that previously acquired a witness; direct raw IR still has a separately reproduced semantic-rule gap (below).

## Continued next actual residual: matrix constructors in constant context

Pinned WGSL #mat2x2-builtin through #mat4x4-builtin and #zero-value-builtin-function distinguish zero, exact scalar component count, exact column-vector count and single same-shape matrix conversion. `MatrixConstants.swift` independently implements those bounded const-only forms for2..4 columns/rows, at most16 scalar leaves, AbstractFloat/f32, including aliases and column indexing. f32 leaves use the existing per-node conversion/finite checks; column-major order and all9shapes are tested. This is NOT full matrix arithmetic or general aggregate/f16 support.

Matrix values remain frontend compile-time objects and can feed module/function assertions; no shared IR/schema change. Runtime return and local matrix-constant lowering are explicitly unsupported until the concrete matrix compose validation/lowering boundary is implemented. Typed matrix declaration annotations are also not yet complete. No matrix runtime acceptance claim follows from successful const-only assertions.

## Concrete boundary audit, not a silent validator bypass

[Executable raw IR repro](boundary-audit.swift) · [observed output](boundary-audit.log):

1. A well-typed matrix constructed from two f32 vector columns hits raw validator `unsupportedConstruct: Constructor overload not implemented`. Current validator can describe matrix types but cannot prove matrix compose semantics. No public fake witness or scalar encoding workaround was added. Follow-up must implement/audit exact matrix component/type/count rules and runtime lowering together, with raw positive/negative tests.
2. Existing direct raw `Access` (dynamic IR opcode) with a literal index2 into vec2 still acquires a witness. In contrast, static AccessIndex has bounds checks. The original WGSL case is NOW rejected by frontend literal bounds checking, and the regression remains. This does **not** repair the independently constructed raw IR hole: a validator-owner rule change needs prior audit and raw tests for literal/const-folded indices, without rejecting genuinely dynamic runtime values. SharedIR and raw validator are explicitly outside this turn's edits and remain byte-identical. This open safety/semantic defect is not hidden as conformance or claimed fixed by frontend filtering.

This is a concrete boundary handoff under the user's audit-first requirement, not a claim of an external approval/tool blocker or new project scope decision. Full SAGA-5/raw semantic DoD is still unmet. No accept_result/SAGA-6.

## Tests and exact limits

11 new functions:8 WGSL+3 integration; existing193 tests retained. Red→green captured original arithmetic/select/access errors, illegal literal source indexing witness and matrix const construction failure. Positive/negative cases cover all requested shape/type/precision/domain rules;96 deterministic arithmetic/select/access properties and192 malformed operation token trees; exact Unicode spans, quotas/depth24/cancellation, all old signal5loop/signal10deep/hex regressions remain.

An intermediate quota-test source had an incorrect expected selected component; corrected to the mathematically accurate vector and added a default-budget success check BEFORE low-budget failure checks, preventing a false-positive budget test. No production checks were weakened or tests removed. Failure logs retained.

Final204 actual strict native Swift Testing tests, strict native build, matchedSwift6.3.2/Wasm compile-only and docs/reference audits are archived in [handoff](handoff.json). No new configured corpus/browser/CTS/WGSL-generated GPU evidence. Existing native Metal regression is not compiler parity. Concrete module vector names still inline; const-only matrix names also live only in the frontend environment. Full reflection/interchange/name retention remains open. Libraries add no runnable target/new Play configuration.
