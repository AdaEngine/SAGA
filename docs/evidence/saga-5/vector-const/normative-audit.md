# Vector constant constructors, reductions and sequential floating residuals

Read-only reference pin a55352911bb26a1b33bc8d5fc03bdebb0a967f5a. The original WGSL child sessions are inactive; this is the same original SAGA-5 implementation owner. Tests are independently authored Swift literal inputs; no upstream shader bytes copied, no Rust runtime/FFI/dependency added.

## Audited before implementation

Pinned WGSL `docs/sources/wgsl-spec.bs` #vec2-builtin / #vec3-builtin / #vec4-builtin and #zero-value-builtin-function define:

- vecN<T>() yields concrete zero lanes, while unparameterized vecN() yields AbstractInt zero lanes.
- A single scalar splats, but typed splat requires AUTOMATIC scalar conversion, not arbitrary explicit conversion.
- Components may combine scalars and vectors if the total lane count is N; component types must have a common leaf scalar. Concrete integer/float/bool types do not automatically cross-convert.
- A single same-width vector argument is the explicit component-wise conversion overload. It can call T(component) even where splat/components cannot, e.g vec2<bool>(vec2<i32>(1,2)) is permitted but vec2<bool>(1,2) is not.
- Type aliases are type synonyms resolved by scope/DAG, not string substitution. A nearer value binding prevents invoking that type name. all/any accept scalar bool or vecN<bool> and return scalar bool; vectors themselves cannot serve as the scalar logical short-circuit condition.

Pinned Naga `front/wgsl/lower/construction.rs` distinguishes PartialVector/Type and the Components::One vector conversion, automatic splat conversion and component composition paths. Its source span/name/type contract and current abstract pruning boundary can be preserved without any shared IR or validator changes. [Archived snippets](reference-constructor-audit.log).

## Bounded independent implementation

`VectorConstants.swift` adds a checked2..4-lane constructor helper and scalar-leaf vector values. `ScalarConstantExpressions.swift` retains the flat frontend AST: a vector constructor is one node pointing to already typed argument nodes. No token/source substitution or recursive runtime aggregate representation is used. Generic angle tokens are recognized only at the specific vec2/3/4 constructor template boundary; comparison grammar remains unchanged elsewhere. This is a bounded template profile, NOT the complete WGSL template disambiguation grammar.

Static typing checks component totals/splat/conversion width, leaf type consensus, constructor arity and scope before lazy evaluation. Automatic abstract conversions are recorded as whole-tree obligations and checked per component even in skipped RHS. Explicit same-width vector conversion remains ordinary evaluation and may be skipped, just like existing scalar conversion. Components are evaluated eagerly before all/any; reduction does not make constructor argument evaluation short-circuit. Node/work/depth24/cancellation accounting and all old integer division/shift/clamp obligations remain enabled.

Module vector constants remain in the compile-time environment. Local declarations hold checked vector values in lexical const scopes. Runtime uses create ordinary scalar literals and compose nodes and acquire witnesses only through the existing concrete validator. Abstract vector leaves materialize to their actual context with representability checks (4294967295 can become u32, not forced to i32); default i32/f32 only applies at an unconstrained runtime use. The frontend pruning/remapping boundary now removes abstract vectors as well as abstract scalars. Existing schema/public raw validator/private witness/frozen baseline stay byte-identical. Concrete module vector names are currently inlined, not retained as named IR constants: no reflection/interchange preservation parity is claimed.

Per-component comparisons and unary !/-/~ let callers inspect values without synthetic runtime evaluation. Other binary vector arithmetic and vector select overloads are intentionally fail-closed in this slice, not routed through scalar rules. Numeric comparison materialization is checked before value evaluation; scalar arithmetic/rounding/conversion helpers retain their typed binary32/binary64 rules.

## Continued next residuals rather than stopping

After generic/vector-alias cases passed, implemented audited #sqrt-builtin / #inverseSqrt-builtin scalar and vector float overloads. Each f32 operation uses Float; AbstractFloat uses Double. Evaluated nonfinite/domain results reject, while unselected ordinary builtin evaluation remains lazy. This does not claim all WGSL accuracy/device builtin modes or other floating builtins.

Also implemented pinned WGSL #literals hexadecimal floating syntax using the existing budgeted numeric lexer. A suffix f requires exact representability, unlike decimal suffix f rounding: `0x1.00000001p0f` and values too small for a subnormal must be errors. `HexFloatLiterals.swift` computes normalized significant-bit count, lowest/highest exponent from the original hex digits BEFORE host float parsing. Thus more-than53-bit hex mantissas cannot silently round through Double and hide an inexact f32 literal. Maximum finite f32, minimum subnormal, uppercase, optional point/exponent forms and hex digit f without exponent are tested. f16 suffix/feature validation is still unsupported, never substituted with f32. Exponent parsing has an explicit supported magnitude bound100000; arbitrary-size numeric grammar/precision parity is not claimed.

## Verification and limits

9 new test functions (7 WGSL,2 integration), red/green retained: all original184 tests remain enabled. Includes96 deterministic all/any lane properties,128 malformed vector token trees, counts0...12, nested/depth80 rejection, source precision boundaries, exact Unicode/UTF8 diagnostics, lazy static validation, budgets/cancellation and actual runtime compose validation. Old signal5 loop-const and deep scalar regressions remain in the full suite. An intermediate failed positive used all(vec<float>) (not a valid WGSL bool reduction); corrected to compare vectors first, retaining all numeric-type-negative tests. The corrected source then passed; no production check was relaxed for that mistaken expectation.

Final193 strict native tests, strict native build and matchedWasm compiler results are archived with source hashes in [handoff](handoff.json). No raw validator/sharedIR/baseline edits. Actual [residual execution](residuals.log) confirms resolved generic/alias/sqrt/hex/contextual cases, with vector arithmetic/select/swizzle and matrix/f16 still explicitly rejected. Syntax errors on unsupported matrix templates are not full family diagnostics or parity claims.

No new configured corpus/browser/CTS or WGSL-produced GPU execution. Existing Metal regressions in native suite do not prove shader codegen. Full SAGA-5/root/3-format frontend+semantic/options/corpus DoD remains open; no accept_result/SAGA-6, children/newrun, board payloads, policies/commits/publication/reference writes. Compiler libraries add no runnable target/new Play recipe. There is no current external execution blocker; handoff is exact implementation residuals.
