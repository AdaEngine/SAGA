# Module dependency and abstract materialization boundary audit

Reference read-only pin: a55352911bb26a1b33bc8d5fc03bdebb0a967f5a. Independently authored Swift implementation/tests; no upstream implementation dependency, shader bytes copied, Cargo/FFI/dependency additions or reference edits.

## Normative and reference evidence

Pinned WGSL `docs/sources/wgsl-spec.bs` scope/name resolution allows a module declaration before **or after** a reference; declarations sharing end-of-scope cannot duplicate identifiers. Names resolve to the nearest local first, then module declaration, then predeclared name. Module const inference preserves abstract type until use; concrete contexts require representability (default i32/f32 only when context does not provide another concrete type).

Pinned Naga `naga/src/front/wgsl/mod.rs` runs parse -> Index::generate -> Lowerer. `front/wgsl/index.rs` indexes declaration references and topologically checks dependencies/cycles. `lower/mod.rs` uses AbstractRule::Allow for constants and Concretize for override/runtime boundaries, and at its end explicitly calls compact to remove abstract unused intermediates. `naga/src/compact/mod.rs` does **not** root abstract named constants even with KeepUnused::Yes; concrete named constants are retained. See [archived reference snippets](pinned-materialization-audit.log). No reference tests or Rust builds run to manufacture parity.

## Independent boundary implemented

`ModuleDeclarations.swift` indexes original immutable token ranges and duplicate module names before lowering. Each typed name lookup resolves a cached declaration DAG; visiting/done states reject cycles at the original reference token. Const initializer references are resolved before entering each evaluator, avoiding one suspended recursive evaluator per module dependency. Work/name bytes/node/edge/depth and cancellation are charged. A conservative graph cap24 remains explicit. All tokens and source positions remain unchanged; no string reorder/regex substitution.

**IR table order is dependency-topological**, not promised to be lexical. That is necessary for checked prior handles and matches pinned Naga. Original declaration ranges/source order remain in the index; exact IR source locations and runtime statement/effect order are preserved and tested. Forward function/global references are handled with caller body/cursor saved and restored; call cycles fail instead of inventing prototypes or accepting recursive IR.

Abstract integer constants stay in the frontend value environment, not globally defaulted to i32. Per use, A=4294967295 may materialize to u32, while A->i32 or an inferred runtime i32 use gives a located representability error. A larger compile-time integer can compute an attribute without any runtime IR object. Frontend scalar abstract literals/negation/name aliases for floats follow the same boundary; full float/vector const evaluation remains unsupported. Module/local shadowing is not ignored; predeclared builtin dispatch refuses a shadowing value.

`Materialization.swift` removes abstract scalar type/expression intermediates only after concrete use-site coercion; it remaps all currently emitted WGSL type/expression/statement/local/function handles. Concrete references still requiring abstract data fail closed. This is frontend-local lowering, **not** a general public IR compaction/transform implementation, not a new SAGA-6 deliverable and not a validator exception. Statement/block handles remain in order, preserving effects; concrete named constants are retained. Future emitted node families must explicitly add checked remapping, otherwise unsupported is returned.

No validator/sharedIR/baseline changes were needed. The raw validator continues to reject abstract scalar IR. Source-memory bool4 / IR bool1 and accepted uniform/storage/witness contracts remain unchanged.

## Tests and limitations

11 independently authored new functions (5 WGSL,6 integration), red/green logs retained. They cover forward/backward types/constants, annotation dependencies, multi-hop diamonds, self/mutual cycles, duplicates/cross-kind names, Unicode/UTF8 diagnostics, malformed index ranges, budgets/cancellation, contextual representability/mixed types, remapped composite/control-flow IR and effect order. A10-layer diamond retains all tests after the implementation was changed to resolve initializer edges ahead of evaluator nesting, rather than raising/removing the depth cap. Deep200decl chains and the existing256deep const-expression regression fail with typed budgets, not traps.

Current exact next failures are in [residuals.log](residuals.log): module boolean comparison constant `const A=1<2` and module `const_assert`. General float/boolean/vector/aggregate const expressions, specialization/overrides, extended resources/types/options/fixture parity remain incomplete. No new browser/GPU/compiler conformance/CTS claims; compile-only Wasm is separate from native regressions. No task/parent acceptance or downstream launch.
