# SAGA-16 partial semantic validation and integration

Implemented independently in Swift: `ShaderSemanticValidator: IRValidator` with invocation-local `ShaderBudget`, recomputed core expression types, topological handles/types/function DAG, core layout, pointer access, binding and stage interfaces, explicit emit/effect-result scope, bounded block ownership/cycles/control flow and conservative transitive uniformity. No raw IR metadata is trusted as already-validated. `IndependentShaderCompiler.parseAndValidate` dispatches the three real frontends and invokes this concrete checker before private witness construction. Codegen still unsupported.

Exact implemented and missing families: [coverage](../../../semantic-validation-coverage.json). Unsupported rules reject with typed errors; this is NOT full SAGA-16 or Naga semantic acceptance. Nominal aggregates only compare identical handles; extended equivalence deliberately rejected rather than presumed. Loops conservatively disable uniformity-requiring operations; dynamic analysis precision is incomplete.

16 raw semantic tests PASS on byte-identical isolated IR+Validation package after concurrent frontend conflict. [isolated manifest](isolated-manifest.json) records exact hashes; [log](isolated-tests.log). Direct SAGAValidation strict native build PASS. Earlier integrated13 raw+3 format/validator tests PASS (green-attempt5.log), but those are dated snapshots, not evidence latest full compiler builds. Latest integrated build fails two concurrent SPIRV lowerers declaring same3 types; raw test isolation neither changes the production build nor hides that failure.

Independent Swift literals/hand-built IR only: no upstream shader/test bytes copied. Type/rule vocabulary follows pinned Naga+WGSL reference; existing MIT/Apache and W3C notices retained. No outbound license decision or fixture clearance.

## Final reconciliation update — 23:44Z

Earlier integrated build failure is now resolved after all frontend children stopped. Alternate lowerer preserved intact outside Sources; exact2 integration representation fixes; full native suite113 PASS (verification-6), matched Wasm compiler build PASS (verification-7), strict full native build PASS (verification-8), final docs59 PASS0skips (verification-9). See ../../../frontend-verification.md. Full semantic profile still incomplete, not SAGA-16 done.
