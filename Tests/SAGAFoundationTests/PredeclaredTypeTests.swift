import Testing
import SAGACompiler
import SAGAFrontends
import SAGAShaderIR

@Suite struct PredeclaredTypeTests {
  @Test func vectorAliasesWorkInTypesAndConstantConstructors() throws {
    for n in 2...4 {
      for (suffix, scalar, value) in [("i","i32","2i"),("u","u32","2u"),("f","f32","2f")] {
        let name = "vec\(n)\(suffix)"
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("""
          alias V = \(name);
          const a = \(name)(\(value));
          const_assert all(a == vec\(n)<\(scalar)>(\(value)));
          fn f(v: \(name)) -> V { return v; }
          fn g() -> \(name) { return \(name)(\(value)); }
          """))
      }
    }
  }
  @Test func matrixAliasesCoverAllNineShapes() throws {
    for c in 2...4 { for r in 2...4 {
      let name = "mat\(c)x\(r)f"
      _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("""
        alias M = \(name);
        const a = \(name)();
        const_assert all(a[0] == vec\(r)f(0));
        fn f(v: M) -> \(name) { return v; }
        fn g() -> M { return \(name)(); }
        """))
    }}
  }
  @Test func userDeclarationsShadowPredeclaredAliases() throws {
    for source in [
      "alias vec2f = vec2<u32>; const x = vec2f(1u); const_assert all(x == vec2<u32>(1u)); fn f(v:vec2f) -> vec2<u32> {return v;}",
      "struct vec2f { inner: vec2<f32> } fn f(v:vec2f) -> vec2<f32> {return v.inner;}",
      "fn f(vec2f: u32) -> u32 {return vec2f;}",
      "fn f() {const vec2f = 3; const_assert vec2f == 3;}",
    ] { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
    #expect(throws:ShaderFailure.self) {
      try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f(vec2f:u32) {let x = vec2f(1);}"))
    }
  }
  @Test func invalidAndHalfAliasesRemainFailClosed() throws {
    for name in ["vec1f","vec5u","vec2b","vec2ff","mat2x2i","mat5x2f","vec2h","mat4x4h"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("fn f(v:\(name)) {}"))}
    }
    for expression in ["vec2u(1u,2u,3u)","mat2x2f(1f)","vec2f<u32>(1u)","vec2u(-1)"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const x = \(expression);"))}
    }
  }
}
