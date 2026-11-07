import Testing
import SAGAShaderIR
import SAGAFrontends
import SAGACompiler
import SAGAValidation

@Suite struct MatrixArithmeticTests {
  @Test func inferredRuntimeConstructorsRespectLocalShadowing() throws {
    for (name,args) in [("vec2","1,2"), ("mat2x2","1,2,3,4")] {
      #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f() { let \(name) = 1; let x = \(name)(\(args)); }"))
      }
    }
  }
  @Test func constantMatrixArithmeticUsesColumnMajorProducts() throws {
    _ = try WGSLFrontend().parse(.wgsl("""
      const a = mat2x3(1,2,3,4,5,6);
      const b = mat4x2(7,8,9,10,11,12,13,14);
      const p = a * b;
      const_assert all(p[0] == vec3(39.0,54.0,69.0));
      const_assert all(p[3] == vec3(69.0,96.0,123.0));
      const_assert all((a * vec2(2.0,3.0)) == vec3(14.0,19.0,24.0));
      const_assert all((vec3(2.0,3.0,4.0) * a) == vec2(20.0,47.0));
      const_assert all((a + a)[1] == vec3(8.0,10.0,12.0));
      const_assert all((a - a)[0] == vec3(0.0));
      const_assert all((2 * a)[0] == vec3(2.0,4.0,6.0));
      const_assert all((a * 3)[1] == vec3(12.0,15.0,18.0));
      """))
  }
  @Test func runtimeMatrixOverloadsReachIndependentValidator() throws {
    for c in 2...4 {
      for r in 2...4 {
        for k in 2...4 {
          let source = """
            fn product(a: mat\(c)x\(r)<f32>, b: mat\(k)x\(c)<f32>) -> mat\(k)x\(r)<f32> { return a * b; }
            fn column(a: mat\(c)x\(r)<f32>, v: vec\(c)<f32>) -> vec\(r)<f32> { return a * v; }
            fn row(v: vec\(r)<f32>, a: mat\(c)x\(r)<f32>) -> vec\(c)<f32> { return v * a; }
            fn scale(a: mat\(c)x\(r)<f32>, s: f32) -> mat\(c)x\(r)<f32> { return (a * s) + (s * a) - a; }
            """
          _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        }
      }
    }
  }
  @Test func invalidMatrixOverloadsRejectEvenInSkippedConstantBranches() throws {
    let invalid = [
      "mat2x2() + 1", "mat2x2() / 2", "mat2x2() % mat2x2()",
      "mat2x2() * vec3(1.0)", "vec3(1.0) * mat2x2()",
      "mat2x3() * mat2x2() + mat2x2()", "mat2x2() == mat2x2()",
      "mat2x2() * 1i", "mat2x2() * true", "-mat2x2()",
      "select(mat2x2(), mat2x2(), true)",
    ]
    for expression in invalid {
      #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("const x = \(expression);")) }
      #expect(throws: ShaderFailure.self) {
        try WGSLFrontend().parse(.wgsl("const x = false && ((\(expression))[0][0] == 0);"))
      }
    }
    for expression in ["a / 2.0", "a + 2.0", "a == a", "a * v"] {
      #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f(a:mat2x2<f32>,v:vec3<f32>) { let x = \(expression); }"))
      }
    }
  }
  @Test func abstractMatrixPromotionAndF32RoundingArePreserved() throws {
    _ = try WGSLFrontend().parse(.wgsl("""
      const a = mat2x2(16777216f,1f,0f,0f);
      const p = a * vec2(1f,1f);
      const_assert p.x == 16777216f;
      const mixed = mat2x2(1.0,2.0,3.0,4.0) * vec2<f32>(2f,3f);
      const_assert all(mixed == vec2<f32>(11f,16f));
      fn f(a:mat2x2<f32>) -> mat2x2<f32> { return a * 2; }
      """))
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f(a:mat2x2<f32>) -> mat2x2<f32> { return a * 2; }"))
  }
  @Test func pureAbstractRuntimeMatrixExpressionsMaterializeAtUse() throws {
    for source in [
      "fn f() -> mat2x2<f32> { return mat2x2(1,2,3,4) * 2; }",
      "const a = mat2x2(1,2,3,4); fn f() -> vec2<f32> { return a * vec2<f32>(2,3); }",
      "const a = mat2x2(1,2,3,4); fn f() -> mat2x2<f32> { return a + a; }",
    ] { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
  }
  @Test func matrixProductsRespectBudgetsAndCancellation() async throws {
    let source = "const a = mat4x4(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16); const b = a * a; const_assert b[0][0] == 90;"
    _ = try WGSLFrontend().parse(.wgsl(source))
    for limits in [try ShaderLimits(work:100),try ShaderLimits(nodes:8),try ShaderLimits(depth:2)] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend(limits:limits).parse(.wgsl(source))}
    }
    let task = Task {while !Task.isCancelled {await Task.yield()};return try WGSLFrontend().parse(.wgsl(source))}
    task.cancel()
    await #expect(throws:CancellationError.self) {try await task.value}
    let chain = "mat2x2()" + String(repeating:" * mat2x2()",count:128)
    #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const m = " + chain + ";"))}
  }
  @Test func rawMatrixResultShapeCannotForgeWitness() throws {
    let kinds: [IRTypeKind] = [.scalar(.f32), .vector(size:.two,scalar:.f32), .vector(size:.three,scalar:.f32),
      .matrix(columns:.two,rows:.three,scalar:.f32), .matrix(columns:.four,rows:.two,scalar:.f32),
      .matrix(columns:.four,rows:.three,scalar:.f32), .matrix(columns:.two,rows:.two,scalar:.f32), .scalar(.i32)]
    for (left,right,result,op,valid) in [
      (3,4,5,IRBinaryOperator.multiply,true),(3,1,2,.multiply,true),(2,3,1,.multiply,true),
      (3,0,3,.multiply,true),(0,3,3,.multiply,true),(3,3,3,.add,true),(3,3,3,.subtract,true),
      (3,4,6,.multiply,false),(3,2,2,.multiply,false),(1,3,1,.multiply,false),
      (3,7,3,.multiply,false),(3,3,3,.divide,false),(3,3,3,.modulo,false),(3,4,3,.add,false),
      (3,0,3,.add,false),(3,3,0,.equal,false),
    ] {
      func h<T: Sendable>(_ i: Int) throws -> IRHandle<T> {try IRHandle(index:i)}
      let expressions: [IRExpression] = [
        .init(type:try h(left),kind:.zeroValue(try h(left))),
        .init(type:try h(right),kind:.zeroValue(try h(right))),
        .init(type:try h(result),kind:.binary(op:op,left:try h(0),right:try h(1)))]
      let function = IRFunction(expressions:IRTable(expressions),statements:IRTable([
        .init(kind:.emit([try h(2)])),.init(kind:.returnValue(nil))]),
        blocks:IRTable([IRBlock(statements:[try h(0),try h(1)])]),body:try h(0))
      let module = IRModule(types:IRTable(kinds.map {IRType(kind:$0)}),functions:IRTable([function]))
      if valid {try ShaderSemanticValidator().validate(module)}
      else {#expect(throws:ShaderFailure.self) {try ShaderSemanticValidator().validate(module)}}
    }
  }
}
