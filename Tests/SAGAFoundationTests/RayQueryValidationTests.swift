import SAGAShaderIR
import SAGAValidation
import SAGAFrontends
import Testing

@Suite struct RayQueryValidationTests {
  private func queryModule(statements: [IRStatement], resultType: Int = 2) throws -> IRModule {
    func h<T: Sendable>(_ index: Int) throws -> IRHandle<T> { try IRHandle(index: index) }
    let types = IRTable([IRType(kind: .rayQuery(vertexReturn: false)),
      IRType(kind: .pointer(base: try h(0), space: .function)),
      IRType(kind: .scalar(.bool)), IRType(kind: .scalar(.u32))])
    let function = IRFunction(localVariables: IRTable([IRLocalVariable(type: try h(0))]),
      expressions: IRTable([IRExpression(type: try h(1), kind: .localVariable(try h(0))),
        IRExpression(type: try h(resultType), kind: .rayQueryProceedResult),
        IRExpression(type: try h(3), kind: .literal(.u32(1)))]),
      statements: IRTable(statements),
      blocks: IRTable([IRBlock(statements: try statements.indices.map { try h($0) })]), body: try h(0))
    return IRModule(types: types, functions: IRTable([function]))
  }

  @Test func rayOperationsRejectIncorrectOperands() throws {
    let query = try IRHandle<IRExpression>(index: 0)
    let integer = try IRHandle<IRExpression>(index: 2)
    let validator = ShaderSemanticValidator(capabilities: [.rayQuery])
    for operation in [IRRayQueryFunction.initialize(accelerationStructure: integer, descriptor: integer),
                      .generateIntersection(hitT: integer)] {
      #expect(throws: ShaderFailure.self) {
        try validator.validate(queryModule(statements: [.init(kind: .rayQuery(query: query, function: operation))]))
      }
    }
    #expect(throws: ShaderFailure.self) {
      try validator.validate(queryModule(statements: [.init(kind: .rayQuery(query: integer, function: .terminate))]))
    }
  }

  @Test func proceedResultMustBeProducedOnceWithBooleanType() throws {
    let query = try IRHandle<IRExpression>(index: 0)
    let result = try IRHandle<IRExpression>(index: 1)
    let proceed = IRStatement(kind: .rayQuery(query: query, function: .proceed(result: result)))
    let validator = ShaderSemanticValidator(capabilities: [.rayQuery])
    try validator.validate(queryModule(statements: [proceed]))
    #expect(throws: ShaderFailure.self) { try validator.validate(queryModule(statements: [proceed, proceed])) }
    #expect(throws: ShaderFailure.self) { try validator.validate(queryModule(statements: [proceed], resultType: 3)) }
    #expect(throws: ShaderFailure.self) {
      try validator.validate(queryModule(statements: [.init(kind: .emit([result]))]))
    }
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: []).validate(queryModule(statements: [proceed])) }
  }
  @Test func rayTypesRequireCapability() throws {
    for kind in [IRTypeKind.rayQuery(vertexReturn: false), .accelerationStructure(vertexReturn: false)] {
      let module = IRModule(types: IRTable([IRType(kind: kind)]))
      do {
        try ShaderSemanticValidator(capabilities: []).validate(module)
        Issue.record("Ray type accepted without capability")
      } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == .unsupportedCapability)
      }
      try ShaderSemanticValidator(capabilities: [.rayQuery]).validate(module)
    }
  }

  @Test func malformedRaySpecialTypesAreRejected() throws {
    let handle = try IRHandle<IRType>(index: 0)
    for special in [IRSpecialTypes(rayDescriptor: handle), IRSpecialTypes(rayIntersection: handle)] {
      let module = IRModule(types: IRTable([IRType(kind: .scalar(.u32))]), specialTypes: special)
      #expect(throws: ShaderFailure.self) {
        try ShaderSemanticValidator(capabilities: [.rayQuery]).validate(module)
      }
    }
  }

  @Test func validatesFrontendRayOperations() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
      enable wgpu_ray_query;
      @group(0) @binding(0) var scene: acceleration_structure;
      @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(RAY_FLAG_NONE, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
        let hit = rayQueryGetCommittedIntersection(&query);
        rayQueryTerminate(&query);
      }
      """))
    try ShaderSemanticValidator(capabilities: [.rayQuery]).validate(module)
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator(capabilities: []).validate(module) }
  }
}
