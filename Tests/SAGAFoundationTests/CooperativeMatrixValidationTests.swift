import SAGAShaderIR
import SAGAValidation
import Testing

@Suite struct CooperativeMatrixValidationTests {
  let capabilities: ShaderCapabilities = [.cooperativeMatrix, .shaderFloat16]

  @Test func validatesRectangularMixedAccumulatorAndElementStride() throws {
    try ShaderSemanticValidator(capabilities: capabilities).validate(matrixModule())
  }

  @Test func requiresCooperativeMatrixCapability() throws {
    try rejects(matrixModule(), .unsupportedCapability, capabilities: [.shaderFloat16])
  }

  @Test func rejectsIncompatibleRolesDimensionsAndScalarTypes() throws {
    try rejects(matrixModule(rightRole: .a), .typeMismatch)
    try rejects(matrixModule(rightRows: .eight), .typeMismatch)
    try rejects(matrixModule(rightScalar: .f32), .typeMismatch)
  }

  @Test func requiresComputeStageAndWritableScalarStorage() throws {
    try rejects(matrixModule(stage: .fragment), .invalidControlFlow)
    try rejects(matrixModule(access: .load), .invalidAddressSpace)
    try rejects(matrixModule(memoryScalar: .f16), .typeMismatch)
  }

  @Test func acceptsOverlappingStridesAndDefersMemoryBoundsToTheBackend() throws {
    try ShaderSemanticValidator(capabilities: capabilities).validate(matrixModule(stride: 7))
    try ShaderSemanticValidator(capabilities: capabilities).validate(matrixModule(count: 63))
    try ShaderSemanticValidator(capabilities: capabilities).validate(matrixModule(base: 1))
  }

  @Test func requiresSubgroupUniformOffsetStrideAndControl() throws {
    try rejects(matrixModule(varying: .offset), .nonUniformControlFlow)
    try rejects(matrixModule(varying: .stride), .nonUniformControlFlow)
    try rejects(matrixModule(varying: .control), .nonUniformControlFlow)
  }

  private func rejects(
    _ module: IRModule, _ expected: ShaderDiagnosticCode, capabilities: ShaderCapabilities? = nil
  ) throws {
    do {
      try ShaderSemanticValidator(capabilities: capabilities ?? self.capabilities).validate(module)
      Issue.record("Invalid cooperative matrix module was accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
      #expect(diagnostics.first?.code == expected)
    }
  }

  private enum Varying { case offset, stride, control }

  private func matrixModule(
    rightRole: IRCooperativeRole = .b, rightRows: IRCooperativeSize = .sixteen, rightScalar: IRScalar = .f16,
    memoryScalar: IRScalar = .f32, access: IRStorageAccess = [.load, .store],
    count: UInt32 = 64, base: UInt32 = 0, stride: UInt32 = 8, stage: IRStage = .compute, varying: Varying? = nil
  ) throws -> IRModule {
    func t(_ index: Int) throws -> IRHandle<IRType> { try .init(index: index) }
    func e(_ index: Int) throws -> IRHandle<IRExpression> { try .init(index: index) }
    func b(_ index: Int) throws -> IRHandle<IRBlock> { try .init(index: index) }
    func s(_ index: Int) throws -> IRHandle<IRStatement> { try .init(index: index) }

    let space = IRAddressSpace.storage(access: access)
    let types = IRTable([
      IRType(kind: .scalar(.u32)),
      .init(kind: .scalar(.bool)),
      .init(kind: .scalar(memoryScalar)),
      .init(kind: .array(base: try t(2), size: .constant(count), stride: UInt32(memoryScalar.width))),
      .init(kind: .pointer(base: try t(3), space: space)),
      .init(kind: .pointer(base: try t(2), space: space)),
      .init(kind: .cooperativeMatrix(columns: .sixteen, rows: .eight, scalar: .f16, role: .a)),
      .init(kind: .cooperativeMatrix(columns: .eight, rows: rightRows, scalar: rightScalar, role: rightRole)),
      .init(kind: .cooperativeMatrix(columns: .eight, rows: .eight, scalar: .f32, role: .c))
    ])
    let data = IRCooperativeData(pointer: try e(5), stride: try e(varying == .stride ? 3 : 2), rowMajor: true)
    let expressions = IRTable([
      IRExpression(type: try t(4), kind: .globalVariable(try .init(index: 0))),
      .init(type: try t(0), kind: .literal(.u32(base))),
      .init(type: try t(0), kind: .literal(.u32(stride))),
      .init(type: try t(0), kind: varying == nil ? .literal(.u32(0)) : .functionArgument(0)),
      .init(type: try t(1), kind: .binary(op: .equal, left: try e(1), right: try e(3))),
      .init(type: try t(5), kind: .access(base: try e(0), index: try e(varying == .offset ? 3 : 1))),
      .init(type: try t(6), kind: .zeroValue(try t(6))),
      .init(type: try t(7), kind: .zeroValue(try t(7))),
      .init(type: try t(8), kind: .cooperativeLoad(columns: .eight, rows: .eight, role: .c, data: data)),
      .init(type: try t(8), kind: .cooperativeMultiplyAdd(a: try e(6), b: try e(7), c: try e(8)))
    ])
    var statements: [IRStatement] = [
      .init(kind: .emit(try [4, 5].map(e))),
      .init(kind: .emit(try [8, 9].map(e))),
      .init(kind: .cooperativeStore(target: try e(9), data: data)),
      .init(kind: .returnValue(nil))
    ]
    let blocks: [IRBlock]
    if varying == .control {
      statements.append(.init(kind: .ifCondition(condition: try e(4), accept: try b(1), reject: try b(2))))
      blocks = [
        .init(statements: try [0, 4, 3].map(s)),
        .init(statements: try [1, 2].map(s)),
        .init()
      ]
    } else {
      blocks = [.init(statements: try statements.indices.map(s))]
    }
    let function = IRFunction(
      name: "main", arguments: varying == nil ? [] : [try .init(type: t(0), binding: .builtIn(.localInvocationIndex))],
      expressions: expressions, statements: IRTable(statements), blocks: IRTable(blocks), body: try b(0))
    return IRModule(
      types: types,
      globalVariables: IRTable([.init(space: space, binding: .init(group: 0, binding: 0), type: try t(3))]),
      functions: IRTable([function]),
      entryPoints: [IREntryPoint(name: "main", stage: stage, function: try .init(index: 0))])
  }
}
