import SAGAShaderIR
import SAGAValidation
import Testing

@Suite struct ShaderSemanticValidationTests {
  func h<T: Sendable>(_ i: Int) throws -> IRHandle<T> { try IRHandle(index: i) }
  func module(
    _ es: [IRExpression] = [], statements: [IRStatement] = [.init(kind: .returnValue(nil))],
    types: [IRType] = [], result: IRFunctionResult? = nil, arguments: [IRFunctionArgument] = [],
    stage: IRStage = .compute
  ) throws -> IRModule {
    let f = IRFunction(
      name: "main", arguments: arguments, result: result, expressions: IRTable(es),
      statements: IRTable(statements),
      blocks: IRTable([IRBlock(statements: try statements.indices.map { try h($0) })]),
      body: try h(0))
    return IRModule(
      types: IRTable(types), functions: IRTable([f]),
      entryPoints: [IREntryPoint(name: "main", stage: stage, function: try h(0))])
  }
  @Test func emptyComputeModuleValidates() throws {
    let m = try module()
    try ShaderSemanticValidator().validate(m)
    #expect(try ValidatedModule.checking(m, with: ShaderSemanticValidator()).module == m)
  }
  @Test func literalTypeIsRecomputedNotTrusted() throws {
    let span = try SourceSpan(startUTF8: 4, endUTF8: 8)
    let m = try module(
      [IRExpression(type: try h(0), kind: .literal(.bool(true)), location: .source(span))],
      types: [IRType(kind: .scalar(.u32))])
    do {
      try ShaderSemanticValidator().validate(m)
      Issue.record("invalid literal accepted")
    } catch ShaderFailure.diagnostics(let ds) {
      #expect(ds.first?.code == .typeMismatch)
      #expect(ds.first?.span == span)
    }
  }
  @Test func invalidHandlesAndTypeCyclesAreRejected() throws {
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        module([IRExpression(type: try h(99), kind: .literal(.u32(1)))]))
    }
    let cycle = IRModule(types: IRTable([IRType(kind: .pointer(base: try h(0), space: .function))]))
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(cycle) }
  }
  @Test func binaryResultTypesAreRecomputed() throws {
    let ts = [IRType(kind: .scalar(.u32)), IRType(kind: .scalar(.bool))]
    let a = IRExpression(type: try h(0), kind: .literal(.u32(1)))
    let bad = IRExpression(
      type: try h(0), kind: .binary(op: .equal, left: try h(0), right: try h(0)))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(module([a, bad], types: ts))
    }
    let good = IRExpression(
      type: try h(1), kind: .binary(op: .equal, left: try h(0), right: try h(0)))
    try ShaderSemanticValidator().validate(module([a, good], types: ts))
  }
  @Test func readonlyUniformCannotBeStored() throws {
    let ts = [IRType(kind: .scalar(.u32)), IRType(kind: .pointer(base: try h(0), space: .uniform))]
    let es = [
      IRExpression(type: try h(1), kind: .globalVariable(try h(0))),
      IRExpression(type: try h(0), kind: .literal(.u32(1))),
    ]
    let f = try module(
      es,
      statements: [
        .init(kind: .store(pointer: try h(0), value: try h(1))), .init(kind: .returnValue(nil)),
      ], types: ts
    ).functions.values[0]
    let m = IRModule(
      types: IRTable(ts),
      globalVariables: IRTable([
        .init(space: .uniform, binding: .init(group: 0, binding: 0), type: try h(0))
      ]), functions: IRTable([f]))
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(m) }
  }
  @Test func capabilitiesAndUnknownBitsAreChecked() throws {
    let m = IRModule(types: IRTable([IRType(kind: .scalar(.f16))]))
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(m) }
    try ShaderSemanticValidator(capabilities: [.shaderFloat16]).validate(m)
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator(capabilities: .init(rawValue: 1 << 63)).validate(IRModule())
    }
  }
  @Test func duplicateBindingsAndBadLayoutsAreRejected() throws {
    let ts = IRTable([IRType(kind: .scalar(.u32))])
    let m = IRModule(
      types: ts,
      globalVariables: IRTable([
        .init(space: .uniform, binding: .init(group: 0, binding: 1), type: try h(0)),
        .init(space: .uniform, binding: .init(group: 0, binding: 1), type: try h(0)),
      ]))
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(m) }
    let bad = IRModule(
      types: IRTable([
        IRType(kind: .scalar(.u32)),
        IRType(kind: .structure(members: [.init(type: try h(0), offset: 1)], span: 8)),
      ]))
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(bad) }
  }
  @Test func builtinStageAndTypeChecked() throws {
    let a = IRFunctionArgument(type: try h(0), binding: .builtIn(.localInvocationIndex))
    try ShaderSemanticValidator().validate(
      module(types: [IRType(kind: .scalar(.u32))], arguments: [a]))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        module(types: [IRType(kind: .scalar(.u32))], arguments: [a], stage: .fragment))
    }
  }
  @Test func blockCyclesBreakAndMissingReturnAreRejected() throws {
    let f = IRFunction(
      statements: IRTable([IRStatement(kind: .block(try h(0)))]),
      blocks: IRTable([IRBlock(statements: [try h(0)])]), body: try h(0))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(IRModule(functions: IRTable([f])))
    }
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(module(statements: [.init(kind: .break)]))
    }
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        module(types: [.init(kind: .scalar(.u32))], result: .init(type: try h(0))))
    }
  }
  @Test func budgetsAndCancellation() async throws {
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator(limits: ShaderLimits(work: 0)).validate(module())
    }
    let m = try module()
    let task = Task {
      while !Task.isCancelled { await Task.yield() }
      try ShaderSemanticValidator().validate(m)
    }
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
  }
}

extension ShaderSemanticValidationTests {
  @Test func derivativeRequiresUniformControlIncludingNestedEarlyExit() throws {
    let ts = IRTable([IRType(kind: .scalar(.bool)), IRType(kind: .scalar(.f32))])
    let es = IRTable([
      IRExpression(type: try h(0), kind: .functionArgument(0)),
      IRExpression(type: try h(1), kind: .literal(.f32(1))),
      IRExpression(
        type: try h(1), kind: .derivative(axis: .x, control: .none, expression: try h(1))),
    ])
    // body -> nested block -> divergent if with early return; derivative follows nested block.
    let ss = IRTable([
      IRStatement(kind: .block(try h(1))),
      IRStatement(kind: .ifCondition(condition: try h(0), accept: try h(2), reject: try h(3))),
      IRStatement(kind: .returnValue(nil)), IRStatement(kind: .emit([try h(2)])),
    ])
    let bs = IRTable([
      IRBlock(statements: [try h(0), try h(3)]), IRBlock(statements: [try h(1)]),
      IRBlock(statements: [try h(2)]), IRBlock(),
    ])
    let f = IRFunction(
      arguments: [.init(type: try h(0))], expressions: es, statements: ss, blocks: bs,
      body: try h(0))
    do {
      try ShaderSemanticValidator().validate(IRModule(types: ts, functions: IRTable([f])))
      Issue.record("Divergent derivative accepted")
    } catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .nonUniformControlFlow) }
    let good = IRFunction(
      expressions: IRTable([
        es.values[1],
        IRExpression(
          type: try h(1), kind: .derivative(axis: .x, control: .none, expression: try h(0))),
      ]), statements: IRTable([.init(kind: .emit([try h(1)])), .init(kind: .returnValue(nil))]),
      blocks: IRTable([IRBlock(statements: [try h(0), try h(1)])]), body: try h(0))
    try ShaderSemanticValidator().validate(
      IRModule(
        types: ts, functions: IRTable([good]),
        entryPoints: [.init(name: "f", stage: .fragment, function: try h(0))]))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        IRModule(
          types: ts, functions: IRTable([good]),
          entryPoints: [.init(name: "f", stage: .compute, function: try h(0))]))
    }
  }
  @Test func effectResultsMustBeProducedByMatchingCallAndRecursionRejected() throws {
    let ts = IRTable([IRType(kind: .scalar(.u32))])
    let callee = IRFunction(
      result: .init(type: try h(0)),
      expressions: IRTable([IRExpression(type: try h(0), kind: .literal(.u32(4)))]),
      statements: IRTable([.init(kind: .returnValue(try h(0)))]),
      blocks: IRTable([IRBlock(statements: [try h(0)])]), body: try h(0))
    let caller = IRFunction(
      result: .init(type: try h(0)),
      expressions: IRTable([IRExpression(type: try h(0), kind: .callResult(try h(0)))]),
      statements: IRTable([
        .init(kind: .call(function: try h(0), arguments: [], result: try h(0))),
        .init(kind: .returnValue(try h(0))),
      ]), blocks: IRTable([IRBlock(statements: [try h(0), try h(1)])]), body: try h(0))
    try ShaderSemanticValidator().validate(
      IRModule(types: ts, functions: IRTable([callee, caller])))
    let forged = IRFunction(
      result: caller.result, expressions: caller.expressions,
      statements: IRTable([.init(kind: .returnValue(try h(0)))]),
      blocks: IRTable([IRBlock(statements: [try h(0)])]), body: try h(0))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        IRModule(types: ts, functions: IRTable([callee, forged])))
    }
    let recursive = IRFunction(
      statements: IRTable([.init(kind: .call(function: try h(0), arguments: [], result: nil))]),
      blocks: IRTable([IRBlock(statements: [try h(0)])]), body: try h(0))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(IRModule(functions: IRTable([recursive])))
    }
  }
  @Test func unsupportedFamiliesRejectInsteadOfIssuingWitness() throws {
    let t = IRType(kind: .image(dimension: .d1, arrayed: false, class: .sampled(kind: .float, multisampled: false)))
    do {
      try ShaderSemanticValidator(capabilities: .allKnown).validate(IRModule(types: IRTable([t])))
      Issue.record("unimplemented type accepted")
    } catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .unsupportedConstruct) }
  }
}

extension ShaderSemanticValidationTests {
  @Test func actualLoopBreaksAndIllegalContinuingReturn() throws {
    let ss = IRTable([
      IRStatement(kind: .loop(body: try h(1), continuing: try h(2), breakIf: nil)),
      IRStatement(kind: .break), IRStatement(kind: .returnValue(nil)),
    ])
    let f = IRFunction(
      statements: ss,
      blocks: IRTable([
        IRBlock(statements: [try h(0), try h(2)]), IRBlock(statements: [try h(1)]), IRBlock(),
      ]), body: try h(0))
    try ShaderSemanticValidator().validate(IRModule(functions: IRTable([f])))
    let invalid = IRFunction(
      statements: ss,
      blocks: IRTable([
        IRBlock(statements: [try h(0)]), IRBlock(statements: [try h(1)]),
        IRBlock(statements: [try h(2)]),
      ]), body: try h(0))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(IRModule(functions: IRTable([invalid])))
    }
  }
  @Test func mathOverloadsAndCastTypesCannotBeForged() throws {
    let types = [IRType(kind: .scalar(.f32)), IRType(kind: .scalar(.u32))]
    let a = IRExpression(type: try h(0), kind: .literal(.f32(2)))
    let b = IRExpression(
      type: try h(0),
      kind: .math(
        function: .sqrt, argument: try h(0), argument1: nil, argument2: nil, argument3: nil))
    try ShaderSemanticValidator().validate(module([a, b], types: types))
    let bad = IRExpression(type: try h(1), kind: b.kind)
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(module([a, bad], types: types))
    }
    let wrongArity = IRExpression(
      type: try h(0),
      kind: .math(
        function: .sqrt, argument: try h(0), argument1: try h(0), argument2: nil, argument3: nil))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(module([a, wrongArity], types: types))
    }
    let cast = IRExpression(type: try h(1), kind: .cast(expression: try h(0), target: .u32))
    try ShaderSemanticValidator().validate(module([a, cast], types: types))
  }
  @Test func adversarialRawHandlesAreDeterministicAndBounded() throws {
    let validator = ShaderSemanticValidator(
      limits: try ShaderLimits(nodes: 256, depth: 16, work: 4096))
    var seed: UInt32 = 0x5341_4741
    for _ in 0..<300 {
      seed = seed &* 1_664_525 &+ 1_013_904_223
      let index = Int(seed % 200)
      let bad = try module(
        [IRExpression(type: try h(index), kind: .load(pointer: try h(index)))],
        types: [IRType(kind: .scalar(.u32))])
      let first = Result { try validator.validate(bad) }
      let second = Result { try validator.validate(bad) }
      switch (first, second) {
      case (.failure(let x as ShaderFailure), .failure(let y as ShaderFailure)): #expect(x == y)
      default: Issue.record("Invalid raw IR did not deterministically fail")
      }
    }
    let huge = IRType(name: String(repeating: "x", count: 4096), kind: .scalar(.u32))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator(limits: ShaderLimits(sourceBytes: 10)).validate(
        IRModule(types: IRTable([huge])))
    }
  }
}

extension ShaderSemanticValidationTests {
  // Independently authored raw IR cases for pinned WGSL address-space-layout-constraints.
  func bufferModule(_ types: [IRType], root: Int, space: IRAddressSpace = .uniform) throws
    -> IRModule
  {
    IRModule(
      types: IRTable(types),
      globalVariables: IRTable([
        .init(name: "buffer", space: space, binding: .init(group: 0, binding: 0), type: try h(root))
      ]))
  }

  @Test func uniformVectorsMatricesAndPaddedArraysAreAccepted() throws {
    let validator = ShaderSemanticValidator()
    for kind: IRTypeKind in [
      .vector(size: .three, scalar: .f32), .matrix(columns: .two, rows: .two, scalar: .f32),
    ] {
      try validator.validate(bufferModule([IRType(kind: kind)], root: 0))
    }
    try validator.validate(
      bufferModule(
        [
          IRType(kind: .scalar(.f32)),
          IRType(kind: .array(base: try h(0), size: .constant(8), stride: 16)),
        ], root: 1))
  }

  @Test func uniformArrayRulesDoNotRejectStorageLayout() throws {
    let types = [
      IRType(kind: .scalar(.f32)),
      IRType(kind: .array(base: try h(0), size: .constant(8), stride: 4)),
    ]
    try ShaderSemanticValidator().validate(
      bufferModule(types, root: 1, space: .storage(access: .load)))
    do {
      try ShaderSemanticValidator().validate(bufferModule(types, root: 1))
      Issue.record("Uniform stride4 array accepted")
    } catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .invalidType) }
  }

  @Test func nestedUniformStructRequiresMemberAlignmentAndFollowingGap() throws {
    let span = try SourceSpan(startUTF8: 24, endUTF8: 30)
    let scalar = IRType(kind: .scalar(.f32))
    let inner = IRType(kind: .structure(members: [.init(type: try h(0), offset: 0)], span: 4))
    let valid = IRType(
      kind: .structure(
        members: [.init(type: try h(1), offset: 0), .init(type: try h(0), offset: 16)], span: 20))
    try ShaderSemanticValidator().validate(bufferModule([scalar, inner, valid], root: 2))
    let shortGap = IRType(
      kind: .structure(
        members: [
          .init(type: try h(1), offset: 0),
          .init(type: try h(0), offset: 4, location: .source(span)),
        ], span: 8))
    try ShaderSemanticValidator().validate(
      bufferModule([scalar, inner, shortGap], root: 2, space: .storage(access: .load)))
    do {
      try ShaderSemanticValidator().validate(bufferModule([scalar, inner, shortGap], root: 2))
      Issue.record("Uniform nested-struct gap accepted")
    } catch ShaderFailure.diagnostics(let ds) {
      #expect(ds.first?.code == .invalidType)
      #expect(ds.first?.span == span)
    }
    let unaligned = IRType(
      kind: .structure(
        members: [.init(type: try h(0), offset: 0), .init(type: try h(1), offset: 4)], span: 8))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(bufferModule([scalar, inner, unaligned], root: 2))
    }
  }

  @Test func uniformViolationsPropagateThroughArraysAndStructures() throws {
    let scalar = IRType(kind: .scalar(.f32))
    let badArray = IRType(kind: .array(base: try h(0), size: .constant(2), stride: 4))
    let holder = IRType(kind: .structure(members: [.init(type: try h(1), offset: 0)], span: 16))
    let outer = IRType(kind: .array(base: try h(2), size: .constant(2), stride: 16))
    try ShaderSemanticValidator().validate(
      bufferModule([scalar, badArray, holder, outer], root: 3, space: .storage(access: .load)))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        bufferModule([scalar, badArray, holder, outer], root: 3))
    }
    let runtime = IRType(kind: .array(base: try h(0), size: .dynamic, stride: 16))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(bufferModule([scalar, runtime], root: 1))
    }
  }
}

extension ShaderSemanticValidationTests {
  @Test func uniformAtomicDoesNotAcquireWitness() throws {
    let m = try bufferModule([IRType(kind: .atomic(.u32))], root: 0)
    #expect(throws: ShaderFailure.self) {
      try ValidatedModule.checking(m, with: ShaderSemanticValidator())
    }
  }
}

extension ShaderSemanticValidationTests {
  @Test func atomicAggregatesHaveAddressSpaceSpecificLegality() throws {
    let ts = [
      IRType(kind: .atomic(.u32)),
      IRType(kind: .array(base: try h(0), size: .constant(2), stride: 4)),
      IRType(kind: .structure(members: [.init(type: try h(1), offset: 0)], span: 8)),
    ]
    let validator = ShaderSemanticValidator()
    for root in 0...2 {
      try validator.validate(
        bufferModule(ts, root: root, space: .storage(access: [.load, .store])))
      let work = IRModule(
        types: IRTable(ts),
        globalVariables: IRTable([
          .init(space: .workgroup, type: try h(root))
        ]))
      try validator.validate(work)
      for space: IRAddressSpace in [.private, .uniform, .storage(access: .load)] {
        let bound = space == .private ? nil : IRResourceBinding(group: 0, binding: 0)
        let invalid = IRModule(
          types: IRTable(ts),
          globalVariables: IRTable([
            .init(space: space, binding: bound, type: try h(root))
          ]))
        #expect(throws: ShaderFailure.self) { try validator.validate(invalid) }
      }
    }
  }

  @Test func boolIsConstructibleButNeverHostShareableTransitively() throws {
    let ts = [
      IRType(kind: .scalar(.bool)),
      IRType(kind: .array(base: try h(0), size: .constant(2), stride: 1)),
      IRType(kind: .structure(members: [.init(type: try h(1), offset: 0)], span: 2)),
    ]
    for space: IRAddressSpace in [.private, .workgroup] {
      try ShaderSemanticValidator().validate(
        IRModule(
          types: IRTable(ts),
          globalVariables: IRTable([
            .init(space: space, type: try h(2))
          ])))
    }
    for space: IRAddressSpace in [.uniform, .storage(access: [.load, .store])] {
      #expect(throws: ShaderFailure.self) {
        try ShaderSemanticValidator().validate(bufferModule(ts, root: 2, space: space))
      }
    }
  }

  @Test func runtimeArraysCannotBeHiddenInsideNestedUnsizedStructures() throws {
    let span = try SourceSpan(startUTF8: 11, endUTF8: 17)
    let ts = [
      IRType(kind: .scalar(.u32)),
      IRType(kind: .array(base: try h(0), size: .dynamic, stride: 4)),
      IRType(kind: .structure(members: [.init(type: try h(1), offset: 0)], span: 4)),
    ]
    let validator = ShaderSemanticValidator()
    try validator.validate(bufferModule(ts, root: 1, space: .storage(access: .load)))
    try validator.validate(bufferModule(ts, root: 2, space: .storage(access: .load)))
    let nested = IRType(
      kind: .structure(
        members: [.init(type: try h(2), offset: 0, location: .source(span))], span: 4))
    do {
      try validator.validate(bufferModule(ts + [nested], root: 3, space: .storage(access: .load)))
      Issue.record("Nested unsized struct acquired witness")
    } catch ShaderFailure.diagnostics(let ds) {
      #expect(ds.first?.code == .invalidType)
      #expect(ds.first?.span == span)
    }
    let array = IRType(kind: .array(base: try h(2), size: .constant(2), stride: 4))
    #expect(throws: ShaderFailure.self) {
      try validator.validate(IRModule(types: IRTable(ts + [array])))
    }
    let nonfinal = IRType(
      kind: .structure(
        members: [.init(type: try h(1), offset: 0), .init(type: try h(0), offset: 4)], span: 8))
    #expect(throws: ShaderFailure.self) {
      try validator.validate(IRModule(types: IRTable(ts + [nonfinal])))
    }
    for space: IRAddressSpace in [.uniform, .private, .workgroup] {
      let binding: IRResourceBinding? = space == .uniform ? .init(group: 0, binding: 0) : nil
      #expect(throws: ShaderFailure.self) {
        try validator.validate(
          IRModule(
            types: IRTable(ts),
            globalVariables: IRTable([.init(space: space, binding: binding, type: h(2))])))
      }
    }
  }

  @Test func pointerTargetsRespectAddressSpaceEvenWhenUnused() throws {
    let scalar = IRType(kind: .atomic(.u32))
    try ShaderSemanticValidator().validate(
      IRModule(types: IRTable([scalar, IRType(kind: .pointer(base: try h(0), space: .workgroup))])))
    for space: IRAddressSpace in [.uniform, .private, .function, .storage(access: .load)] {
      #expect(throws: ShaderFailure.self) {
        try ShaderSemanticValidator().validate(
          IRModule(types: IRTable([scalar, IRType(kind: .pointer(base: h(0), space: space))])))
      }
    }
  }

  @Test func nonconstructibleValuesCannotBeZeroedOrPassedByValue() throws {
    let ts = [IRType(kind: .atomic(.u32))]
    let f = IRFunction(
      localVariables: IRTable([IRLocalVariable(type: try h(0))]), blocks: IRTable([IRBlock()]),
      body: try h(0))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(IRModule(types: IRTable(ts), functions: IRTable([f])))
    }
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        IRModule(
          types: IRTable(ts),
          globalExpressions: IRTable([IRExpression(type: h(0), kind: .zeroValue(h(0)))])))
    }
  }
}

extension ShaderSemanticValidationTests {
  func ioModule(
    _ types: [IRType], input: Int? = nil, output: Int? = nil, stage: IRStage = .fragment,
    extra: [IRFunctionArgument] = []
  ) throws -> IRModule {
    let args = try input.map { [IRFunctionArgument(type: try h($0))] } ?? []
    let result = try output.map { IRFunctionResult(type: try h($0)) }
    let es = try output.map { [IRExpression(type: try h($0), kind: .zeroValue(try h($0)))] } ?? []
    return try module(
      es, statements: [.init(kind: .returnValue(output == nil ? nil : h(0)))], types: types,
      result: result, arguments: args + extra, stage: stage)
  }
  @Test func aggregateEntryPointBindingsAreValidatedAcrossMembersAndArguments() throws {
    let loc: IRBinding = .location(
      index: 0, interpolation: nil, sampling: nil, blendSource: nil, perPrimitive: false)
    let ts = [
      IRType(kind: .scalar(.f32)),
      IRType(kind: .structure(members: [.init(type: try h(0), binding: loc, offset: 0)], span: 4)),
    ]
    try ShaderSemanticValidator().validate(ioModule(ts, input: 1))
    try ShaderSemanticValidator().validate(ioModule(ts, output: 1))
    let span = try SourceSpan(startUTF8: 9, endUTF8: 12)
    let collision = IRType(
      kind: .structure(
        members: [
          .init(type: try h(0), binding: loc, offset: 0),
          .init(type: try h(0), binding: loc, offset: 4, location: .source(span)),
        ], span: 8))
    do {
      try ShaderSemanticValidator().validate(ioModule([ts[0], collision], input: 1))
      Issue.record("Collision accepted")
    } catch ShaderFailure.diagnostics(let ds) {
      #expect(ds.first?.code == .invalidBinding)
      #expect(ds.first?.span == span)
    }
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        ioModule(ts, input: 1, extra: [.init(type: h(0), binding: loc)]))
    }
  }
  @Test func aggregateVertexPositionAndBuiltinIdentityAreChecked() throws {
    let pos: IRBinding = .builtIn(.position(invariant: false))
    let ts = [
      IRType(kind: .vector(size: .four, scalar: .f32)),
      IRType(kind: .structure(members: [.init(type: try h(0), binding: pos, offset: 0)], span: 16)),
    ]
    try ShaderSemanticValidator().validate(ioModule(ts, output: 1, stage: .vertex))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(ioModule(ts, input: 1, stage: .compute))
    }
    let dup = IRType(
      kind: .structure(
        members: [
          .init(type: try h(0), binding: pos, offset: 0),
          .init(type: try h(0), binding: .builtIn(.position(invariant: true)), offset: 16),
        ], span: 32))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(ioModule([ts[0], dup], output: 1, stage: .vertex))
    }
    let missing = IRType(kind: .structure(members: [.init(type: try h(0), offset: 0)], span: 16))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(ioModule([ts[0], missing], output: 1, stage: .vertex))
    }
  }
}

extension ShaderSemanticValidationTests {
  @Test func atomicAggregateCannotBeLoadedAsOrdinaryValue() throws {
    let ts = [
      IRType(kind: .atomic(.u32)),
      IRType(kind: .structure(members: [.init(type: try h(0), offset: 0)], span: 4)),
      IRType(kind: .pointer(base: try h(1), space: .workgroup)),
    ]
    let es = [
      IRExpression(type: try h(2), kind: .globalVariable(try h(0))),
      IRExpression(type: try h(1), kind: .load(pointer: try h(0))),
    ]
    let f = try module(es, types: ts).functions.values[0]
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(
        IRModule(
          types: IRTable(ts), globalVariables: IRTable([.init(space: .workgroup, type: h(1))]),
          functions: IRTable([f])))
    }
  }
  @Test func ioTraversalBudgetAndNestedShapesAreChecked() throws {
    let b: IRBinding = .location(
      index: 0, interpolation: nil, sampling: nil, blendSource: nil, perPrimitive: false)
    let ts = [
      IRType(kind: .scalar(.f32)),
      IRType(kind: .structure(members: [.init(type: try h(0), binding: b, offset: 0)], span: 4)),
    ]
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator(limits: ShaderLimits(depth: 1)).validate(ioModule(ts, input: 1))
    }
    let nested = IRType(kind: .structure(members: [.init(type: try h(1), offset: 0)], span: 4))
    #expect(throws: ShaderFailure.self) {
      try ShaderSemanticValidator().validate(ioModule(ts + [nested], input: 2))
    }
    let boundStruct = try ioModule(ts, extra: [.init(type: h(1), binding: b)])
    #expect(throws: ShaderFailure.self) { try ShaderSemanticValidator().validate(boundStruct) }
  }
}

extension ShaderSemanticValidationTests {
  @Test func typeLegalityCoversScalarVectorMatrixAndPointerAggregates() throws {
    for k: IRTypeKind in [
      .scalar(.u32), .scalar(.i32), .scalar(.f32), .vector(size: .two, scalar: .f32),
      .matrix(columns: .two, rows: .two, scalar: .f32),
    ] {
      for space: IRAddressSpace in [.private, .workgroup, .uniform, .storage(access: .load)] {
        let binding: IRResourceBinding? =
          space == .private || space == .workgroup ? nil : .init(group: 0, binding: 0)
        try ShaderSemanticValidator().validate(
          IRModule(
            types: IRTable([IRType(kind: k)]),
            globalVariables: IRTable([.init(space: space, binding: binding, type: try h(0))])))
      }
      let f = IRFunction(
        localVariables: IRTable([.init(type: try h(0))]), blocks: IRTable([IRBlock()]),
        body: try h(0))
      try ShaderSemanticValidator().validate(
        IRModule(types: IRTable([IRType(kind: k)]), functions: IRTable([f])))
    }
    let base = [
      IRType(kind: .scalar(.u32)), IRType(kind: .pointer(base: try h(0), space: .function)),
    ]
    let array = IRType(kind: .array(base: try h(1), size: .constant(2), stride: 4))
    let structure = IRType(kind: .structure(members: [.init(type: try h(1), offset: 0)], span: 4))
    for aggregate in [array, structure] {
      #expect(throws: ShaderFailure.self) {
        try ShaderSemanticValidator().validate(IRModule(types: IRTable(base + [aggregate])))
      }
    }
  }
}

extension ShaderSemanticValidationTests {
  func rawAccessModule(_ index: IRExpressionKind, signed: Bool = false, constantValue: IRLiteral? = nil,
                       dynamic: Bool = false, baseKind: Int = 0) throws -> IRModule {
    // baseKind: value vector, matrix, fixed array, pointer to fixed array, runtime storage array.
    let ts = [IRType(kind: .scalar(.f32)), IRType(kind: .vector(size: .two, scalar: .f32)),
              IRType(kind: .scalar(signed ? .i32 : .u32)),
              IRType(kind: .matrix(columns: .two, rows: .two, scalar: .f32)),
              IRType(kind: .array(base: try h(0), size: .constant(2), stride: 4)),
              IRType(kind: .pointer(base: try h(4), space: .private)),
              IRType(kind: .pointer(base: try h(0), space: .private)),
              IRType(kind: .array(base: try h(0), size: .dynamic, stride: 4)),
              IRType(kind: .pointer(base: try h(7), space: .storage(access: .load))),
              IRType(kind: .pointer(base: try h(0), space: .storage(access: .load)))]
    let typeIndex = baseKind == 0 ? 1 : baseKind == 1 ? 3 : baseKind == 2 ? 4 : baseKind == 3 ? 5 : 8
    let resultIndex = baseKind == 1 ? 1 : baseKind == 3 ? 6 : baseKind == 4 ? 9 : 0
    let base: IRExpressionKind = baseKind >= 3 ? .globalVariable(try h(0)) : .zeroValue(try h(typeIndex))
    let at = ShaderLocation.source(try SourceSpan(startUTF8: 9, endUTF8: 12))
    let es = [IRExpression(type: try h(typeIndex), kind: base),
              IRExpression(type: try h(2), kind: index),
              IRExpression(type: try h(resultIndex), kind: .access(base: try h(0), index: try h(1)), location: at)]
    let f = IRFunction(arguments: dynamic ? [.init(type: try h(2))] : [], expressions: IRTable(es),
       statements: IRTable([.init(kind: .emit([try h(2)])), .init(kind: .returnValue(nil))]),
       blocks: IRTable([IRBlock(statements: [try h(0),try h(1)])]), body: try h(0))
    let globals: [IRGlobalVariable] = baseKind == 3 ? [.init(space: .private, type: try h(4))]
      : baseKind == 4 ? [.init(space: .storage(access: .load), binding: .init(group: 0, binding: 0), type: try h(7))] : []
    return IRModule(types: IRTable(ts), constants: IRTable(try constantValue.map { _ in [.init(name: "Index", type: try h(2), initializer: try h(0))] } ?? []),
       globalVariables: IRTable(globals), globalExpressions: IRTable(try constantValue.map { [IRExpression(type: try h(2), kind: .literal($0))] } ?? []), functions: IRTable([f]))
  }
  @Test func rawAccessLiteralAndNamedConstantBoundsCannotForgeWitness() throws {
    for base in 0...3 {
      for index: IRLiteral in [.u32(2), .u32(UInt32.max), .i32(-1), .i32(2)] {
        let signed: Bool = {if case .i32 = index {return true};return false}()
        for named in [false,true] {
          let m = try rawAccessModule(named ? .constant(h(0)) : .literal(index), signed: signed,
                                      constantValue: named ? index : nil, baseKind: base)
          do { _ = try ValidatedModule.checking(m, with: ShaderSemanticValidator()); Issue.record("OOB raw Access witness") }
          catch ShaderFailure.diagnostics(let ds) {
            #expect(ds.first?.code == .typeMismatch)
            #expect(ds.first?.span == (try SourceSpan(startUTF8: 9, endUTF8: 12)))
          }
        }
      }
    }
  }
  @Test func rawAccessValidConstantZeroAndDynamicIndicesStillValidate() throws {
    for base in 0...4 {
      for index: IRExpressionKind in [.literal(.u32(0)), .literal(.u32(1)), .zeroValue(try h(2))] {
        try ShaderSemanticValidator().validate(rawAccessModule(index, baseKind: base))
      }
      try ShaderSemanticValidator().validate(rawAccessModule(.functionArgument(0), dynamic: true, baseKind: base))
    }
    // Runtime length is unknown, not zero or the minimum footprint's element count.
    try ShaderSemanticValidator().validate(rawAccessModule(.literal(.u32(500)), baseKind: 4))
    #expect(throws: ShaderFailure.self) {try ShaderSemanticValidator().validate(rawAccessModule(.literal(.i32(-1)), signed: true, baseKind: 4))}
  }
}

extension ShaderSemanticValidationTests {
  @Test func rawMatrixComposeRequiresExactColumnTypeAndCount() throws {
    for cols in [IRVectorSize.two,.three,.four] {
      for rows in [IRVectorSize.two,.three,.four] {
        let ts = [IRType(kind: .scalar(.f32)), IRType(kind: .vector(size: rows, scalar: .f32)),
                  IRType(kind: .matrix(columns: cols, rows: rows, scalar: .f32)),
                  IRType(kind: .vector(size: rows == .two ? .three : .two, scalar: .f32)),
                  IRType(kind: .vector(size: rows, scalar: .i32))]
        for variant in 0...4 {
          let count = Int(cols.rawValue) + (variant == 1 ? -1 : variant == 2 ? 1 : 0)
          let component = variant == 3 ? 3 : variant == 4 ? 4 : 1
          let es = [IRExpression(type: try h(component), kind: .zeroValue(try h(component))),
                    IRExpression(type: try h(2), kind: .compose(type: try h(2), components: Array(repeating: try h(0), count: count)))]
          let f = IRFunction(expressions: IRTable(es), statements: IRTable([.init(kind: .emit([try h(1)])), .init(kind: .returnValue(nil))]), blocks: IRTable([IRBlock(statements: [try h(0),try h(1)])]), body: try h(0))
          let m = IRModule(types: IRTable(ts), functions: IRTable([f]))
          if variant == 0 {try ShaderSemanticValidator().validate(m)}
          else {#expect(throws: ShaderFailure.self) {try ShaderSemanticValidator().validate(m)}}
        }
      }
    }
  }
}

extension ShaderSemanticValidationTests {
 @Test func rawAccessOpcodeStillRejectsStructsEvenWithKnownIndex() throws {
  let types = [IRType(kind:.scalar(.u32)),IRType(kind:.structure(members:[.init(type:try h(0),offset:0)],span:4))]
  let e = [IRExpression(type:try h(1),kind:.zeroValue(try h(1))),IRExpression(type:try h(0),kind:.literal(.u32(0))),IRExpression(type:try h(0),kind:.access(base:try h(0),index:try h(1)))]
  #expect(throws:ShaderFailure.self) {try ShaderSemanticValidator().validate(module(e,types:types))}
 }
 @Test func rawAccessAndMatrixValidationBudgetsAndCancellation() async throws {
  let m = try rawAccessModule(.constant(h(0)),constantValue:.u32(1))
  for limits in [try ShaderLimits(work:0),try ShaderLimits(nodes:2),try ShaderLimits(depth:0)] {
   #expect(throws:ShaderFailure.self) {try ShaderSemanticValidator(limits:limits).validate(m)}
  }
  let t = Task {while !Task.isCancelled {await Task.yield()}; try ShaderSemanticValidator().validate(m)}
  t.cancel();await #expect(throws:CancellationError.self) {try await t.value}
 }
}

extension ShaderSemanticValidationTests {
 @Test func rawVectorComposeCountsScalarAndVectorComponentsLikePinnedNaga() throws {
  let ts = [IRType(kind:.scalar(.f32)),IRType(kind:.vector(size:.two,scalar:.f32)),IRType(kind:.vector(size:.four,scalar:.f32)),IRType(kind:.vector(size:.two,scalar:.i32))]
  for parts in [[0,1,0], [1,1], [0,0,0,0], [1], [1,1,0], [3,1]] {
   var es:[IRExpression]=[]
   for part in parts {es.append(.init(type:try h(part),kind:.zeroValue(try h(part))))}
   es.append(.init(type:try h(2),kind:.compose(type:try h(2),components:try parts.indices.map {try h($0)})))
   let m=try module(es,types:ts)
   if parts == [0,1,0] || parts == [1,1] || parts == [0,0,0,0] {try ShaderSemanticValidator().validate(m)}
   else {#expect(throws:ShaderFailure.self) {try ShaderSemanticValidator().validate(m)}}
  }
 }
}

extension ShaderSemanticValidationTests {
 @Test func rawScalarVectorOperatorsDoNotAdmitMatrixOrBoolXorWitnesses() throws {
  let ts = [IRType(kind:.scalar(.bool)),IRType(kind:.matrix(columns:.two,rows:.two,scalar:.f32)),IRType(kind:.vector(size:.two,scalar:.bool)),IRType(kind:.vector(size:.two,scalar:.f32))]
  for (ty,operation) in [(1,0),(1,1),(1,2),(0,3)] {
   let e:IRExpressionKind
   if operation == 0 {e = .unary(op:.negate,expression:try h(0))}
   else if operation == 1 {e = .derivative(axis:.x,control:.none,expression:try h(0))}
   else if operation == 2 {e = .select(condition:try h(1),accept:try h(0),reject:try h(0))}
   else {e = .binary(op:.exclusiveOr,left:try h(0),right:try h(0))}
   let es = [IRExpression(type:try h(ty),kind:.zeroValue(try h(ty))),IRExpression(type:try h(0),kind:.literal(.bool(true))),IRExpression(type:try h(ty),kind:e)]
   #expect(throws:ShaderFailure.self) {try ShaderSemanticValidator().validate(module(es,types:ts,stage:.fragment))}
  }
  let es = [IRExpression(type:try h(2),kind:.zeroValue(try h(2))),IRExpression(type:try h(3),kind:.zeroValue(try h(3))),IRExpression(type:try h(3),kind:.select(condition:try h(0),accept:try h(1),reject:try h(1)))]
  try ShaderSemanticValidator().validate(module(es,types:ts))
 }
}
