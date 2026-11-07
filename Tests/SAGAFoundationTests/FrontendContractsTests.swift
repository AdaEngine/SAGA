import Testing
import SAGAShaderIR
import SAGAFrontends

@Suite struct FrontendContractsTests {
    @Test func immutableCrossFormatModuleAndCheckedHandles() throws {
        let types = IRTable([IRType(name: "u32", kind: .scalar(.u32))])
        let u32 = try types.handle(at: 0)
        var original = [IRExpression(type: u32, kind: .literal(.u32(7)))]
        let expressions = IRTable(original)
        original.removeAll()
        let value = try expressions.handle(at: 0)
        let statements = IRTable([IRStatement(kind: .returnValue(value))])
        let blocks = IRTable([IRBlock(statements: [try statements.handle(at: 0)])])
        let function = IRFunction(name: "value", result: IRFunctionResult(type: u32), expressions: expressions,
                                  statements: statements, blocks: blocks, body: try blocks.handle(at: 0))
        let functions = IRTable([function])
        let module = IRModule(types: types, functions: functions,
                              entryPoints: [IREntryPoint(name: "main", stage: .compute, function: try functions.handle(at: 0))])
        func requireSendable<T: Sendable>(_ value: T) -> T { value }
        #expect(requireSendable(module).schemaVersion == 1)
        #expect(try module.functions.value(for: functions.handle(at: 0)).expressions.count == 1)
        #expect(try expressions.value(for: value).kind == .literal(.u32(7)))
        #expect(throws: ShaderFailure.self) { try types.handle(at: -1) }
        #expect(throws: ShaderFailure.self) { try types.handle(at: Int.max) }
        #expect(throws: ShaderFailure.self) { try types.value(for: IRHandle<IRType>(index: 4)) }
        #expect(types[try IRHandle<IRType>(index: 4)] == nil)
    }

    @Test func diagnosticsPreserveExactUnitsAndLegacyInitializer() throws {
        let source = "a\u{e9}\u{1f30f}z"
        let span = try SourceSpan(startUTF8: 1, endUTF8: 7)
        try span.validate(in: source)
        #expect(throws: ShaderFailure.self) { try SourceSpan(startUTF8: 2, endUTF8: 7).validate(in: source) }
        #expect(throws: ShaderFailure.self) { try SourceSpan(startUTF8: 0, endUTF8: Int.max).validate(in: source) }
        let words = try BinaryWordSpan(startWord: 5, endWord: 8)
        #expect(try words.byteSpan() == BinaryByteSpan(startByte: 20, endByte: 32))
        #expect(throws: ShaderFailure.self) { try BinaryWordSpan(startWord: 0, endWord: Int.max).byteSpan() }
        #expect(throws: ShaderFailure.self) { try words.validate(wordCount: 7) }
        let diagnostic = ShaderDiagnostic(severity: .error, code: .invalidBinary, message: "bad operand", location: .binary(words))
        #expect(diagnostic.code.category == .binary)
        #expect(diagnostic.location == .binary(words))
        #expect(diagnostic.span == nil)
        let legacy = ShaderDiagnostic(severity: .warning, message: "legacy", span: span)
        #expect(legacy.span == span)
        #expect(legacy.location == .source(span))
    }

    @Test func boundedAccountingNeverOverflowsAndFailsClosed() throws {
        let limits = try ShaderLimits(sourceBytes: 3, words: 2, tokens: 2, nodes: 1, depth: 1, work: 8)
        var budget = ShaderBudget(limits: limits)
        try budget.checkInput(.wgsl("é"))
        try budget.consume(.tokens, amount: 2)
        #expect(throws: ShaderFailure.self) { try budget.consume(.tokens) }
        #expect(throws: ShaderFailure.self) { try budget.consume(.work, amount: Int.max) }
        #expect(throws: ShaderFailure.self) { try budget.consume(.nodes, amount: -1) }
        try budget.enterDepth()
        #expect(throws: ShaderFailure.self) { try budget.enterDepth() }
        try budget.leaveDepth()
        #expect(throws: ShaderFailure.self) { try budget.leaveDepth() }
        #expect(throws: ShaderFailure.self) { try ShaderLimits(sourceBytes: -1) }
        var inputBudget = ShaderBudget(limits: limits)
        #expect(throws: ShaderFailure.self) { try inputBudget.checkInput(.wgsl("🌏")) }
        #expect(throws: ShaderFailure.self) { try inputBudget.checkInput(.spirv([1, 2, 3])) }
        #expect(throws: ShaderFailure.self) { try inputBudget.checkInput(.glsl("", stage: .compute, defines: ["huge": "value"])) }
    }

    @Test func cancellationIsCheckedBeforeAnyBudgetWork() async throws {
        let task = Task {
            while !Task.isCancelled { await Task.yield() }
            var budget = ShaderBudget()
            try budget.consume(.work)
        }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test func flatBlocksAndAdvancedIRAreRepresentableWithoutRecursiveValues() throws {
        let u32 = try IRHandle<IRType>(index: 0)
        let expression = try IRHandle<IRExpression>(index: 0)
        let block = try IRHandle<IRBlock>(index: 0)
        let types: [IRTypeKind] = [
            .scalar(.u32), .vector(size: .four, scalar: .f32),
            .matrix(columns: .four, rows: .three, scalar: .f32),
            .atomic(.u32), .pointer(base: u32, space: .storage(access: [.load, .store])),
            .array(base: u32, size: .dynamic, stride: 4),
            .bindingArray(base: u32, size: .constant(4)),
            .image(dimension: .d2, arrayed: false, class: .storage(format: .r32Uint, access: [.load])),
            .sampler(comparison: true), .accelerationStructure(vertexReturn: true), .rayQuery(vertexReturn: false),
            .cooperativeMatrix(columns: .eight, rows: .sixteen, scalar: .f32, role: .a)
        ]
        #expect(types.count == 12)
        let loop = IRStatement(kind: .loop(body: block, continuing: block, breakIf: expression))
        #expect(loop.location == nil)
        #expect(IRStage.allCases.count == 9)
        #expect(IRMathFunction.allCases.contains(.pack4xI8Clamp))
    }
}

extension FrontendContractsTests {
    @Test func boundedBuilderAndScopedDepthRestoreState() throws {
        var budget = ShaderBudget(limits: try ShaderLimits(nodes: 1, depth: 1, work: 8))
        var builder = IRTableBuilder<IRType>()
        let handle = try builder.append(IRType(kind: .scalar(.u32)), budget: &budget)
        let snapshot = builder.freeze()
        #expect(try snapshot.value(for: handle).kind == .scalar(.u32))
        #expect(throws: ShaderFailure.budgetExceeded(.nodes)) {
            try builder.append(IRType(kind: .scalar(.f32)), budget: &budget)
        }
        #expect(builder.count == 1)
        #expect(throws: ShaderFailure.budgetExceeded(.depth)) {
            try budget.withDepth { nested in try nested.enterDepth() }
        }
        #expect(budget.usage(.depth) == 0)
        #expect(snapshot.count == 1)
        try budget.withDepth { nested in try nested.leaveDepth() }
        #expect(budget.usage(.depth) == 0)
    }

    @Test func malformedBlockGraphsRejectCyclesDanglingIndicesAndDepth() throws {
        let block = try IRHandle<IRBlock>(index: 0)
        let statement = try IRHandle<IRStatement>(index: 0)
        let cyclic = IRFunction(statements: IRTable([IRStatement(kind: .block(block))]),
                                blocks: IRTable([IRBlock(statements: [statement])]), body: block)
        var budget = ShaderBudget()
        #expect(throws: ShaderFailure.self) { try cyclic.walkBlocks(budget: &budget) { _ in } }
        let dangling = IRFunction(body: block)
        #expect(throws: ShaderFailure.invalidHandle) { try dangling.walkBlocks(budget: &budget) { _ in } }
        let leaf = IRFunction(blocks: IRTable([IRBlock()]), body: block)
        var zeroDepth = ShaderBudget(limits: try ShaderLimits(depth: 0))
        #expect(throws: ShaderFailure.budgetExceeded(.depth)) { try leaf.walkBlocks(budget: &zeroDepth) { _ in } }
    }

    @Test func deterministicBoundaryFuzzDoesNotTrap() throws {
        for limit in 0..<32 {
            var budget = ShaderBudget(limits: try ShaderLimits(nodes: limit, work: 64))
            var table = IRTableBuilder<IRType>()
            for _ in 0..<limit { _ = try table.append(IRType(kind: .scalar(.u32)), budget: &budget) }
            #expect(throws: ShaderFailure.budgetExceeded(.nodes)) {
                try table.append(IRType(kind: .scalar(.u32)), budget: &budget)
            }
            let frozen = table.freeze()
            for invalid in [-1, Int.min, Int.max] {
                #expect(throws: ShaderFailure.invalidHandle) { try frozen.handle(at: invalid) }
            }
        }
        let text = "a\u{e9}\u{1f30f}z"
        for start in 0...9 {
            for end in start...10 {
                let span = try SourceSpan(startUTF8: start, endUTF8: end)
                let valid = [0, 1, 3, 7, 8].contains(start) && [0, 1, 3, 7, 8].contains(end)
                if valid { try span.validate(in: text) }
                else { #expect(throws: ShaderFailure.invalidSpan) { try span.validate(in: text) } }
            }
        }
    }

    @Test func frontendOptionsAndBudgetDispatchRemainExplicit() throws {
        #expect(WGSLFrontendOptions().capabilities == .allKnown)
        #expect(SPIRVFrontendOptions().adjustCoordinateSpace)
        #expect(SPIRVFrontendOptions().strictCapabilities)
        #expect(GLSLFrontendOptions(stage: .mesh, defines: ["A": "2"]).stage == .mesh)
        #expect(ShaderCapabilities.allKnown.rawValue.nonzeroBitCount == 46)
        #expect(IRStage(ShaderStage.compute) == .compute)
    }
}

/// Synthetic dispatch sentinel, deliberately not a shader implementation.
private struct ContractBudgetProbe: BoundedShaderFrontend {
    typealias Module = IRModule
    let inputFormat: ShaderInputFormat = .wgsl
    let limits: ShaderLimits
    func parse(_ input: ShaderInput, budget: inout ShaderBudget) throws -> IRModule {
        try budget.consume(.tokens)
        throw ShaderFailure.unsupported("synthetic frontend entered")
    }
}

extension FrontendContractsTests {
    @Test func wrapperEnforcesPreflightBeforeSyntheticConsumer() throws {
        let probe = ContractBudgetProbe(limits: try ShaderLimits(sourceBytes: 2))
        #expect(throws: ShaderFailure.budgetExceeded(.sourceBytes)) { try probe.parse(.wgsl("123")) }
        #expect(throws: ShaderFailure.unsupported("Frontend input format mismatch")) { try probe.parse(.spirv([])) }
        #expect(throws: ShaderFailure.unsupported("synthetic frontend entered")) { try probe.parse(.wgsl("a")) }
        #expect(try IRHandle<IRType>(rawValue: 7).index == 7)
    }

    @Test func blockTraversalPreservesOrderAndRejectsWorkExhaustion() throws {
        let root = try IRHandle<IRBlock>(index: 0)
        let child1 = try IRHandle<IRBlock>(index: 1)
        let child2 = try IRHandle<IRBlock>(index: 2)
        let statement = try IRHandle<IRStatement>(index: 0)
        let condition = try IRHandle<IRExpression>(index: 0)
        let function = IRFunction(statements: IRTable([IRStatement(kind: .ifCondition(condition: condition, accept: child1, reject: child2))]),
                                  blocks: IRTable([IRBlock(statements: [statement]), IRBlock(location: .source(try SourceSpan(startUTF8: 1, endUTF8: 2))), IRBlock(location: .source(try SourceSpan(startUTF8: 2, endUTF8: 3)))]), body: root)
        var budget = ShaderBudget()
        var order: [Int] = []
        try function.walkBlocks(budget: &budget) { block in
            if case .source(let span) = block.location { order.append(span.startUTF8) }
        }
        #expect(order == [1, 2])
        var tiny = ShaderBudget(limits: try ShaderLimits(work: 2))
        #expect(throws: ShaderFailure.budgetExceeded(.work)) { try function.walkBlocks(budget: &tiny) { _ in } }
    }
}
