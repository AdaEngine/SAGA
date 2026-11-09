import Testing
import SAGAShaderIR
import SAGAFrontends

// Independently authored binary fixtures. Numeric encoding: SPIR-V universal grammar,
// vulkan-sdk-1.4.341.0; no upstream shader/test bytes or expected IR copied.
@Suite struct SPIRVFrontendTests {
    static func i(_ op: UInt32, _ a: UInt32...) -> [UInt32] { [UInt32(a.count + 1) << 16 | op] + a }
    static let head: [UInt32] = [0x07230203, 0x00010300, 0, 100, 0]
    static var pre: [UInt32] { head + i(17, 1) + i(14, 0, 1) }
    static var compute: [UInt32] {
        pre + i(15, 5, 10, 0x6e69616d, 0) + i(16, 10, 17, 2, 3, 4)
        + i(19, 1) + i(33, 2, 1) + i(54, 1, 10, 0, 2) + i(248, 11) + i(253) + i(56)
    }
    func parse(_ words: [UInt32], options: SPIRVFrontendOptions = .init()) throws -> IRModule {
        try SPIRVFrontend(options: options).parse(.spirv(words))
    }
    func fails(_ words: [UInt32], code: ShaderDiagnosticCode = .invalidBinary) throws {
        do { _ = try parse(words); Issue.record("accepted malformed words") }
        catch ShaderFailure.diagnostics(let ds) {
            #expect(ds.first?.code == code)
            guard case .binary(let span) = ds.first?.location else { Issue.record("missing word span"); return }
            #expect(span.startWord >= 0 && span.endWord <= words.count)
        }
    }
    @Test func actualComputeHeaderEntrySizeAndEndian() throws {
        let m = try parse(Self.compute)
        #expect(m.entryPoints.count == 1)
        #expect(m.entryPoints[0].name == "main")
        #expect(m.entryPoints[0].workgroupSize == IRWorkgroupSize(x: 2, y: 3, z: 4))
        #expect(m.functions.count == 1)
        #expect(try parse(Self.compute.map(\.byteSwapped)) == m)
    }
    @Test func spirvRepeatedDebugNamesKeepTheLastName() throws {
        let instructions = [
            Self.i(15, 5, 10, 0x6e69616d, 0), Self.i(16, 10, 17, 1, 1, 1),
            Self.i(5, 10, 0x00646c6f), Self.i(5, 10, 0x0077656e), // old, new
            Self.i(6, 4, 0, 0x00646c6f), Self.i(6, 4, 0, 0x0077656e),
            Self.i(19, 1), Self.i(33, 2, 1), Self.i(21, 3, 32, 0), Self.i(30, 4, 3),
            Self.i(54, 1, 10, 0, 2), Self.i(248, 11), Self.i(253), Self.i(56),
        ]
        let words = Self.pre + instructions.flatMap { $0 }
        let module = try parse(words)
        #expect(module.functions.values.contains { $0.name == "new" })
        #expect(module.types.values.contains {
            if case .structure(let members, _) = $0.kind { return members.first?.name == "new" }
            return false
        })
    }
    @Test func actualArithmeticAndPrivateStore() throws {
        let w = Self.pre + Self.i(19,1) + Self.i(21,2,32,0) + Self.i(32,3,6,2)
          + Self.i(43,2,4,7) + Self.i(43,2,5,9) + Self.i(59,3,6,6) + Self.i(33,7,1)
          + Self.i(54,1,10,0,7) + Self.i(248,11) + Self.i(128,2,12,4,5)
          + Self.i(62,6,12) + Self.i(253) + Self.i(56)
        let m = try parse(w); let f = m.functions.values[0]
        #expect(m.constants.count == 2 && m.globalVariables.count == 1)
        #expect(f.expressions.values.contains { if case .binary(.add,_,_) = $0.kind { true } else { false } })
        #expect(f.statements.values.contains { if case .store = $0.kind { true } else { false } })
    }
    @Test func structuredSelectionPhiHasRealEdgeStores() throws {
        let w = Self.pre + Self.i(19,1) + Self.i(20,2) + Self.i(21,3,32,0)
          + Self.i(41,2,4) + Self.i(43,3,5,7) + Self.i(43,3,6,9) + Self.i(33,7,3)
          + Self.i(54,3,10,0,7) + Self.i(248,11) + Self.i(247,14,0) + Self.i(250,4,12,13)
          + Self.i(248,12) + Self.i(249,14) + Self.i(248,13) + Self.i(249,14)
          + Self.i(248,14) + Self.i(245,3,15,5,12,6,13) + Self.i(254,15) + Self.i(56)
        let m = try parse(w); let f = m.functions.values[0]
        #expect(f.localVariables.count >= 1)
        #expect(f.statements.values.filter { if case .store = $0.kind { true } else { false } }.count == 2)
        #expect(f.statements.values.contains { if case .ifCondition = $0.kind { true } else { false } })
        #expect(f.expressions.values.contains { if case .load = $0.kind { true } else { false } })
    }
    @Test func headerWordCountsIDsAndOrderRejected() throws {
        for w in [[], [0x07230203], [0,0,0,0,0], [0x07230203,0x00020000,0,4,0],
                  [0x07230203,0x00010300,0,0,0], Self.head + [0], Self.head + [0xffff0001],
                  Self.pre + Self.i(19,1) + Self.i(19,1), Self.pre + Self.i(19,100),
                  Self.pre + Self.i(19,1) + Self.i(17,1)] { try fails(w) }
    }
    @Test func unsupportedNeverReturnsDummyAndCoordinateNotIgnored() throws {
        try fails(Self.pre + Self.i(9999), code: .unsupportedConstruct)
        try fails(Self.head + Self.i(17,6) + Self.i(14,0,1), code: .unsupportedCapability)
        // Vertex IO/Position is explicitly unavailable rather than silently not flipped.
        let w = Self.pre + Self.i(22,1,32) + Self.i(23,2,1,4) + Self.i(32,3,3,2) + Self.i(59,3,4,3)
        try fails(w, code: .unsupportedConstruct)
    }
    @Test func sparseHugeBoundAndBudgetsCancellation() async throws {
        var huge = Self.compute; huge[3] = UInt32.max
        #expect(try parse(huge).entryPoints.count == 1) // sparse IDs, no bound-sized allocation
        let small = try ShaderLimits(words: 10)
        #expect(throws: ShaderFailure.self) { try SPIRVFrontend(limits: small).parse(.spirv(Self.compute)) }
        let work = try ShaderLimits(work: 1)
        #expect(throws: ShaderFailure.self) { try SPIRVFrontend(limits: work).parse(.spirv(Self.compute)) }
        let t = Task { while !Task.isCancelled { await Task.yield() }; return try SPIRVFrontend().parse(.spirv(Self.compute)) }
        t.cancel(); await #expect(throws: CancellationError.self) { try await t.value }
    }
    @Test func deterministicMalformedWordFuzz() throws {
        let limits = try ShaderLimits(words: 512, tokens: 512, nodes: 512, depth: 16, work: 8192)
        let frontend = SPIRVFrontend(limits: limits)
        var state: UInt32 = 0x5a6a15
        for n in 0..<400 {
            var w = Self.compute
            state = state &* 1664525 &+ 1013904223
            if n % 3 == 0 { w = Array(w.prefix(Int(state % UInt32(w.count)))) }
            else { w[Int(state % UInt32(w.count))] = state }
            do { _ = try frontend.parse(.spirv(w)) } catch is ShaderFailure { }
        }
    }
}


extension SPIRVFrontendTests {
    @Test func invalidDecorationTargetsAndModesAreNotSilentlyIgnored() throws {
        try fails(Self.pre + Self.i(71, 1, 2) + Self.i(21, 1, 32, 0))
        try fails(Self.pre + Self.i(72, 1, 0, 35, 0) + Self.i(21, 1, 32, 0))
        try fails(Self.pre + Self.i(71, 1, 33, 0) + Self.i(21, 1, 32, 0))
        var w = Self.compute
        let mode = Self.i(16, 10, 9)
        w.insert(contentsOf: mode, at: 22) // after compute LocalSize, before declarations
        try fails(w)
    }
    @Test func relaxedCapabilityPolicyIsExplicitlyRejected() throws {
        do {
            _ = try parse(Self.compute, options: .init(strictCapabilities: false))
            Issue.record("unimplemented option was ignored")
        } catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .unsupportedConstruct) }
    }
    @Test func malformedUTF8NamesAndMissingFragmentOriginAreRejected() throws {
        try fails(Self.pre + Self.i(5, 1, 0x000080c0) + Self.i(21, 1, 32, 0))
        let w = Self.pre + Self.i(15,4,10,0x6e69616d,0) + Self.i(19,1) + Self.i(33,2,1)
            + Self.i(54,1,10,0,2) + Self.i(248,11) + Self.i(253) + Self.i(56)
        try fails(w)
    }
}

extension SPIRVFrontendTests {
    static var resource: [UInt32] {
        pre + i(71,3,2) + i(72,3,0,35,0) + i(71,6,34,0) + i(71,6,33,1)
          + i(19,1) + i(21,2,32,0) + i(30,3,2) + i(32,4,12,3) + i(32,5,12,2)
          + i(59,4,6,12) + i(43,2,7,0) + i(43,2,8,42) + i(33,9,1)
          + i(54,1,10,0,9) + i(248,11) + i(65,5,12,6,7) + i(62,12,8) + i(253) + i(56)
    }
    @Test func resourceAccessChainPreservesBindingAndMemberStore() throws {
        let m = try parse(Self.resource)
        #expect(m.globalVariables.values[0].binding == IRResourceBinding(group:0,binding:1))
        #expect(m.functions.values[0].expressions.values.contains { if case .accessIndex(_,0) = $0.kind {true} else {false} })
        #expect(m.functions.values[0].statements.values.contains { if case .store = $0.kind {true} else {false} })
    }
    @Test func compositeConstructAndExtractAreTyped() throws {
        let w = Self.pre + Self.i(19,1) + Self.i(21,2,32,0) + Self.i(23,3,2,2)
          + Self.i(43,2,4,7) + Self.i(43,2,5,9) + Self.i(33,6,2)
          + Self.i(54,2,10,0,6) + Self.i(248,11) + Self.i(80,3,12,4,5) + Self.i(81,2,13,12,1)
          + Self.i(254,13) + Self.i(56)
        let m=try parse(w)
        #expect(m.functions.values[0].expressions.values.contains { if case .compose = $0.kind {true} else {false} })
        #expect(m.functions.values[0].expressions.values.contains { if case .accessIndex(_,1) = $0.kind {true} else {false} })
    }
    @Test func resourceWrongPointeeOrIndexRejects() throws {
        var w = Self.resource
        // replace constant zero used as struct index with out-of-range one
        for j in 5..<w.count where w[j] == (4 << 16 | 43) && w[j+2] == 7 { w[j+3]=1 }
        try fails(w)
    }
}


extension SPIRVFrontendTests {
    static func loopModule() -> [UInt32] {
        var w=pre
        for x in [i(19,1),i(20,2),i(21,3,32,0),i(43,3,4,0),i(43,3,5,1),i(43,3,6,3),i(33,7,3),i(54,3,10,0,7),
                  i(248,11),i(249,12),i(248,12),i(245,3,20,4,11,22,15),i(176,2,21,20,6),i(246,16,15,0),i(250,21,14,16),
                  i(248,14),i(249,15),i(248,15),i(128,3,22,20,5),i(249,12),i(248,16),i(254,20),i(56)] {w += x}
        return w
    }
    @Test func canonicalLoopPhiBecomesLoopAndContinuingStores() throws {
        let f=try parse(Self.loopModule()).functions.values[0]
        #expect(f.statements.values.contains {if case .loop = $0.kind {true}else{false}})
        #expect(f.statements.values.contains {if case .break = $0.kind {true}else{false}})
        #expect(f.statements.values.filter {if case .store = $0.kind {true}else{false}}.count == 2)
    }
    @Test func compositeExtractionAndBackwardFunctionCalls() throws {
        var w=Self.pre
        for x in [Self.i(21,1,32,0),Self.i(23,2,1,2),Self.i(43,1,3,7),Self.i(43,1,4,9),Self.i(33,5,1),
                  Self.i(54,1,10,0,5),Self.i(248,11),Self.i(80,2,12,3,4),Self.i(81,1,13,12,1),Self.i(254,13),Self.i(56),
                  Self.i(54,1,20,0,5),Self.i(248,21),Self.i(57,1,22,10),Self.i(254,22),Self.i(56)] {w += x}
        let m=try parse(w)
        #expect(m.functions.count==2)
        #expect(m.functions.values[0].expressions.values.contains {if case .accessIndex(_,1) = $0.kind {true}else{false}})
        #expect(m.functions.values[1].statements.values.contains {if case .call = $0.kind {true}else{false}})
    }
    @Test func malformedPhiCFGAndUndefinedValuesRejected() throws {
        var bad=Self.loopModule()
        // The header phi cannot name an ID not defined by any predecessor.
        let phi=(0..<bad.count).first {bad[$0] & 0xffff == 245 && bad[$0]>>16 == 7}!
        bad[phi+3]=99
        try fails(bad)
        var w=Self.pre
        for x in [Self.i(19,1),Self.i(21,2,32,0),Self.i(33,3,1),Self.i(54,1,10,0,3),Self.i(248,11),Self.i(128,2,12,98,99),Self.i(253),Self.i(56)] {w += x}
        try fails(w)
    }
}

extension SPIRVFrontendTests {
    static var countedLoop: [UInt32] {
        pre + i(19,1) + i(20,2) + i(21,3,32,0) + i(43,3,4,0) + i(43,3,5,1) + i(43,3,6,4)
          + i(33,7,1) + i(54,1,10,0,7) + i(248,11) + i(249,12)
          + i(248,12) + i(245,3,20,4,11,22,14) + i(176,2,21,20,6) + i(246,15,14,0)
          + i(250,21,13,15) + i(248,13) + i(249,14)
          + i(248,14) + i(128,3,22,20,5) + i(249,12)
          + i(248,15) + i(253) + i(56)
    }
    @Test func canonicalLoopHasLoopContinuingBreakAndPhiStores() throws {
        let f=try parse(Self.countedLoop).functions.values[0]
        #expect(f.statements.values.contains {if case .loop = $0.kind {true} else {false}})
        #expect(f.statements.values.contains {if case .break = $0.kind {true} else {false}})
        #expect(f.statements.values.contains {if case .continue = $0.kind {true} else {false}})
        #expect(f.statements.values.filter {if case .store = $0.kind {true} else {false}}.count == 2)
    }
    @Test func exactUnknownOpcodeDiagnosticAndDepthBudget() throws {
        do {_ = try parse(Self.pre + Self.i(9999));Issue.record("accepted unknown")}
        catch ShaderFailure.diagnostics(let ds) {
            #expect(ds.first?.code == .unsupportedConstruct)
            #expect(ds.first?.location == .binary(try BinaryWordSpan(startWord:10,endWord:11)))
        }
        let limits=try ShaderLimits(depth:0)
        #expect(throws:ShaderFailure.self) {try SPIRVFrontend(limits:limits).parse(.spirv(Self.compute))}
    }
}
