import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvControlFlowInput = ShaderInput.wgsl("""
    @group(0) @binding(0) var<storage, read> source: array<u32, 3>;
    @group(0) @binding(1) var<storage, read_write> result: array<u32, 5>;

    fn classify(value: u32) -> u32 {
        if value == 7u { return 11u; }
        switch value {
            case 2u, 3u: { return 20u; }
            default: { return 30u; }
        }
    }

    @compute @workgroup_size(1) fn controlFlow() {
        var first: u32;
        if source[0u] != 0u {
            first = source[1u] + 10u;
            result[0u] = first;
        } else {
            first = source[2u] + 20u;
            result[0u] = first;
        }
        result[1u] = source[2u] + first;

        var total = 0u;
        var outer = 0u;
        loop {
            var inner = 0u;
            loop {
                switch inner {
                    case 0u, 2u: { total += 2u; continue; }
                    case 1u: { total += 10u; break; }
                    default: { total += 1u; }
                }
                if inner == 3u {
                    if outer == 0u { break; }
                }
                total += 100u;
                continuing { inner += 1u; break if inner >= 4u; }
            }
            total += 1000u;
            continuing { outer += 1u; break if outer >= 2u; }
        }
        result[2u] = total;
        result[3u] = classify(source[1u]);
        result[4u] = classify(source[2u]);
    }
    """)

private func exportedControlFlow(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvControlFlowInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvControlFlowCodeGeneratorEmitsStructuredBranchesAndRoundTrips(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportedControlFlow(compiler)
    try #require(words.count > 5)
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        opcodes.insert(words[offset] & 0xffff)
        offset += count
    }
    // OpLoopMerge, OpSelectionMerge, OpBranch, OpBranchConditional, OpSwitch, OpReturnValue.
    #expect(opcodes.isSuperset(of: [246, 247, 249, 250, 251, 254]))
    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input).module
    #expect(module.entryPoints.map(\.name) == ["controlFlow"])
    let shader = try compiler.compileCompute(input, entryPoint: "controlFlow", to: format)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [12, 20])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
        _ = try compiler.compileWebGPUCompute(input, entryPoint: "controlFlow")
    }
}

@Test func spirvControlFlowCodeGeneratorPreservesImportedSwitchFallthrough() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(SPIRVFrontendTests.switchModule())
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    let module = try compiler.parseAndValidate(.spirv(words)).module
    let entry = try #require(module.entryPoints.first)
    let function = try module.functions.value(for: entry.function)
    let switches = function.statements.values.compactMap { statement -> [IRSwitchCase]? in
        if case .switchValue(_, let cases) = statement.kind { return cases }
        return nil
    }
    let selection = try #require(switches.first { $0.contains(where: \.fallThrough) })
    #expect(selection.map(\.value) == [.i32(-1), .i32(0), .default, .i32(2), .i32(9)])
    #expect(selection.map(\.fallThrough) == [true, true, true, false, false])
}

#if canImport(Metal)
@Test func spirvControlFlowCodeGeneratorPreservesBranchAndLoopResultsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportedControlFlow(compiler)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "controlFlow", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createBuffer(.init(size: 12, usage: [.storage, .copyDestination]))
        let result = try await device.createBuffer(.init(size: 20, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 20, usage: [.copyDestination, .mapRead]))
        let inputs: [[UInt32]] = [[1, 7, 2], [0, 3, 5]]
        var commands: [BufferCommand] = []
        for words in inputs {
            commands += [
                .write(buffer: source, offset: 0, words: words),
                .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): source, .init(binding: 1): result],
                                 workgroups: .init(x: 1)),
                .copy(source: result, destination: readback, byteCount: 20), .read(buffer: readback, byteCount: 20),
            ]
        }
        let snapshots = try await device.submitCommands(commands).wait()
        #expect(snapshots == [[17, 19, 2330, 11, 20], [25, 30, 2330, 20, 30]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
