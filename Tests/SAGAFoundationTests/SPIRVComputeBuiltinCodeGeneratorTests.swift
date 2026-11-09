import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func computeBuiltinExportInput(structured: Bool) -> ShaderInput {
    let inputs = [
        "@builtin(global_invocation_id) gid: vec3u",
        "@builtin(local_invocation_id) lid: vec3u",
        "@builtin(local_invocation_index) lane: u32",
        "@builtin(workgroup_id) group: vec3u",
        "@builtin(num_workgroups) counts: vec3u",
    ].joined(separator: ",\n")
    let declaration = structured ? "struct Invocation { \(inputs), }" : ""
    let arguments = structured ? "input: Invocation" : inputs
    let prefix = structured ? "input." : ""
    return .wgsl("""
        \(declaration)
        @group(0) @binding(0) var<storage, read_write> result: array<u32, 720>;
        fn pack(value: vec3u) -> u32 { return value.x + 16u * value.y + 256u * value.z; }
        @compute @workgroup_size(2, 3, 2) fn dispatchBuiltins(\(arguments)) {
            let id = \(prefix)gid;
            let groups = \(prefix)counts;
            let width = groups.x * 2u;
            let height = groups.y * 3u;
            let slot = 5u * (id.x + width * (id.y + height * id.z));
            result[slot] = pack(id);
            result[slot + 1u] = pack(\(prefix)lid);
            result[slot + 2u] = \(prefix)lane;
            result[slot + 3u] = pack(\(prefix)group);
            result[slot + 4u] = pack(groups);
        }
        """)
}

private func exportComputeBuiltins(_ compiler: IndependentShaderCompiler, structured: Bool) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(computeBuiltinExportInput(structured: structured), to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func expectComputeBuiltinInterface(_ words: [UInt32]) throws {
    try #require(words.count > 5)
    var decoratedBuiltins: [UInt32: UInt32] = [:]
    var inputs: Set<UInt32> = []
    var interfaces: [UInt32]?
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let operands = Array(words[(offset + 1)..<(offset + count)])
        switch words[offset] & 0xffff {
        case 15 where operands.count >= 3 && operands[0] == 5: // OpEntryPoint GLCompute
            let nameWordCount = ("dispatchBuiltins".utf8.count + 4) / 4
            interfaces = Array(operands.dropFirst(2 + nameWordCount))
        case 59 where operands.count >= 3 && operands[2] == 1: // OpVariable Input
            inputs.insert(operands[1])
        case 71 where operands.count == 3 && operands[1] == 11: // OpDecorate BuiltIn
            decoratedBuiltins[operands[0]] = operands[2]
        default: break
        }
        offset += count
    }
    let entryInputs = try #require(interfaces)
    #expect(inputs.count == 5)
    #expect(Set(entryInputs).count == entryInputs.count)
    #expect(inputs.isSubset(of: Set(entryInputs)))
    #expect(Set(inputs.compactMap { decoratedBuiltins[$0] }) == [24, 26, 27, 28, 29])
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvComputeBuiltinCodeGeneratorEmitsAndRoundTripsInterface(_ format: ShaderOutputFormat, _ structured: Bool) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportComputeBuiltins(compiler, structured: structured)
    try expectComputeBuiltinInterface(words)
    let input = ShaderInput.spirv(words)
    let entry = try #require(compiler.parseAndValidate(input).module.entryPoints.first)
    #expect(entry.name == "dispatchBuiltins")
    #expect(entry.stage == .compute)
    #expect(entry.workgroupSize == .init(x: 2, y: 3, z: 2))
    let shader = try compiler.compileCompute(input, entryPoint: "dispatchBuiltins", to: format)
    #expect(shader.workgroupSize == entry.workgroupSize)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [2880])
    if format == .hlsl { #expect(shader.hlslWorkgroupCountBinding != nil) }
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text))
        _ = try compiler.compileWebGPUCompute(input, entryPoint: "dispatchBuiltins")
    }
}

@Test func spirvComputeBuiltinCodeGeneratorKeepsEntryInterfacesSeparate() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> out: array<u32, 8>;
        @compute @workgroup_size(2, 1, 1)
        fn left(@builtin(global_invocation_id) id: vec3u) { out[id.x] = id.x; }
        @compute @workgroup_size(1, 2, 1)
        fn right(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
            out[4u + group.y * 2u + lane] = lane;
        }
        """)
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    let module = try compiler.parseAndValidate(.spirv(words)).module
    let left = try #require(module.entryPoints.first { $0.name == "left" })
    let right = try #require(module.entryPoints.first { $0.name == "right" })
    #expect(left.workgroupSize == .init(x: 2))
    #expect(right.workgroupSize == .init(y: 2))
    let leftArguments = try module.functions.value(for: left.function).arguments.compactMap(\.binding)
    let rightArguments = try module.functions.value(for: right.function).arguments.compactMap(\.binding)
    #expect(leftArguments == [.builtIn(.globalInvocationId)])
    #expect(rightArguments == [.builtIn(.workgroupId), .builtIn(.localInvocationIndex)])
}

#if canImport(Metal)
@Test(arguments: [false, true])
func spirvComputeBuiltinCodeGeneratorPreserves3DInvocationValuesOnMetal(_ structured: Bool) async throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportComputeBuiltins(compiler, structured: structured)
    let shader = try compiler.compileCompute(.spirv(words), entryPoint: "dispatchBuiltins", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let dispatch = DispatchWorkgroups(x: 3, y: 2, z: 2)
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 2880, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 2880, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: dispatch),
            .copy(source: output, destination: readback, byteCount: 2880), .read(buffer: readback, byteCount: 2880),
        ]).wait()
        func pack(_ x: UInt32, _ y: UInt32, _ z: UInt32) -> UInt32 { x + 16 * y + 256 * z }
        var expected: [UInt32] = []
        for z in 0..<(dispatch.z * 2) {
            for y in 0..<(dispatch.y * 3) {
                for x in 0..<(dispatch.x * 2) {
                    let (lx, ly, lz) = (x % 2, y % 3, z % 2)
                    expected += [pack(x, y, z), pack(lx, ly, lz), lx + 2 * ly + 6 * lz,
                                 pack(x / 2, y / 3, z / 2), pack(dispatch.x, dispatch.y, dispatch.z)]
                }
            }
        }
        #expect(snapshots == [expected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
