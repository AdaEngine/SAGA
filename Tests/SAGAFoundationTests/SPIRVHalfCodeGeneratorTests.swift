import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

@Test func spirvHalfCodeGeneratorReparsesComputedHalfComparison() throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl("""
        enable f16;
        @group(0) @binding(0) var<storage, read_write> words: array<u32, 8>;
        var<private> state: f16 = -0.0h;
        var<workgroup> values: array<vec2h, 2>;
        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
            let input = bitcast<f32>(words[lane]);
            let halfInput = f16(input);
            let a = halfInput * 0.75h + f16(lane);
            let b = select(-a, a, a > 0.0h);
            values[lane] = vec2h(b, state);
            workgroupBarrier();
            let pair = values[1u - lane];
            let bits = bitcast<u32>(pair);
            let wide = bitcast<vec4h>(vec2u(bits, words[lane]));
            let back = bitcast<vec2u>(wide);
            words[2u + 2u * lane] = back.x;
            words[3u + 2u * lane] = bitcast<u32>(f32(wide.z));
        }
        """)
    guard case .spirv(let words) = try compiler.compile(input, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    let shader = try compiler.compileCompute(.spirv(words), to: .wgsl)
    guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
    _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderFloat16])
}

private let spirvHalfInput = ShaderInput.wgsl("""
    enable f16;
    struct Packet { low: f16, high: f16, sentinel: u32, lanes: vec3h }
    @group(0) @binding(0) var<storage, read_write> packet: Packet;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 2>;

    @compute @workgroup_size(1) fn run() {
        let converted = f16(f32(packet.low) + 0.5f);
        let increment = packet.high * 2.0h;
        let before = packet.lanes;
        packet.low = converted + increment;
        packet.lanes = before * vec3h(2.0h, -0.5h, 4.0h) + vec3h(packet.high);
        output[0] = vec4f(f32(converted), f32(increment), f32(packet.low), f32(packet.high));
        output[1] = vec4f(vec3f(packet.lanes), f32(before.z));
    }
    """)

private func exportHalf(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvHalfInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvHalfCodeGeneratorRoundTripsNativeHalfTypes(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportHalf(compiler)
    try #require(words.count > 5)
    var capabilities: Set<UInt32> = []
    var hasHalfType = false
    var opcodes: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let opcode = words[offset] & 0xffff
        opcodes.insert(opcode)
        if opcode == 17 && count == 2 { capabilities.insert(words[offset + 1]) }
        if opcode == 22 && count == 3 && words[offset + 2] == 16 { hasHalfType = true }
        offset += count
    }
    #expect(capabilities.isSuperset(of: [9, 4433])) // Float16, StorageBuffer16BitAccess
    #expect(hasHalfType)
    #expect(opcodes.isSuperset(of: [115, 129, 133])) // OpFConvert, OpFAdd, OpFMul
    let input = ShaderInput.spirv(words)
    let raw = try compiler.parseAndValidate(input, capabilities: [.shaderFloat16]).module
    #expect(raw.types.values.contains { $0.kind == .scalar(.f16) })
    let shader = try compiler.compileCompute(input, to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.storageBindings.map(\.minimumByteSize) == [16, 32])
    if format == .wgsl {
        guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
        _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderFloat16])
    }
}

@Test func spirvHalfCodeGeneratorRoundTripsRectangularMatrixWidthConversions() throws {
    let source = ShaderInput.wgsl("""
        enable f16;
        @group(0) @binding(0) var<storage, read> inputs: array<vec4f, 2>;
        @group(0) @binding(1) var<storage, read_write> output: vec4f;
        @compute @workgroup_size(1) fn run() {
            let wide = mat2x3f(inputs[0].xyz, inputs[1].xyz);
            let narrow = mat2x3h(wide);
            let scaled = narrow * 2.0h;
            let widened = mat2x3f(scaled);
            output = vec4f(widened[0], widened[1].z);
        }
        """)
    let compiler = IndependentShaderCompiler()
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    let input = ShaderInput.spirv(words)
    let raw = try compiler.parseAndValidate(input, capabilities: [.shaderFloat16]).module
    #expect(raw.types.values.contains { $0.kind == .matrix(columns: .two, rows: .three, scalar: .f16) })
    for format: ShaderOutputFormat in [.msl, .wgsl, .hlsl, .glsl] {
        let shader = try compiler.compileCompute(input, to: format)
        #expect(shader.requiresShaderF16)
        if format == .wgsl {
            guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(text), capabilities: [.shaderFloat16])
        }
    }
}

#if canImport(Metal)
@Test func spirvHalfCodeGeneratorPreservesPackedLayoutAndArithmeticOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.spirv(try exportHalf(compiler)), to: .msl)
    func halfPair(_ low: Double, _ high: Double) -> UInt32 {
        UInt32(Binary16.encode(low)) | UInt32(Binary16.encode(high)) << 16
    }
    let initial: [UInt32] = [halfPair(-2, 1.5), 0xfeedface, halfPair(1.5, -2), halfPair(0.75, -9)]
    let packetExpected: [UInt32] = [halfPair(1.5, 1.5), 0xfeedface, halfPair(4.5, 2.5), halfPair(4.5, -9)]
    let expected: [Float] = [-1.5, 3, 1.5, 1.5, 4.5, 2.5, 4.5, 0.75]
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderF16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let packet = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination, .copySource]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .write(buffer: packet, offset: 0, words: initial),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): packet, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 32), .read(buffer: readback, byteCount: 32),
            .copy(source: packet, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(snapshots == [expected.map(\.bitPattern), packetExpected])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
