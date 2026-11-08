import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .glsl, .hlsl])
func computeWorkgroupLimitsSeparateTranslationFromDeviceLimits(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let dimensions: [(String, IRWorkgroupSize)] = [
        ("1, 1, 65", .init(x: 1, y: 1, z: 65)),
        ("32, 33", .init(x: 32, y: 33, z: 1)),
    ]
    for (attribute, expected) in dimensions {
        let input = ShaderInput.wgsl("@compute @workgroup_size(\(attribute)) fn run() {}")
        if format == .hlsl {
            #expect(throws: ShaderFailure.self) { try compiler.compileCompute(input, to: format) }
        } else {
            let shader = try compiler.compileCompute(input, to: format)
            #expect(shader.workgroupSize == expected)
            if format == .wgsl, case .text(_, let source) = shader.artifact {
                _ = try compiler.parseAndValidate(.wgsl(source))
            }
        }
    }
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(.wgsl("@compute @workgroup_size(65536u, 65536u) fn run() {}"), to: format)
    }
}

#if canImport(Metal)
@Test func computeWorkgroupLimitsRunA512ThreadWorkgroupOnMetal() async throws {
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(512)
        fn run(@builtin(local_invocation_index) lane: u32) {
            if lane == 511u { output = lane + 1u; }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[512]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
