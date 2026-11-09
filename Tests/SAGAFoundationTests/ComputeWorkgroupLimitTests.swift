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
    if format == .wgsl {
        let large = ShaderInput.wgsl("@compute @workgroup_size(4294967295u) fn run() {}")
        let shader = try compiler.compileCompute(large, to: format)
        guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.invalidHandle }
        let module = try compiler.parseAndValidate(.wgsl(text)).module
        #expect(module.entryPoints.first?.workgroupSize.x == UInt32.max)
    }
}

#if canImport(Metal)
@Test(arguments: [false, true])
func computeWorkgroupLimitsUseNativeThreadgroupDimensionsOnMetal(_ useDepth: Bool) async throws {
    let lanes: UInt32 = useDepth ? 65 : 512
    let dimensions = useDepth ? "1u, 1u, 65u" : "512u"
    let source = ShaderInput.wgsl("""
        @group(0) @binding(0) var<storage, read_write> output: u32;
        @compute @workgroup_size(\(dimensions))
        fn run(@builtin(local_invocation_index) lane: u32) {
            if lane == \(lanes - 1)u { output = lane + 1u; }
        }
        """)
    let shader = try IndependentShaderCompiler().compileCompute(source, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let limit = useDepth ? device.capabilities.maxComputeWorkgroupSize.z : device.capabilities.maxComputeWorkgroupSize.x
        if lanes > min(limit, device.capabilities.maxComputeInvocationsPerWorkgroup) {
            await #expect(throws: GPUError.validation("Workgroup size exceeds the device limits")) {
                try await device.createComputePipeline(shader)
            }
            await device.close()
            return
        }
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[lanes]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
