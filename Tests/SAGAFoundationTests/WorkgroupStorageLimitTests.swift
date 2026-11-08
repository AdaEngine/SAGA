import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import Metal
import SAGAMetal
#endif

private let workgroupLimitShader = """
    override scratchWords: u32 = 8188u;
    var<workgroup> scratch: array<u32, scratchWords>;
    var<workgroup> flag: u32;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) lane: u32) {
        let index = scratchWords - 1u - lane;
        let initial = scratch[index] + flag;
        workgroupBarrier();
        scratch[index] = 10u + lane;
        if lane == 0u { flag = 70u; }
        workgroupBarrier();
        output[lane] = initial * 1000u + flag + scratch[scratchWords - 1u - (lane + 1u) % 4u];
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func workgroupStorageLimitReflectsRoundedAllocationWithoutADevice(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(workgroupLimitShader)
    let exact = try compiler.compileCompute(source, to: format)
    let above = try compiler.compileCompute(source, to: format, constants: ["scratchWords": 8189])
    #expect(exact.workgroupStorageByteSize == 32_768)
    #expect(above.workgroupStorageByteSize == 32_784)
    guard case .text(let actual, let code) = exact.artifact else { throw ShaderFailure.invalidHandle }
    #expect(actual == format)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(code)) }
}

#if canImport(Metal)
@Test func workgroupStorageLimitRunsAboveTheWebGPUDefaultOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(workgroupLimitShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        if device.capabilities.maxComputeWorkgroupStorageSize < 32_768 {
            await #expect(throws: GPUError.validation("Workgroup storage exceeds the device limit")) {
                try await device.createComputePipeline(shader)
            }
            await device.close()
            return
        }
        let metal = try #require(MTLCreateSystemDefaultDevice())
        guard case .text(.msl, let code) = shader.artifact else { throw ShaderFailure.invalidHandle }
        let library = try await metal.makeLibrary(source: code, options: nil)
        let function = try #require(library.makeFunction(name: shader.entryPoint))
        let state = try await metal.makeComputePipelineState(function: function)
        if state.staticThreadgroupMemoryLength > Int(device.capabilities.maxComputeWorkgroupStorageSize) {
            await #expect(throws: GPUError.validation("Workgroup storage exceeds the device limit")) {
                try await device.createComputePipeline(shader)
            }
            await device.close()
            return
        }
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(result == [[81, 82, 83, 80]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
