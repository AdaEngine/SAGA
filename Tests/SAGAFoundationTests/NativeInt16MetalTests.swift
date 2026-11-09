import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import SAGAWebGPU
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let int16ComputeSource: ShaderInput = .wgsl("""
    enable wgpu_int16;

    @group(0) @binding(0) var<storage, read> input: array<vec4u, 2>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 2>;

    @compute @workgroup_size(1) fn main() {
        let signedValues = vec2<i16>(bitcast<vec2i>(input[0].xy));
        let unsignedValues = vec2<u16>(input[0].zw);
        let delta = i16(input[1].x);
        let step = u16(input[1].y);
        let signedSum = signedValues + vec2<i16>(delta);
        let unsignedSum = unsignedValues + vec2<u16>(step);
        output[0] = vec4u(bitcast<vec2u>(vec2i(signedSum)), vec2u(unsignedSum));
        let signedDifference = i16(input[1].z) - delta;
        let unsignedProduct = u16(input[1].w) * u16(input[1].w);
        output[1] = vec4u(bitcast<u32>(i32(signedDifference)), u32(unsignedProduct),
                          select(0u, 1u, signedValues.x > signedValues.y),
                          select(0u, 1u, unsignedValues.x > unsignedValues.y));
    }
    """)

private let int16VertexSource: ShaderInput = .wgsl("""
    enable wgpu_int16;
    @vertex fn main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        let x = i16(index) + i16(1);
        return vec4f(f32(i32(x)), 0.0, 0.0, 1.0);
    }
    """)

private let int16FragmentSource: ShaderInput = .wgsl("""
    enable wgpu_int16;
    @group(0) @binding(0) var<uniform> input: vec4u;
    @fragment fn main() -> @location(0) vec4u {
        return vec4u(vec4<u16>(input) + vec4<u16>(u16(1)));
    }
    """)

private actor Int16AdvertisingBrowserHost: BrowserGPUComputeHost {
    nonisolated let capabilities: DeviceCapabilities
    private(set) var allocated = false

    init() throws {
        capabilities = try DeviceCapabilities(maxBufferSize: 4096, features: [.shaderInt16])
    }

    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) {}
    func destroyBuffer(id: ResourceID) {}
    func submit(_ submission: Submission) {}
    func submitCommands(_ submission: Submission, commands: [BufferCommand]) -> [[UInt32]] { [] }
    func allocateComputePipeline(_ pipeline: ComputePipeline, shader: ComputeShaderArtifact) { allocated = true }
    func destroyComputePipeline(id: ResourceID) {}
    func close(owner: UInt64) {}
}

@Test func nativeInt16RequiresCapabilityBeforePipelineAllocation() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(int16ComputeSource, to: .msl)
    let vertex = try compiler.compileRenderStage(int16VertexSource, entryPoint: "main", to: .msl)
    let fragment = try compiler.compileRenderStage(int16FragmentSource, entryPoint: "main", to: .msl)
    #expect(shader.requiresShaderInt16)
    #expect(vertex.requiresShaderInt16)
    #expect(fragment.requiresShaderInt16)
    #expect(!shader.requiresShaderF16 && !shader.requiresShaderInt64)
    let backend = RenderTestBackend()
    let device = try await Device.make(backend: backend)
    await #expect(throws: GPUError.validation("Feature not advertised")) {
        try await device.createComputePipeline(shader)
    }
    await #expect(throws: GPUError.validation("Feature not advertised")) {
        try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
    }
    #expect(await backend.allocatedRenderPipelines == 0)
    await device.close()
}

@Test func browserBackendRejectsNativeInt16EvenWhenHostAdvertisesIt() async throws {
    let compiler = IndependentShaderCompiler()
    let native = try compiler.compileCompute(int16ComputeSource, to: .wgsl)
    #expect(native.requiresShaderInt16)
    let host = try Int16AdvertisingBrowserHost()
    let device = try await Device.make(backend: BrowserComputeBackend(host: host))
    await #expect(throws: GPUError.unsupported("Browser WebGPU does not support native 16-bit shader integers")) {
        try await device.createComputePipeline(native)
    }
    #expect(await !host.allocated)
    await device.close()
}

#if canImport(Metal)
@Test func nativeInt16ArithmeticAndConversionsRunOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(int16ComputeSource, to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 32, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let words: [UInt32] = [0x7fff, 0xffff8000, 0xffff, 0x1234, 1, 2, 0x8000, 0xffff]
        let snapshots = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: words),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): input, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ]).wait()
        #expect(snapshots == [[0xffff8000, 0xffff8001, 1, 0x1236, 0x7fff, 1, 1, 1]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test func nativeInt16VertexAndFragmentRunOnMetal() async throws {
    let source: ShaderInput = .wgsl("""
        enable wgpu_int16;

        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(flat) values: vec2u,
        }

        @vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            let signed = i16(index + 32767u) + i16(1) - i16(index);
            let unsigned = u16(index + 65535u) + u16(2) - u16(index);
            return VertexOut(vec4f(points[index], 0, 1), vec2u(bitcast<u32>(i32(signed)), u32(unsigned)));
        }

        @fragment fn fs(@location(0) @interpolate(flat) values: vec2u) -> @location(0) vec4u {
            let signed = i16(bitcast<i32>(values.x)) - i16(1);
            let unsigned = u16(values.y) * u16(3);
            let pair = vec2<u16>(values.yy) + vec2<u16>(u16(7), u16(11));
            return vec4u(select(0u, 255u, i32(signed) == 32767), u32(unsigned), vec2u(pair));
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    #expect(vertex.requiresShaderInt16 && fragment.requiresShaderInt16)
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderInt16])
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
        let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
            usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .render(pipeline: pipeline, target: color, colorLoad: .clear(.transparentBlack), vertexCount: 3),
            .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[0x0c0803ff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
