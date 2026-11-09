import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let unusedFragmentOutputShader = """
    @group(0) @binding(0) var<storage, read_write> calls: atomic<u32>;
    struct Colors {
        @location(0) visible: vec4u,
        @location(2) extra: vec4u,
    }
    fn extraColor() -> vec4u {
        return vec4u(atomicAdd(&calls, 1u), 11, 13, 255);
    }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(points[i], 0, 1);
    }
    @fragment fn fs() -> Colors { return Colors(vec4u(19, 37, 73, 255), extraColor()); }
    @fragment fn extraOnly() -> @location(2) vec4u { return extraColor(); }
    """

@Test func unusedFragmentOutputsAcceptSparseLayoutsAndRetainDeviceLocationLimits() async throws {
    let caps = try DeviceCapabilities(maxBufferSize: 1024, maxColorAttachments: 2)
    let host = RenderTestBackend(capabilities: caps)
    let device = try await Device.make(backend: host)
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(unusedFragmentOutputShader)
    let vertex = try compiler.compileWebGPURenderStage(input, entryPoint: "vs")
    let above = try compiler.compileWebGPURenderStage(input, entryPoint: "fs")
    do {
        await #expect(throws: GPUError.validation("Fragment output location exceeds the device attachment limit")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: above, colorFormat: .rgba8Uint)
        }
        #expect(await host.allocatedRenderPipelines == 0)
        let within = try compiler.compileWebGPURenderStage(.wgsl("""
            struct Colors { @location(0) visible: vec4u, @location(1) unused: vec4u, }
            @fragment fn fs() -> Colors { return Colors(vec4u(1), vec4u(2)); }
            """), entryPoint: "fs")
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: within, colorFormat: .rgba8Uint)
        #expect(pipeline.additionalColorOutputs.isEmpty)
        #expect(pipeline.writesColor)
        #expect(await host.allocatedRenderPipelines == 1)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

#if canImport(Metal)
@Test func unusedFragmentOutputsPreserveColorAndSideEffectsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.wgsl(unusedFragmentOutputShader)
    let vertex = try compiler.compileRenderStage(input, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let sparse = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
        let bound = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint,
                                                           additionalColorTargets: [2: .init(format: .rgba8Uint)])
        let unusedOnly = try compiler.compileRenderStage(input, entryPoint: "extraOnly", to: .msl)
        let noWrites = try await device.createRenderPipeline(vertex: vertex, fragment: unusedOnly, colorFormat: .rgba8Uint)
        #expect(!noWrites.writesColor)
        let counter = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let primary = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                            usage: [.renderAttachment, .copySource]))
        let extra = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                          usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let bindings = [ComputeBindingSlot(binding: 0): ComputeBufferBinding(buffer: counter)]
        let result = try await device.submitCommands([
            .render(pipeline: sparse, target: primary, clear: .transparentBlack, vertexCount: 3,
                    storageBuffers: bindings),
            .copyTextureToBuffer(source: primary, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copy(source: counter, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
            .render(pipeline: bound, target: primary, clear: .transparentBlack, vertexCount: 3,
                    additionalColors: [2: .init(texture: extra, loadAction: .clear(.transparentBlack))],
                    storageBuffers: bindings),
            .copyTextureToBuffer(source: extra, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copy(source: counter, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
            .render(pipeline: noWrites, target: primary, clear: .transparentBlack, vertexCount: 3,
                    storageBuffers: bindings),
            .copyTextureToBuffer(source: primary, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copy(source: counter, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff492513], [1], [0xff0d0b01], [2], [0], [3]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
