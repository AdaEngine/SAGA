#if os(Windows) || os(Linux) || os(Android)
import SAGACore
import SAGACompiler
import SAGATypes
@testable import SAGAVulkan
import Testing

@Test(.enabled(if: VulkanBackend.requestAdapter() != nil), .gpuExclusive)
func vulkanGroupsAdjacentColorDrawsAndEndsThePassBeforeACopy() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                return vec4f(points[index], 0, 1);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
            """
        let compiler = IndependentShaderCompiler()
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv),
            fragment: compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv),
            colorFormat: .rgba8Unorm
        )
        let target = try await device.createTexture(.init(width: 4, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let green = RenderClearColor(red: 0, green: 1, blue: 0, alpha: 1)
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: target, clear: blue, vertexCount: 0),
            .render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 3,
                    raster: .init(scissor: .init(x: 0, y: 0, width: 2, height: 1))),
            .render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 3,
                    raster: .init(scissor: .init(x: 2, y: 0, width: 1, height: 1))),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 16),
            .render(pipeline: pipeline, target: target, clear: green, vertexCount: 0),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 16),
        ]).wait()
        #expect(result == [[0xff0000ff, 0xff0000ff, 0xff0000ff, 0xffff0000], Array(repeating: 0xff00ff00, count: 4)])
        #if DEBUG
        #expect(await backend.lastRenderPassCount == 2)
        #expect(await backend.lastImageViewCount == 1)
        #expect(await backend.lastGraphicsPipelineBindCount == 2)
        #endif
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: VulkanBackend.requestAdapter() != nil), .gpuExclusive)
func vulkanGroupsMatchingGeometryPassesWithoutLosingDepthOrColor() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let shader = """
            fn position(index: u32, z: f32) -> vec4f {
                let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                return vec4f(points[index], z, 1);
            }
            @vertex fn near(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f { return position(index, 0.25); }
            @vertex fn far(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f { return position(index, 0.75); }
            struct Output { @location(0) main: vec4f, @location(1) extra: vec4f, }
            @fragment fn redBlue() -> Output { return Output(vec4f(1, 0, 0, 1), vec4f(0, 0, 1, 1)); }
            @fragment fn green() -> Output { return Output(vec4f(0, 1, 0, 1), vec4f(0, 1, 0, 1)); }
            """
        let compiler = IndependentShaderCompiler()
        func pipeline(_ vertex: String, _ fragment: String) async throws -> RenderPipeline {
            try await device.createRenderPipeline(
                vertex: compiler.compileRenderStage(.wgsl(shader), entryPoint: vertex, to: .spirv),
                fragment: compiler.compileRenderStage(.wgsl(shader), entryPoint: fragment, to: .spirv),
                colorFormat: .rgba8Unorm, depthState: .init(format: .depth32Float, compare: .less, writeEnabled: true),
                additionalColorTargets: [1: .init(format: .rgba8Unorm)])
        }
        let nearPipeline = try await pipeline("near", "redBlue")
        let farPipeline = try await pipeline("far", "green")
        let color = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource])
        let first = try await device.createTexture(color)
        let second = try await device.createTexture(color)
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: .renderAttachment))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: nearPipeline, target: first, clear: .transparentBlack, vertexCount: 3,
                    depth: .init(texture: depth), additionalColors: [1: .init(texture: second, loadAction: .clear(.transparentBlack))]),
            .render(pipeline: farPipeline, target: first, colorLoad: .load, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .load), additionalColors: [1: .init(texture: second)]),
            .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff0000ff], [0xffff0000]])
        #if DEBUG
        #expect(await backend.lastRenderPassCount == 1)
        #expect(await backend.lastGraphicsPipelineBindCount == 2)
        #endif
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
@Test(.enabled(if: VulkanBackend.requestAdapter() != nil), .gpuExclusive)
func vulkanRebindsChangedUniformsInsideSharedRenderPass() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @group(0) @binding(0) var<uniform> color: vec4f;
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                return vec4f(points[index], 0, 1);
            }
            @fragment fn fs() -> @location(0) vec4f { return color; }
            """
        let compiler = IndependentShaderCompiler()
        let pipeline = try await device.createRenderPipeline(
            vertex: compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv),
            fragment: compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv),
            colorFormat: .rgba8Unorm
        )
        let target = try await device.createTexture(.init(width: 3, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 12, usage: [.copyDestination, .mapRead]))
        let red = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let green = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let one = Float(1).bitPattern
        let result = try await device.submitCommands([
            .write(buffer: red, offset: 0, words: [one, 0, 0, one]),
            .write(buffer: green, offset: 0, words: [0, one, 0, one]),
            .render(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), vertexCount: 3,
                    uniforms: [.init(binding: 0): .init(buffer: red)], raster: .init(scissor: .init(x: 0, y: 0, width: 1, height: 1))),
            .render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 3,
                    uniforms: [.init(binding: 0): .init(buffer: red)], raster: .init(scissor: .init(x: 1, y: 0, width: 1, height: 1))),
            .render(pipeline: pipeline, target: target, colorLoad: .load, vertexCount: 3,
                    uniforms: [.init(binding: 0): .init(buffer: green)], raster: .init(scissor: .init(x: 2, y: 0, width: 1, height: 1))),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 12),
        ]).wait()
        #expect(result == [[0xff0000ff, 0xff0000ff, 0xff00ff00]])
        #if DEBUG
        #expect(await backend.lastGraphicsPipelineBindCount == 1)
        #endif
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
