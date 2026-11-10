#if os(Windows)
import Foundation
import WinSDK
#endif
import Testing
import SAGATypes
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGACompiler
@testable import SAGAVulkan

// These tests execute on a real Vulkan driver. Without one they are reported as skipped, never as passed.
private let vulkanAvailable = VulkanBackend.requestAdapter() != nil

private let program = """
    @group(0) @binding(1) var<storage, read> source: array<u32, 4>;
    @group(0) @binding(3) var<storage, read_write> destination: array<u32, 4>;

    @compute @workgroup_size(2, 2)
    fn run(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x + id.y * 2u;
        destination[i] = source[i] * 3u + 7u;
    }
    """

private let sourceSlot = ComputeBindingSlot(binding: 1)
private let destinationSlot = ComputeBindingSlot(binding: 3)

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanAdapterReportsUsableLimits() throws {
    let adapter = try #require(VulkanBackend.requestAdapter())
    #expect(!adapter.name.isEmpty)
    #expect(adapter.apiVersion.major > 1 || adapter.apiVersion.minor >= 1)
    let device = try VulkanBackend.makeDevice(adapter: adapter)
    let capabilities = device.capabilities
    #expect(capabilities.maxBufferSize >= 128 << 20)
    #expect(capabilities.maxComputeInvocationsPerWorkgroup >= 128)
    #expect(capabilities.maxComputeWorkgroupsPerDimension >= 65_535)
    #expect(capabilities.maxComputeWorkgroupSize.x >= 128)
    #expect(capabilities.features.isSubset(of: [.timestampQuery, .depth32FloatStencil8, .shaderF16, .shaderInt16, .shaderInt64, .subgroups, .multiview, .selectiveMultiview, .textureCompressionBC, .textureCompressionETC2, .textureCompressionASTC, .cooperativeMatrix, .meshShaders]))  // only probed, supported features
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanReadbacksAreZeroInitializedAndSnapshotOrdered() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let a = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .copySource]))
        let r = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .read(buffer: r, byteCount: 16),
            .write(buffer: a, offset: 4, words: [42, 99]),
            .copy(source: a, destination: r, byteCount: 16),
            .read(buffer: r, byteCount: 16),
            .clearBuffer(buffer: a, offset: 8, byteCount: 4),
            .copy(source: a, destination: r, byteCount: 16),
            .read(buffer: r, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[0, 0, 0, 0], [0, 42, 99, 0], [0, 42, 0, 0]])
        await device.close()
    } catch { await device.close(); throw error }
    #expect(await backend.resourceCountsForTesting().buffers == 0)
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRejectsUnsupportedCommandsAndQueueRemainsUsable() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let a = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let b = try await device.createBuffer(.init(size: 4, usage: [.storage, .copyDestination]))
        let invalid = try await device.submitCommands([
            .dispatch(pipeline: .init(device: device.identity, serial: 999), input: a, output: b, count: 1)
        ])
        await #expect(throws: GPUError.self) { try await invalid.wait() }
        let valid = try await device.submitCommands([.copy(source: a, destination: b, byteCount: 4)])
        #expect(try await valid.wait().isEmpty)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func wgslComputeRunsThroughSpirvDeviceQueueAndVulkanReadback() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    let compiler = IndependentShaderCompiler()
    do {
        let shader = try compiler.compileCompute(.wgsl(program), to: .spirv)
        let pipeline = try await device.createComputePipeline(shader)
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let bindings = [sourceSlot: input, destinationSlot: output]
        let ticket = try await device.submitCommands([
            .write(buffer: input, offset: 0, words: [0, 1, 2, .max]),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 0)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1, y: 2)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[0, 0, 0, 0], [7, 10, 13, 4]])

        let empty = try await device.createComputePipeline(
            try compiler.compileCompute(.wgsl("@compute @workgroup_size(1) fn run() {}"), to: .spirv))
        let noBuffers = try await device.submitCommands([
            .dispatchCompute(pipeline: empty, bindings: [:], workgroups: .init(x: 1))
        ])
        #expect(try await noBuffers.wait().isEmpty)

        let hlsl = try compiler.compileCompute(.wgsl(program), to: .hlsl)
        await #expect(throws: GPUError.self) { try await device.createComputePipeline(hlsl) }
        #expect(await backend.resourceCountsForTesting().pipelines == 2)
        await device.close()
    } catch { await device.close(); throw error }
    #expect(await backend.resourceCountsForTesting().pipelines == 0)
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRunsLargeDispatchAcrossManyWorkgroups() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
            @group(0) @binding(0) var<storage, read_write> data: array<u32>;
            @compute @workgroup_size(64)
            fn run(@builtin(global_invocation_id) id: vec3<u32>) {
                if (id.x < arrayLength(&data)) { data[id.x] = id.x * 2654435761u; }
            }
            """), to: .spirv)
        let pipeline = try await device.createComputePipeline(shader)
        let count = 4099
        let size = UInt64(count * 4)
        let storage = try await device.createBuffer(.init(size: size, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage], workgroups: .init(x: UInt32((count + 63) / 64))),
            .copy(source: storage, destination: readback, byteCount: size),
            .read(buffer: readback, byteCount: size),
        ])
        let result = try await ticket.wait()
        #expect(result == [(0..<count).map { UInt32(truncatingIfNeeded: $0) &* 2654435761 }])
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanBindsUniformBuffersAndImmediateData() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let uniformShader = try compiler.compileCompute(.wgsl("""
            struct Params { values: vec4u, }
            @group(0) @binding(0) var<uniform> params: Params;
            @group(1) @binding(2) var<storage, read_write> output: array<u32, 2>;
            @compute @workgroup_size(1) fn run() { output[0] = params.values.x + params.values.w; output[1] = params.values.y * params.values.z; }
            """), to: .spirv)
        let uniformPipeline = try await device.createComputePipeline(uniformShader)
        let params = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 8, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let first = try await device.submitCommands([
            .write(buffer: params, offset: 0, words: [1, 5, 6, 40]),
            .dispatchCompute(pipeline: uniformPipeline,
                             bindings: [.init(group: 0, binding: 0): params, .init(group: 1, binding: 2): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 8),
            .read(buffer: readback, byteCount: 8),
        ])
        #expect(try await first.wait() == [[41, 30]])

        let immediateShader = try compiler.compileCompute(.wgsl("""
            requires immediate_address_space;
            struct Pair { first: u32, second: u32, }
            struct Settings { axis: vec3u, pair: Pair, last: u32, }
            var<immediate> settings: Settings;
            @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
            @compute @workgroup_size(1) fn run() {
                output[0] = settings.axis.z; output[1] = settings.pair.first;
                output[2] = settings.pair.second; output[3] = settings.last;
            }
            """), to: .spirv)
        let immediatePipeline = try await device.createComputePipeline(immediateShader)
        let wide = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let wideReadback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let second = try await device.submitCommands([
            .dispatchCompute(pipeline: immediatePipeline, bindings: [.init(binding: 0): wide], workgroups: .init(x: 1),
                             immediates: [1, 2, 3, 10, 20, 30, 0, 0]),
            .copy(source: wide, destination: wideReadback, byteCount: 16),
            .read(buffer: wideReadback, byteCount: 16),
        ])
        #expect(try await second.wait() == [[3, 10, 20, 30]])
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanIndirectDispatchValidatesGPUAuthoredCounts() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
            @group(0) @binding(0) var<storage, read_write> data: array<u32, 8>;
            @compute @workgroup_size(1)
            fn run(@builtin(workgroup_id) id: vec3<u32>) { data[id.x] = id.x + 1u; }
            """), to: .spirv)
        let pipeline = try await device.createComputePipeline(shader)
        let arguments = try await device.createBuffer(.init(size: 32, usage: [.indirect, .copyDestination]))
        let data = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let limit = backend.capabilities.maxComputeWorkgroupsPerDimension
        var cases: [(counts: [UInt32], expected: [UInt32])] = [
            ([4, 1, 1], [1, 2, 3, 4, 0, 0, 0, 0]),
            ([0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]),
            ([3, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0]),
        ]
        if limit < UInt32.max { cases.append(([limit + 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0])) }
        for (counts, expected) in cases {
            let ticket = try await device.submitCommands([
                .clearBuffer(buffer: data),
                .write(buffer: arguments, offset: 0, words: counts),
                .dispatchComputeIndirect(pipeline: pipeline, bindings: [.init(binding: 0): .init(buffer: data)],
                                         indirectBuffer: arguments),
                .copy(source: data, destination: readback, byteCount: 32),
                .read(buffer: readback, byteCount: 32),
            ])
            #expect(try await ticket.wait() == [expected], "counts \(counts)")
        }
        await device.close()
    } catch { await device.close(); throw error }
}

private func pixel(_ words: [UInt32], x: Int, y: Int) -> UInt32 { words[y * 64 + x] }

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRendersTriangleAndReadsBackPixels() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, 1.0f, 0.25f, 1.0f); }
                if index == 1u { return vec4f(1.0f, 1.0f, 0.25f, 1.0f); }
                return vec4f(-1.0f, -1.0f, 0.25f, 1.0f);
            }
            @fragment fn fragmentMain() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vertexMain", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fragmentMain", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(pixel(snapshots[0], x: 0, y: 0) == 0)  // new textures are zeroed
        #expect(pixel(snapshots[1], x: 0, y: 0) == 0xff00ff00)  // inside the triangle: green
        #expect(pixel(snapshots[1], x: 7, y: 7) == 0xffff0000)  // outside: cleared blue
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRendersVertexBuffersInstancesAndUniforms() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            struct Params { color: vec4f, }
            @group(0) @binding(0) var<uniform> params: Params;
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return params.color; }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: [layout])
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource, .copyDestination]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let params = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [(-1.0 as Float).bitPattern, (-1.0 as Float).bitPattern,
                                                       (3.0 as Float).bitPattern, (-1.0 as Float).bitPattern,
                                                       (-1.0 as Float).bitPattern, (3.0 as Float).bitPattern]),
            .write(buffer: params, offset: 0, words: [(1.0 as Float).bitPattern, 0, 0, (1.0 as Float).bitPattern]),
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)], uniforms: [.init(binding: 0): .init(buffer: params)]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .writeTexture(destination: target, data: [UInt8](repeating: 0x7f, count: 8 * 8 * 4),
                          layout: .init(bytesPerRow: 32, rowsPerImage: 8), region: .init(width: 8, height: 8)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        for (x, y) in [(0, 0), (7, 0), (0, 7), (7, 7), (3, 4)] {
            #expect(pixel(snapshots[0], x: x, y: y) == 0xff0000ff, "pixel \(x),\(y)")
            #expect(pixel(snapshots[1], x: x, y: y) == 0x7f7f7f7f, "uploaded pixel \(x),\(y)")
        }
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanSamplesTexturesAndWritesStorageTexturesInCompute() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
            @group(0) @binding(0) var source: texture_2d<f32>;
            @group(0) @binding(1) var filtering: sampler;
            @group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
            @compute @workgroup_size(4, 4)
            fn run(@builtin(global_invocation_id) id: vec3<u32>) {
                let c = textureSampleLevel(source, filtering, (vec2f(id.xy) + 0.5) / 4.0, 0.0);
                textureStore(output, id.xy, vec4f(c.b, c.g, c.r, c.a));
            }
            """), to: .spirv)
        let pipeline = try await device.createComputePipeline(shader)
        let source = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                          usage: [.textureBinding, .copyDestination]))
        let output = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                          usage: [.storageBinding, .copySource]))
        let sampler = try await device.createSampler()
        let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        var bytes: [UInt8] = []
        for y in 0..<4 { for x in 0..<4 { bytes += [UInt8(x * 10), UInt8(y * 10), 5, 255] } }
        let ticket = try await device.submitCommands([
            .writeTexture(destination: source, data: bytes, layout: .init(bytesPerRow: 16, rowsPerImage: 4),
                          region: .init(width: 4, height: 4)),
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [.init(binding: 0): source, .init(binding: 2): output],
                             samplers: [.init(binding: 1): sampler]),
            .copyTextureToBuffer(source: output, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 784),
        ])
        let words = try await ticket.wait()[0]
        for y in 0..<4 {
            for x in 0..<4 {
                #expect(words[y * 64 + x] == UInt32(5) | UInt32(y * 10) << 8 | UInt32(x * 10) << 16 | 0xff00_0000,
                        "texel \(x),\(y)")
            }
        }
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanSamplesTexturesInFragmentShaders() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var nearest: sampler;
            @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
                var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                return vec4f(p[i], 0.5, 1.0);
            }
            @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
                return textureSampleLevel(image, nearest, pos.xy / 8.0, 0.0);
            }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let image = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                         usage: [.textureBinding, .copyDestination]))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let sampler = try await device.createSampler()
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let colors: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]
        let ticket = try await device.submitCommands([
            .writeTexture(destination: image, data: colors, layout: .init(bytesPerRow: 8, rowsPerImage: 2),
                          region: .init(width: 2, height: 2)),
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): image], samplers: [.init(binding: 1): sampler]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let words = try await ticket.wait()[0]
        #expect(words[1 * 64 + 1] == 0xff00_00ff)   // top-left quadrant: red
        #expect(words[1 * 64 + 6] == 0xff00_ff00)   // top-right: green
        #expect(words[6 * 64 + 1] == 0xffff_0000)   // bottom-left: blue
        #expect(words[6 * 64 + 6] == 0xffff_ffff)   // bottom-right: white
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanIndexedDrawsUseIndicesAndSignedBaseVertex() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: [layout])
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 32, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 16, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }
        let red = RenderClearColor(red: 1, green: 0, blue: 0, alpha: 1)
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: bits([0, 0, -1, -1, 3, -1, -1, 3])),
            .write(buffer: indices, offset: 0, words: [0x0002_0001, 0x0000_0003, 0x0003_0002, 0x0000_0004]),
            // uint16 indices [1, 2, 3, 0]: covers the target.
            .renderIndexed(pipeline: pipeline, target: target, clear: red, indexBuffer: .init(buffer: indices, format: .uint16),
                           draw: .init(indexCount: 3), vertexBuffers: [.init(buffer: vertices)]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            // Indices [2, 3, 4] with base vertex -1 select vertices 1, 2, 3.
            .renderIndexed(pipeline: pipeline, target: target, clear: red, indexBuffer: .init(buffer: indices, format: .uint16, offset: 8),
                           draw: .init(indexCount: 3, baseVertex: -1), vertexBuffers: [.init(buffer: vertices)]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            // Zero index count draws nothing.
            .renderIndexed(pipeline: pipeline, target: target, clear: red, indexBuffer: .init(buffer: indices, format: .uint16),
                           draw: .init(indexCount: 0), vertexBuffers: [.init(buffer: vertices)]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 3)
        #expect(pixel(snapshots[0], x: 3, y: 4) == 0xff00ff00)
        #expect(pixel(snapshots[1], x: 3, y: 4) == 0xff00ff00)
        #expect(pixel(snapshots[2], x: 3, y: 4) == 0xff0000ff)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanDepthTestingKeepsNearestFragmentsAndPersistsDepth() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            fn corner(index: u32, z: f32) -> vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, z, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, z, 1.0f); }
                return vec4f(-1.0f, 3.0f, z, 1.0f);
            }
            @vertex fn near(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f { return corner(index, 0.25f); }
            @vertex fn far(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f { return corner(index, 0.75f); }
            @fragment fn green() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            @fragment fn red() -> @location(0) vec4f { return vec4f(1.0f, 0.0f, 0.0f, 1.0f); }
            """
        func stage(_ entry: String) throws -> RenderStageArtifact {
            try compiler.compileRenderStage(.wgsl(source), entryPoint: entry, to: .spirv)
        }
        let less = RenderDepthState(format: .depth32Float, compare: .less)
        let greater = RenderDepthState(format: .depth32Float, compare: .greater)
        let nearPipeline = try await device.createRenderPipeline(
            vertex: stage("near"), fragment: stage("green"), colorFormat: .rgba8Unorm, depthState: less)
        let farPipeline = try await device.createRenderPipeline(
            vertex: stage("far"), fragment: stage("red"), colorFormat: .rgba8Unorm, depthState: less)
        let farGreater = try await device.createRenderPipeline(
            vertex: stage("far"), fragment: stage("red"), colorFormat: .rgba8Unorm, depthState: greater)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 8, height: 8, format: .depth32Float,
                                                         usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func draw(_ pipeline: RenderPipeline, clear: Bool) -> BufferCommand {
            .render(pipeline: pipeline, target: target, colorLoad: clear ? .clear(blue) : .load, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: clear ? .clear(1) : .load))
        }
        func snapshot() -> [BufferCommand] {
            [.copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 1824),
             .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 1824)]
        }
        let ticket = try await device.submitCommands(
            [draw(nearPipeline, clear: true)] + snapshot()
            + [draw(farPipeline, clear: false)] + snapshot()      // farther: rejected
            + [draw(farGreater, clear: false)] + snapshot())      // `greater` passes against 0.25
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 6)
        #expect(pixel(snapshots[0], x: 3, y: 3) == 0xff00ff00)
        #expect(pixel(snapshots[1], x: 3, y: 3) == Float(0.25).bitPattern)
        #expect(pixel(snapshots[2], x: 3, y: 3) == 0xff00ff00)  // the far draw lost the depth test
        #expect(pixel(snapshots[3], x: 3, y: 3) == Float(0.25).bitPattern)  // and did not write depth
        #expect(pixel(snapshots[4], x: 3, y: 3) == 0xff0000ff)  // greater compare wins
        #expect(pixel(snapshots[5], x: 3, y: 3) == Float(0.75).bitPattern)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanResolvesMultisampledColorTargets() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(1.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 1.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, sampleCount: 4)
        let multisampled = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                                usage: [.renderAttachment], sampleCount: 4))
        let resolved = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                            usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: multisampled, colorLoad: .clear(blue), vertexCount: 3,
                    resolve: .init(texture: resolved), colorStore: .discard),
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 1)
        let pixels = snapshots[0]
        #expect(pixel(pixels, x: 0, y: 0) == 0xff00ff00 || pixel(pixels, x: 0, y: 7) == 0xff00ff00)  // inside: green
        let mixed = (0..<8).flatMap { y in (0..<8).map { pixel(pixels, x: $0, y: y) } }
            .filter { $0 != 0xff00ff00 && $0 != 0xffff0000 }
        #expect(!mixed.isEmpty)  // diagonal edge pixels are partially covered, so resolve blends them
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanIndirectDrawsUseGPUAuthoredArguments() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, vertexBuffers: [layout])
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let arguments = try await device.createBuffer(.init(size: 64, usage: [.indirect, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }
        func draw(offset: UInt64, indexed: Bool) -> BufferCommand {
            .renderIndirect(pipeline: pipeline, target: target, colorLoad: .clear(blue), indirectBuffer: arguments,
                            indirectOffset: offset,
                            indexBuffer: indexed ? .init(buffer: indices, format: .uint16) : nil,
                            vertexBuffers: [.init(buffer: vertices)])
        }
        let snapshot: [BufferCommand] = [
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 1824)]
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: bits([-1, -1, 3, -1, -1, 3])),
            .write(buffer: indices, offset: 0, words: [0x0001_0000, 0x0000_0002]),
            // [vertexCount, instanceCount, firstVertex, firstInstance] at 0; [3, 1, 0, 0, 0] indexed at 16;
            // a nonzero firstInstance at 40 must be neutralized without the optional feature.
            .write(buffer: arguments, offset: 0, words: [3, 1, 0, 0, 3, 1, 0, 0, 0, 0, 3, 1, 0, 0, 1]),
            draw(offset: 0, indexed: false),
        ] + snapshot + [draw(offset: 16, indexed: true)] + snapshot + [draw(offset: 40, indexed: true)] + snapshot)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 3)
        #expect(pixel(snapshots[0], x: 3, y: 4) == 0xff00ff00)
        #expect(pixel(snapshots[1], x: 3, y: 4) == 0xff00ff00)
        #expect(pixel(snapshots[2], x: 3, y: 4) == 0xffff0000)  // firstInstance != 0: draw skipped, target cleared blue
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanStencilTestsUseReferenceAndPersist() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn green() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            @fragment fn red() -> @location(0) vec4f { return vec4f(1.0f, 0.0f, 0.0f, 1.0f); }
            """
        func stage(_ entry: String) throws -> RenderStageArtifact {
            try compiler.compileRenderStage(.wgsl(source), entryPoint: entry, to: .spirv)
        }
        let replace = RenderStencilFaceState(compare: .always, pass: .replace)
        let equal = RenderStencilFaceState(compare: .equal)
        func state(_ face: RenderStencilFaceState) -> RenderDepthState {
            RenderDepthState(format: .depth24PlusStencil8, compare: .always, writeEnabled: false,
                             stencil: RenderStencilState(front: face, back: face))
        }
        let writer = try await device.createRenderPipeline(vertex: stage("vs"), fragment: stage("green"),
                                                           colorFormat: .rgba8Unorm, depthState: state(replace))
        let tester = try await device.createRenderPipeline(vertex: stage("vs"), fragment: stage("red"),
                                                           colorFormat: .rgba8Unorm, depthState: state(equal))
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 8, height: 8, format: .depth24PlusStencil8,
                                                         usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func draw(_ pipeline: RenderPipeline, reference: UInt32, first: Bool) -> BufferCommand {
            .render(pipeline: pipeline, target: target, colorLoad: first ? .clear(blue) : .load, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: first ? .clear(1) : .load,
                                 stencil: .init(loadAction: first ? .clear(0) : .load, reference: reference)))
        }
        let ticket = try await device.submitCommands([
            draw(writer, reference: 7, first: true),
            draw(tester, reference: 3, first: false),  // stencil is 7, so the test fails
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            draw(tester, reference: 7, first: false),  // matches
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256, aspect: .stencilOnly),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 3)
        #expect(pixel(snapshots[0], x: 3, y: 3) == 0xff00ff00)
        #expect(pixel(snapshots[1], x: 3, y: 3) == 0xff0000ff)
        #expect(snapshots[2][0] == 0x0707_0707)  // the stencil aspect holds the replaced reference
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanTextureViewsSelectMipLevelsForSamplingAndRendering() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let mips = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let level1 = try await device.createTextureView(texture: mips, .init(baseMipLevel: 1, mipLevelCount: 1))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            // Rendering through the view covers only mip 1 (4x4), leaving mip 0 untouched.
            .render(pipeline: pipeline, target: level1, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: mips, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 780),
            .copyTextureToBuffer(source: mips, destination: read, bytesPerRow: 256, mipLevel: 0),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(pixel(snapshots[0], x: 1, y: 1) == 0xff00ff00)
        #expect(pixel(snapshots[1], x: 1, y: 1) == 0)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanSupportsArrayLayerViewsAndThreeDimensionalCopies() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let layers = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource], arrayLayerCount: 2))
        let layer1 = try await device.createTextureView(texture: layers, .init(dimension: .d2, baseArrayLayer: 1, arrayLayerCount: 1))
        let volume = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                          usage: [.copySource, .copyDestination], dimension: .d3, depth: 2))
        let read = try await device.createBuffer(.init(size: 1024, usage: [.copyDestination, .mapRead]))
        let slices = [UInt8](repeating: 0x11, count: 64) + [UInt8](repeating: 0x22, count: 64)
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: layer1, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: layers, destination: read, bytesPerRow: 256, arrayLayer: 0),
            .read(buffer: read, byteCount: 784),
            .copyTextureToBuffer(source: layers, destination: read, bytesPerRow: 256, arrayLayer: 1),
            .read(buffer: read, byteCount: 784),
            .writeTexture(destination: volume, data: slices, layout: .init(bytesPerRow: 16, rowsPerImage: 4),
                          region: .init(width: 4, height: 4, depth: 2)),
            .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, z: 1),
            .read(buffer: read, byteCount: 784),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 3)
        #expect(pixel(snapshots[0], x: 1, y: 1) == 0)  // layer 0 was not rendered
        #expect(pixel(snapshots[1], x: 1, y: 1) == 0xff00ff00)
        #expect(pixel(snapshots[2], x: 1, y: 1) == 0x2222_2222)  // depth slice 1
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanMapsBuffersForHostWritesAndReads() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let upload = try await device.createBuffer(.init(size: 16, usage: [.mapWrite, .copySource]))
        let download = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
        let writing = try await device.mapBuffer(upload, mode: .write)
        try await device.writeMappedBytes(writing, offset: 0, bytes: Array(1...16))
        try await device.unmapBuffer(upload)
        _ = try await device.submitCommands([.copy(source: upload, destination: download, byteCount: 16)])
        let reading = try await device.mapBuffer(download, mode: .read)
        let bytes = try await device.readMappedBytes(reading)
        try await device.unmapBuffer(download)
        #expect(bytes == Array(1...16))
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanOcclusionAndTimestampQueriesResolveToBuffers() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment]))
        let occlusion = try await device.createQuerySet(.init(type: .occlusion, count: 2))
        let resolved = try await device.createBuffer(.init(size: 512, usage: [.queryResolve, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func pass(vertexCount: UInt32, query: UInt32) -> BufferCommand {
            .occlusionQueried(
                pass: .render(pipeline: pipeline, target: target, clear: blue, vertexCount: vertexCount),
                query: .init(querySet: occlusion, queryIndex: query))
        }
        let ticket = try await device.submitCommands([
            pass(vertexCount: 3, query: 0),
            pass(vertexCount: 0, query: 1),
            .resolveQuerySet(querySet: occlusion, firstQuery: 0, queryCount: 2, destination: resolved),
            .copy(source: resolved, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 1)
        #expect(snapshots[0][0] != 0 || snapshots[0][1] != 0)  // visible samples
        #expect(snapshots[0][2] == 0 && snapshots[0][3] == 0)  // nothing drawn
        await device.close()
    } catch { await device.close(); throw error }
}

#if os(Windows)
/// A window owned by its own thread, so that resizing happens on the thread that created it.
private final class TestWindow: @unchecked Sendable {
    private(set) var handle: HWND?
    private let ready = DispatchSemaphore(value: 0)
    private let wake = DispatchSemaphore(value: 0)
    private let done = DispatchSemaphore(value: 0)
    private var work: (() -> Void)?
    private var quitting = false

    init() {
        let thread = Thread { [self] in
            let module = GetModuleHandleW(nil)
            handle = "STATIC".withCString(encodedAs: UTF16.self) { windowClass in
                "SAGA".withCString(encodedAs: UTF16.self) { title in
                    CreateWindowExW(0, windowClass, title, DWORD(WS_OVERLAPPEDWINDOW), 0, 0, 200, 150, nil, nil, module, nil)
                }
            }
            ready.signal()
            while true {
                wake.wait()
                if quitting { break }
                work?()
                done.signal()
            }
            if let handle { DestroyWindow(handle) }
            done.signal()
        }
        thread.start()
        ready.wait()
    }

    func perform(_ body: @escaping () -> Void) {
        work = body
        wake.signal()
        done.wait()
    }

    func close() {
        quitting = true
        wake.signal()
        done.wait()
    }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanPresentsToAWin32WindowAndRebuildsItsSwapchain() async throws {
    let window = TestWindow()
    defer { window.close() }
    try #require(window.handle != nil)
    let target = VulkanSurfaceTarget.win32(hinstance: UInt(bitPattern: GetModuleHandleW(nil)),
                                           hwnd: UInt(bitPattern: window.handle))
    let backend: VulkanDevice
    do { backend = try VulkanBackend.makeDevice(surface: target) } catch GPUError.unsupported { return }  // no presentation support
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .bgra8Unorm)
        let frame = try await device.createTexture(.init(width: 64, height: 64, format: .bgra8Unorm,
                                                         usage: [.renderAttachment, .copySource]))
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: frame, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3)])
        try await device.present(frame).wait()
        // Resizing forces the swapchain to be rebuilt on the next present.
        let handle = window.handle
        window.perform { SetWindowPos(handle, nil, 0, 0, 320, 240, UINT(SWP_NOMOVE | SWP_NOZORDER)) }
        try await device.present(frame).wait()
        try await device.present(frame).wait()
        #expect(await backend.presentationRebuildsForTesting() == 2)
        await device.close()
    } catch { await device.close(); throw error }
}
#endif

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRunsHalfAndSixteenBitIntegerShaders() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let features = backend.capabilities.features
        let compiler = IndependentShaderCompiler()
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        func run(_ body: String, enable: String) async throws -> [UInt32] {
            let source = """
                \(enable)
                @group(0) @binding(3) var<storage, read_write> destination: array<u32, 4>;
                @compute @workgroup_size(4) fn run(@builtin(global_invocation_id) id: vec3u) { \(body) }
                """
            let pipeline = try await device.createComputePipeline(try compiler.compileCompute(.wgsl(source), to: .spirv))
            let ticket = try await device.submitCommands([
                .dispatchCompute(pipeline: pipeline, bindings: [destinationSlot: output], workgroups: .init(x: 1)),
                .copy(source: output, destination: readback, byteCount: 16),
                .read(buffer: readback, byteCount: 16),
            ])
            return try await ticket.wait()[0]
        }
        if features.contains(.shaderF16) {
            // f16(id) * 1.5 converted to f32 bits: 0, 1.5, 3.0, 4.5.
            let words = try await run("let h = f16(id.x) * f16(1.5); destination[id.x] = bitcast<u32>(f32(h));", enable: "enable f16;")
            #expect(words == [0, 1.5, 3.0, 4.5].map { Float($0).bitPattern })
        }
        // Without 16-bit support the shader is rejected before it reaches the driver.
        else {
            await #expect(throws: GPUError.self) {
                _ = try await run("destination[id.x] = 0u;", enable: "enable f16;")
            }
        }
    } catch { await device.close(); throw error }
    await device.close()
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRunsSubgroupOperationsWhenAdvertised() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            enable subgroups;
            @group(0) @binding(3) var<storage, read_write> destination: array<u32, 4>;
            @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) i: u32) { destination[i] = subgroupAdd(i + 1u); }
            """
        let shader = try compiler.compileCompute(.wgsl(source), to: .spirv)
        guard backend.capabilities.features.contains(.subgroups) else {
            await #expect(throws: GPUError.self) { _ = try await device.createComputePipeline(shader) }
            await device.close()
            return
        }
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [destinationSlot: output], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        // The four invocations share a subgroup, so each sees 1 + 2 + 3 + 4.
        #expect(try await ticket.wait() == [[10, 10, 10, 10]])
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanStoresBlockCompressedTextures() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.textureCompressionBC) else {
            await #expect(throws: GPUError.self) {
                _ = try await device.createTexture(.init(width: 8, height: 8, format: .bc1RGBAUnorm, usage: [.textureBinding, .copyDestination]))
            }
            await device.close()
            return
        }
        let texture = try await device.createTexture(.init(width: 8, height: 8, format: .bc1RGBAUnorm,
                                                           usage: [.textureBinding, .copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 272, usage: [.copyDestination, .mapRead]))
        // Four BC1 blocks; the first two bytes of each are color0.
        let blocks: [UInt8] = (0..<4).flatMap { block -> [UInt8] in [UInt8(0x10 + block), 0xF8, 0, 0, 0, 0, 0, 0] }
        let ticket = try await device.submitCommands([
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 272),
            .writeTexture(destination: texture, data: blocks, layout: .init(bytesPerRow: 16, rowsPerImage: 2),
                          region: .init(width: 8, height: 8)),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 272),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(snapshots[0][0] == 0)  // compressed textures start zeroed
        #expect(snapshots[1][0] & 0xffff == 0xF810)  // block (0, 0)
        #expect(snapshots[1][2] & 0xffff == 0xF811)  // block (1, 0)
        #expect(snapshots[1][64] & 0xffff == 0xF812)  // block (0, 1) starts the second block row
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanDepthTestsMultisampledTargetsAndStores1DTextures() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            fn corner(index: u32, z: f32) -> vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, z, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, z, 1.0f); }
                return vec4f(-1.0f, 3.0f, z, 1.0f);
            }
            @vertex fn near(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f { return corner(index, 0.25f); }
            @vertex fn far(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f { return corner(index, 0.75f); }
            @fragment fn green() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            @fragment fn red() -> @location(0) vec4f { return vec4f(1.0f, 0.0f, 0.0f, 1.0f); }
            """
        func stage(_ entry: String) throws -> RenderStageArtifact {
            try compiler.compileRenderStage(.wgsl(source), entryPoint: entry, to: .spirv)
        }
        let less = RenderDepthState(format: .depth32Float, compare: .less)
        let nearPipeline = try await device.createRenderPipeline(
            vertex: stage("near"), fragment: stage("green"), colorFormat: .rgba8Unorm, depthState: less, sampleCount: 4)
        let farPipeline = try await device.createRenderPipeline(
            vertex: stage("far"), fragment: stage("red"), colorFormat: .rgba8Unorm, depthState: less, sampleCount: 4)
        let color = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                         usage: [.renderAttachment], sampleCount: 4))
        let depth = try await device.createTexture(.init(width: 8, height: 8, format: .depth32Float,
                                                         usage: [.renderAttachment], sampleCount: 4))
        let resolved = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                            usage: [.renderAttachment, .copySource]))
        let line = try await device.createTexture(.init(width: 8, height: 1, format: .rgba8Unorm,
                                                        usage: [.copySource, .copyDestination], dimension: .d1))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func draw(_ pipeline: RenderPipeline, first: Bool) -> BufferCommand {
            .render(pipeline: pipeline, target: color, colorLoad: first ? .clear(blue) : .load, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: first ? .clear(1) : .load),
                    resolve: .init(texture: resolved))
        }
        let pixels: [UInt8] = (0..<32).map { UInt8($0) }
        let ticket = try await device.submitCommands([
            draw(nearPipeline, first: true),
            draw(farPipeline, first: false),  // behind the near triangle in every sample
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .writeTexture(destination: line, data: pixels, region: .init(width: 8, height: 1)),
            .copyTextureToBuffer(source: line, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 32),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(pixel(snapshots[0], x: 3, y: 3) == 0xff00ff00)
        let expected: [UInt32] = (0..<8).map { (index: UInt32) -> UInt32 in
            let base = index * 4
            return base | ((base + 1) << 8) | ((base + 2) << 16) | ((base + 3) << 24)
        }
        #expect(snapshots[1] == expected)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanBindsStorageBufferArrays() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            enable wgpu_binding_array;
            @group(0) @binding(0) var<storage, read_write> destination: array<u32, 4>;
            struct Data { values: array<u32, 4> }
            @group(0) @binding(1) var<storage, read> sources: binding_array<Data, 2>;
            @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) i: u32) {
                destination[i] = sources[0].values[i] * 10u + sources[1].values[i];
            }
            """
        let shader = try compiler.compileCompute(.wgsl(source), to: .spirv)
        let pipeline = try await device.createComputePipeline(shader)
        let first = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let second = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: first, offset: 0, words: [1, 2, 3, 4]),
            .write(buffer: second, offset: 0, words: [5, 6, 7, 8]),
            .dispatchCompute(pipeline: pipeline, bindings: [
                ComputeBindingSlot(binding: 0): output,
                ComputeBindingSlot(binding: 1, arrayElement: 0): first,
                ComputeBindingSlot(binding: 1, arrayElement: 1): second,
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[15, 26, 37, 48]])
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRendersMultipleViewsInOnePass() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.multiview) else { await device.close(); return }
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs(@builtin(view_index) view: u32) -> @location(0) vec4f {
                return vec4f(f32(view), 1.0f - f32(view), 0.0f, 1.0f);
            }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, multiviewMask: 0b11)
        let layers = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource], arrayLayerCount: 2))
        let both = try await device.createTextureView(texture: layers, .init(dimension: .d2Array, baseArrayLayer: 0, arrayLayerCount: 2))
        let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: both, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: layers, destination: read, bytesPerRow: 256, arrayLayer: 0),
            .read(buffer: read, byteCount: 784),
            .copyTextureToBuffer(source: layers, destination: read, bytesPerRow: 256, arrayLayer: 1),
            .read(buffer: read, byteCount: 784),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(pixel(snapshots[0], x: 1, y: 1) == 0xff00ff00)  // view 0: green
        #expect(pixel(snapshots[1], x: 1, y: 1) == 0xff0000ff)  // view 1: red
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanRendersIntoOneDepthSliceOfAThreeDimensionalTexture() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let volume = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource], dimension: .d3, depth: 2))
        let read = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: volume, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3,
                    targetDepthSlice: 1),
            .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, z: 0),
            .read(buffer: read, byteCount: 784),
            .copyTextureToBuffer(source: volume, destination: read, bytesPerRow: 256, z: 1),
            .read(buffer: read, byteCount: 784),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(pixel(snapshots[0], x: 1, y: 1) == 0)  // slice 0 untouched
        #expect(pixel(snapshots[1], x: 1, y: 1) == 0xff00ff00)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanKeepsMultisampledContentBetweenPassesWithoutResolve() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, sampleCount: 4)
        let multisampled = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                                usage: [.renderAttachment], sampleCount: 4))
        let resolved = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                            usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let ticket = try await device.submitCommands([
            // First pass keeps the samples and resolves nothing.
            .render(pipeline: pipeline, target: multisampled, clear: blue, vertexCount: 3),
            // Second pass draws nothing, loads the samples and resolves them.
            .render(pipeline: pipeline, target: multisampled, colorLoad: .load, vertexCount: 0, resolve: .init(texture: resolved)),
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 1)
        #expect(pixel(snapshots[0], x: 3, y: 3) == 0xff00ff00)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanMultipliesCooperativeMatrices() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        let shape = CooperativeMatrixProperties(mSize: 16, nSize: 16, kSize: 16, inputType: .float16, accumulatorType: .float32)
        guard backend.capabilities.features.contains(.cooperativeMatrix),
              backend.capabilities.cooperativeMatrixProperties.contains(shape) else {
            await device.close()
            return
        }
        let source = """
            enable f16;
            enable wgpu_cooperative_matrix;
            @group(0) @binding(0) var<storage, read> factors: array<f16, 512>;
            @group(0) @binding(1) var<storage, read_write> accumulators: array<f32, 512>;

            @compute @workgroup_size(32)
            fn main() {
                let a = coopLoad<coop_mat16x16<f16, A>>(&factors[0]);
                let b = coopLoad<coop_mat16x16<f16, B>>(&factors[256]);
                let c = coopLoad<coop_mat16x16<f32, C>>(&accumulators[0]);
                coopStore(coopMultiplyAdd(a, b, c), &accumulators[256]);
            }
            """
        let pipeline = try await device.createComputePipeline(
            try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .spirv))
        let factors = try await device.createBuffer(.init(size: 1024, usage: [.storage, .copyDestination]))
        let accumulators = try await device.createBuffer(.init(size: 2048, usage: [.storage, .copyDestination, .copySource]))
        let readback = try await device.createBuffer(.init(size: 1024, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            // First matrix of 1.0 (0x3C00), second of 2.0 (0x4000), packed two half floats per word.
            .write(buffer: factors, offset: 0, words: [UInt32](repeating: 0x3C00_3C00, count: 128) + [UInt32](repeating: 0x4000_4000, count: 128)),
            .dispatchCompute(pipeline: pipeline, bindings: [ComputeBindingSlot(binding: 0): factors,
                                                             ComputeBindingSlot(binding: 1): accumulators],
                             workgroups: .init(x: 1)),
            .copy(source: accumulators, sourceOffset: 1024, destination: readback, byteCount: 1024),
            .read(buffer: readback, byteCount: 1024),
        ])
        let words = try await ticket.wait()[0]
        // Every element is the sum of 16 products of 1.0 and 2.0.
        #expect(words.allSatisfy { Float(bitPattern: $0) == 32 })
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanDrawsWithMeshShaders() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.meshShaders) else { await device.close(); return }
        let source = """
            enable wgpu_mesh_shader;
            struct V { @builtin(position) position: vec4f, @location(0) color: vec4f, }
            struct P { @builtin(triangle_indices) indices: vec3u, }
            struct O {
              @builtin(vertices) vertices: array<V, 3>,
              @builtin(primitives) primitives: array<P, 1>,
              @builtin(vertex_count) vertex_count: u32,
              @builtin(primitive_count) primitive_count: u32,
            }
            var<workgroup> out: O;
            @mesh(out) @workgroup_size(1) fn mesh() {
              out.vertex_count = 3u; out.primitive_count = 1u;
              out.vertices[0].position = vec4f(-1.0, -1.0, 0.5, 1.0);
              out.vertices[1].position = vec4f(3.0, -1.0, 0.5, 1.0);
              out.vertices[2].position = vec4f(-1.0, 3.0, 0.5, 1.0);
              out.vertices[0].color = vec4f(0.0, 1.0, 0.0, 1.0);
              out.vertices[1].color = vec4f(0.0, 1.0, 0.0, 1.0);
              out.vertices[2].color = vec4f(0.0, 1.0, 0.0, 1.0);
              out.primitives[0].indices = vec3u(0u, 1u, 2u);
            }
            @fragment fn fragment(@location(0) color: vec4f) -> @location(0) vec4f { return color; }
            """
        let compiler = IndependentShaderCompiler()
        let mesh = try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fragment", to: .spirv)
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.init(red: 0, green: 0, blue: 1, alpha: 1)),
                      workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        #expect(try await ticket.wait()[0][3 * 64 + 3] == 0xff00_ff00)
        await device.close()
    } catch { await device.close(); throw error }
}

private func vulkanTaskMeshSource(taskSize: String) -> String {
    """
    enable wgpu_mesh_shader;
    struct Payload { color: vec4f, }
    struct V { @builtin(position) position: vec4f, @location(0) color: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O {
      @builtin(vertices) vertices: array<V, 3>,
      @builtin(primitives) primitives: array<P, 1>,
      @builtin(vertex_count) vertex_count: u32,
      @builtin(primitive_count) primitive_count: u32,
    }
    var<task_payload> payload: Payload;
    var<workgroup> out: O;
    @task @payload(payload) @workgroup_size(1) fn task_main() -> @builtin(mesh_task_size) vec3u {
      payload.color = vec4f(0.0, 1.0, 0.0, 1.0);
      return \(taskSize);
    }
    @mesh(out) @payload(payload) @workgroup_size(1) fn mesh() {
      out.vertex_count = 3u; out.primitive_count = 1u;
      out.vertices[0].position = vec4f(-1.0, -1.0, 0.5, 1.0);
      out.vertices[1].position = vec4f(3.0, -1.0, 0.5, 1.0);
      out.vertices[2].position = vec4f(-1.0, 3.0, 0.5, 1.0);
      out.vertices[0].color = payload.color;
      out.vertices[1].color = payload.color;
      out.vertices[2].color = payload.color;
      out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    @fragment fn fragment(@location(0) color: vec4f) -> @location(0) vec4f { return color; }
    """
}

/// Draws the task-driven triangle and returns the center pixel; blue means the mesh stage never ran.
private func vulkanTaskMeshPixel(taskSize: String) async throws -> UInt32? {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.meshShaders) else { await device.close(); return nil }
        let source = vulkanTaskMeshSource(taskSize: taskSize)
        let compiler = IndependentShaderCompiler()
        let task = try compiler.compileReflectedTaskStage(.wgsl(source), entryPoint: "task_main", to: .spirv)
        let mesh = try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fragment", to: .spirv)
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.init(red: 0, green: 0, blue: 1, alpha: 1)),
                      workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let pixel = try await ticket.wait()[0][3 * 64 + 3]
        await device.close()
        return pixel
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanTaskShadersPassPayloadsToMeshShaders() async throws {
    guard let pixel = try await vulkanTaskMeshPixel(taskSize: "vec3u(1u, 1u, 1u)") else { return }
    #expect(pixel == 0xff00_ff00)
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanTaskShadersDropOversizedMeshDispatches() async throws {
    // The task stage asks for more mesh workgroups than any device guarantees; the dispatch is dropped, not submitted.
    guard let pixel = try await vulkanTaskMeshPixel(taskSize: "vec3u(70000u, 1u, 1u)") else { return }
    #expect(pixel == 0xffff_0000)
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanDrawsLineMeshes() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.meshShaders) else { await device.close(); return }
        let source = """
            enable wgpu_mesh_shader;
            struct V { @builtin(position) position: vec4f, }
            struct P { @builtin(line_indices) indices: vec2u, }
            struct O {
              @builtin(vertices) vertices: array<V, 2>,
              @builtin(primitives) primitives: array<P, 1>,
              @builtin(vertex_count) vertex_count: u32,
              @builtin(primitive_count) primitive_count: u32,
            }
            var<workgroup> out: O;
            @mesh(out) @workgroup_size(1) fn mesh() {
              out.vertex_count = 2u; out.primitive_count = 1u;
              out.vertices[0].position = vec4f(-1.0, 0.0, 0.5, 1.0);
              out.vertices[1].position = vec4f(1.0, 0.0, 0.5, 1.0);
              out.primitives[0].indices = vec2u(0u, 1u);
            }
            @fragment fn fragment() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
            """
        let compiler = IndependentShaderCompiler()
        let pipeline = try await device.createMeshPipeline(
            mesh: try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh", to: .spirv),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fragment", to: .spirv), colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.init(red: 0, green: 0, blue: 1, alpha: 1)),
                      workgroups: .init(x: 1)),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let words = try await ticket.wait()[0]
        let green: UInt32 = 0xff00_ff00
        #expect(words[3 * 64 + 3] == green || words[4 * 64 + 3] == green)  // the line crosses the middle rows
        #expect(words[0] == 0xffff_0000)  // far from the line stays cleared
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: vulkanAvailable), .gpuExclusive) func vulkanMeshDrawsReadValidatedIndirectGroupCounts() async throws {
    let backend = try VulkanBackend.makeDevice()
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.meshShaders) else { await device.close(); return }
        let source = """
            enable wgpu_mesh_shader;
            struct V { @builtin(position) position: vec4f, }
            struct P { @builtin(triangle_indices) indices: vec3u, }
            struct O {
              @builtin(vertices) vertices: array<V, 3>,
              @builtin(primitives) primitives: array<P, 1>,
              @builtin(vertex_count) vertex_count: u32,
              @builtin(primitive_count) primitive_count: u32,
            }
            var<workgroup> out: O;
            @mesh(out) @workgroup_size(1) fn mesh() {
              out.vertex_count = 3u; out.primitive_count = 1u;
              out.vertices[0].position = vec4f(-1.0, -1.0, 0.5, 1.0);
              out.vertices[1].position = vec4f(3.0, -1.0, 0.5, 1.0);
              out.vertices[2].position = vec4f(-1.0, 3.0, 0.5, 1.0);
              out.primitives[0].indices = vec3u(0u, 1u, 2u);
            }
            @fragment fn fragment() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
            """
        let compiler = IndependentShaderCompiler()
        let pipeline = try await device.createMeshPipeline(
            mesh: try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh", to: .spirv),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fragment", to: .spirv), colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let arguments = try await device.createBuffer(.init(size: 32, usage: [.indirect, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func draw(offset: UInt64) -> BufferCommand {
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(blue), workgroups: .init(x: 0),
                      indirectBuffer: arguments, indirectOffset: offset)
        }
        let snapshot: [BufferCommand] = [
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 1824)]
        let ticket = try await device.submitCommands([
            .write(buffer: arguments, offset: 0, words: [1, 1, 1, 5_000_000, 1, 1]),
            draw(offset: 0),
        ] + snapshot + [draw(offset: 12)] + snapshot)  // 5,000,000 groups exceed the per-dimension limit, so nothing is drawn
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(snapshots[0][3 * 64 + 3] == 0xff00_ff00)
        #expect(snapshots[1][3 * 64 + 3] == 0xffff_0000)
        await device.close()
    } catch { await device.close(); throw error }
}
