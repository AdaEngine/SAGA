#if os(Windows)
import Foundation
import WinSDK
import Testing
import SAGATypes
import SAGABackend
import SAGACore
import SAGAShaderIR
import SAGACompiler
import SAGAD3D12

// These tests run on a real Direct3D 12 device and need the DirectX Shader Compiler. Without either they are skipped.
private let dxcDirectories: [String] = {
    let candidates = [
        #"C:\VulkanSDK\1.4.313.0\Bin"#,
        #"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22000.0\x64"#,
        #"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"#,
    ]
    return candidates.filter { FileManager.default.fileExists(atPath: $0) }
}()

private let d3d12Available: Bool = {
    guard D3D12Backend.isAvailable() else { return false }
    return dxcDirectories.contains { FileManager.default.fileExists(atPath: $0 + #"\dxcompiler.dll"#) }
}()

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12RunsHlslComputeWithUniformAndStorageBuffers() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            struct Parameters { factor: u32 }
            @group(0) @binding(0) var<uniform> parameters: Parameters;
            @group(0) @binding(1) var<storage, read> source: array<u32, 4>;
            @group(0) @binding(3) var<storage, read_write> destination: array<u32, 4>;

            @compute @workgroup_size(4)
            fn run(@builtin(global_invocation_id) id: vec3<u32>) {
                destination[id.x] = source[id.x] * parameters.factor + 1u;
            }
            """
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl)
        let pipeline = try await device.createComputePipeline(shader)
        let factor = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let input = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: factor, offset: 0, words: [3, 0, 0, 0]),
            .write(buffer: input, offset: 0, words: [1, 2, 3, 4]),
            .dispatchCompute(pipeline: pipeline, bindings: [
                ComputeBindingSlot(binding: 0): factor,
                ComputeBindingSlot(binding: 1): input,
                ComputeBindingSlot(binding: 3): output,
            ], workgroups: .init(x: 1)),
            .copy(source: output, destination: readback, byteCount: 16),
            .read(buffer: readback, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[4, 7, 10, 13]])
        await device.close()
    } catch { await device.close(); throw error }
}
#endif

#if os(Windows)
@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12PassesImmediatesWorkgroupCountsAndValidatesIndirectDispatch() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let counted = try await device.createComputePipeline(try compiler.compileCompute(.wgsl("""
            struct Parameters { factor: u32, add: u32 }
            var<immediate> imm: Parameters;
            @group(0) @binding(3) var<storage, read_write> destination: array<u32, 4>;
            @compute @workgroup_size(4)
            fn run(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
                destination[id.x] = id.x * imm.factor + imm.add + n.x;
            }
            """), to: .hlsl))
        let plain = try await device.createComputePipeline(try compiler.compileCompute(.wgsl("""
            @group(0) @binding(3) var<storage, read_write> destination: array<u32, 4>;
            @compute @workgroup_size(4)
            fn run(@builtin(local_invocation_index) i: u32) { destination[i] = destination[i] + 1u; }
            """), to: .hlsl))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource, .copyDestination]))
        let arguments = try await device.createBuffer(.init(size: 24, usage: [.indirect, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        func snapshot() -> [BufferCommand] {
            [.copy(source: output, destination: readback, byteCount: 16), .read(buffer: readback, byteCount: 16)]
        }
        let slot = ComputeBindingSlot(binding: 3)
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: counted, bindings: [slot: output], workgroups: .init(x: 1), immediates: [10, 5]),
        ] + snapshot() + [
            .write(buffer: arguments, offset: 0, words: [1, 1, 1, 70_000, 1, 1]),
            .clearBuffer(buffer: output, offset: 0, byteCount: nil),
            .dispatchComputeIndirect(pipeline: plain, bindings: [slot: .init(buffer: output)], indirectBuffer: arguments),
        ] + snapshot() + [
            // 70_000 groups exceeds the per-dimension limit, so the validated count is zero and nothing runs.
            .dispatchComputeIndirect(pipeline: plain, bindings: [slot: .init(buffer: output)], indirectBuffer: arguments,
                                     indirectOffset: 12),
        ] + snapshot())
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 3)
        #expect(snapshots[0] == [6, 16, 26, 36])  // id * 10 + 5 + one workgroup
        #expect(snapshots[1] == [1, 1, 1, 1])
        #expect(snapshots[2] == [1, 1, 1, 1])
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12SamplesTexturesAndWritesStorageTextures() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @group(0) @binding(0) var source: texture_2d<f32>;
            @group(0) @binding(1) var filtering: sampler;
            @group(0) @binding(2) var target: texture_storage_2d<rgba8unorm, write>;
            @compute @workgroup_size(2, 2)
            fn run(@builtin(global_invocation_id) id: vec3<u32>) {
                let uv = (vec2f(id.xy) + vec2f(0.5)) / 2.0;
                textureStore(target, vec2<i32>(id.xy), textureSampleLevel(source, filtering, uv, 0.0));
            }
            """
        let pipeline = try await device.createComputePipeline(
            try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl))
        let input = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                         usage: [.textureBinding, .copyDestination]))
        let output = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                          usage: [.storageBinding, .copySource]))
        let sampler = try await device.createSampler(.init())
        let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let colors: [UInt8] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        let ticket = try await device.submitCommands([
            .writeTexture(destination: input, data: colors, layout: .init(bytesPerRow: 8, rowsPerImage: 2),
                          region: .init(width: 2, height: 2)),
            .dispatchCompute(pipeline: pipeline, bindings: [:], workgroups: .init(x: 1),
                             textures: [ComputeBindingSlot(binding: 0): input, ComputeBindingSlot(binding: 2): output],
                             samplers: [ComputeBindingSlot(binding: 1): sampler]),
            .copyTextureToBuffer(source: output, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 264),
        ])
        let words = try await ticket.wait()[0]
        func texel(_ x: Int, _ y: Int) -> UInt32 { words[y * 64 + x] }
        #expect(texel(0, 0) == UInt32(10) | UInt32(20) << 8 | UInt32(30) << 16 | UInt32(40) << 24)
        #expect(texel(1, 0) == UInt32(50) | UInt32(60) << 8 | UInt32(70) << 16 | UInt32(80) << 24)
        #expect(texel(0, 1) == UInt32(90) | UInt32(100) << 8 | UInt32(110) << 16 | UInt32(120) << 24)
        #expect(texel(1, 1) == UInt32(130) | UInt32(140) << 8 | UInt32(150) << 16 | UInt32(160) << 24)
        await device.close()
    } catch { await device.close(); throw error }
}
#endif

#if os(Windows)
@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12LoadsPlanarExternalTexturesInCompute() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @group(0) @binding(0) var rgba: texture_external;
            @group(0) @binding(1) var nv12: texture_external;
            @group(0) @binding(2) var<storage, read_write> output: array<vec4f, 3>;
            @compute @workgroup_size(1) fn run() {
                output[0] = textureLoad(rgba, vec2u(1, 0));
                output[1] = textureLoad(nv12, vec2u(0));
                output[2] = textureLoad(nv12, vec2u(1, 0));
            }
            """
        let pipeline = try await device.createComputePipeline(try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl))
        let usage: TextureUsage = [.textureBinding, .copyDestination]
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: usage))
        let luma = try await device.createTexture(.init(width: 2, height: 1, format: .r8Unorm, usage: usage))
        let chroma = try await device.createTexture(.init(width: 1, height: 1, format: .rg8Unorm, usage: usage))
        let frame = try await device.importExternalTexture(.init(format: .rgba, planes: [color]))
        let video = try await device.importExternalTexture(.init(format: .nv12, planes: [luma, chroma]))
        let output = try await device.createBuffer(.init(size: 48, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 48, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .writeTexture(destination: color, data: [255, 0, 0, 255, 0, 255, 0, 255], region: .init(width: 2, height: 1)),
            .writeTexture(destination: luma, data: [255, 0], region: .init(width: 2, height: 1)),
            .writeTexture(destination: chroma, data: [128, 128], region: .init(width: 1, height: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: [ComputeBindingSlot(binding: 2): output], workgroups: .init(x: 1),
                             textures: [ComputeBindingSlot(binding: 0): frame, ComputeBindingSlot(binding: 1): video]),
            .copy(source: output, destination: read),
            .read(buffer: read, byteCount: 48),
        ])
        let values = try await ticket.wait()[0].map { Float(bitPattern: $0) }
        #expect(values[0] == 0 && values[1] == 1 && values[2] == 0 && values[3] == 1)  // the green RGBA texel
        // With the identity conversion matrix the planes read as (Y, Cb, Cr) in RGB.
        #expect(values[4] == 1 && abs(values[5] - 128.0 / 255) < 0.01 && abs(values[6] - 128.0 / 255) < 0.01)
        #expect(values[8] == 0)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12RendersWithExternalTexturesInFragmentShaders() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @group(0) @binding(0) var frame: texture_external;
            @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
                let xy = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                return vec4f(xy[i], 0.5, 1);
            }
            @fragment fn fs() -> @location(0) vec4f { return textureLoad(frame, vec2u(1, 0)); }
            """
        let compiler = IndependentShaderCompiler()
        let pipeline = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .hlsl), colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let color = try await device.createTexture(.init(width: 2, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination]))
        let frame = try await device.importExternalTexture(.init(format: .rgba, planes: [color]))
        let read = try await device.createBuffer(.init(size: 1824, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .writeTexture(destination: color, data: [255, 0, 0, 255, 0, 255, 0, 255], region: .init(width: 2, height: 1)),
            .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3,
                    textures: [ComputeBindingSlot(binding: 0): frame]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        #expect(try await ticket.wait()[0][3 * 64 + 3] == 0xff00_ff00)  // the second texel is green
        await device.close()
    } catch { await device.close(); throw error }
}

private func d3d12MeshSource(withTask: Bool) -> String {
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
      return vec3u(1u, 1u, 1u);
    }
    @mesh(out) @payload(payload) @workgroup_size(1) fn mesh_task() {
      out.vertex_count = 3u; out.primitive_count = 1u;
      out.vertices[0].position = vec4f(-1.0, -1.0, 0.5, 1.0);
      out.vertices[1].position = vec4f(3.0, -1.0, 0.5, 1.0);
      out.vertices[2].position = vec4f(-1.0, 3.0, 0.5, 1.0);
      out.vertices[0].color = payload.color;
      out.vertices[1].color = payload.color;
      out.vertices[2].color = payload.color;
      out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    @mesh(out) @workgroup_size(1) fn mesh_plain() {
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
}

private func d3d12MeshPixel(withTask: Bool) async throws -> UInt32? {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.meshShaders) else { await device.close(); return nil }
        let source = d3d12MeshSource(withTask: withTask)
        let compiler = IndependentShaderCompiler()
        let task = withTask ? try compiler.compileReflectedTaskStage(.wgsl(source), entryPoint: "task_main", to: .hlsl) : nil
        let mesh = try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: withTask ? "mesh_task" : "mesh_plain", to: .hlsl)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fragment", to: .hlsl)
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
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

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12DrawsWithMeshShaders() async throws {
    guard let pixel = try await d3d12MeshPixel(withTask: false) else { return }
    #expect(pixel == 0xff00_ff00)
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12DrawsWithTaskAndMeshShaders() async throws {
    guard let pixel = try await d3d12MeshPixel(withTask: true) else { return }
    #expect(pixel == 0xff00_ff00)
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12MeshDrawsReadValidatedIndirectGroupCounts() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        guard backend.capabilities.features.contains(.meshShaders) else { await device.close(); return }
        let source = d3d12MeshSource(withTask: false)
        let compiler = IndependentShaderCompiler()
        let pipeline = try await device.createMeshPipeline(
            mesh: try compiler.compileReflectedMeshStage(.wgsl(source), entryPoint: "mesh_plain", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fragment", to: .hlsl), colorFormat: .rgba8Unorm)
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

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12RendersVertexBuffersTexturesUniformsAndIndexedDraws() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            struct Tint { color: vec4f }
            @group(0) @binding(0) var<uniform> tint: Tint;
            @group(0) @binding(1) var image: texture_2d<f32>;
            @group(0) @binding(2) var filtering: sampler;
            struct Varyings { @builtin(position) position: vec4f, @location(0) uv: vec2f }
            @vertex fn vs(@location(0) position: vec2f, @location(1) uv: vec2f) -> Varyings {
                return Varyings(vec4f(position, 0.5, 1.0), uv);
            }
            @fragment fn fs(v: Varyings) -> @location(0) vec4f { return textureSample(image, filtering, v.uv) * tint.color; }
            """
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .hlsl)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .hlsl)
        let layout = VertexBufferLayout(stride: 16, attributes: [
            .init(location: 0, offset: 0, format: .float32x2), .init(location: 1, offset: 8, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                             vertexBuffers: [layout])
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                         usage: [.textureBinding, .copyDestination]))
        let sampler = try await device.createSampler(.init())
        let tint = try await device.createBuffer(.init(size: 16, usage: [.uniform, .copyDestination]))
        let vertices = try await device.createBuffer(.init(size: 64, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))
        func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let slots = (uniform: ComputeBindingSlot(binding: 0), image: ComputeBindingSlot(binding: 1), sampler: ComputeBindingSlot(binding: 2))
        let ticket = try await device.submitCommands([
            .writeTexture(destination: image, data: [255, 128, 0, 255], region: .init(width: 1, height: 1)),
            .write(buffer: tint, offset: 0, words: bits([1, 1, 1, 1])),
            // A triangle covering the whole target, then four vertices for an indexed draw.
            .write(buffer: vertices, offset: 0, words: bits([-1, -1, 0, 0, 3, -1, 0, 0, -1, 3, 0, 0])),
            .write(buffer: indices, offset: 0, words: [0x0002_0001, 0x0000_0000]),
            .render(pipeline: pipeline, target: target, clear: blue, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)],
                    uniforms: [slots.uniform: .init(buffer: tint)], textures: [slots.image: image], samplers: [slots.sampler: sampler]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .renderIndexed(pipeline: pipeline, target: target, clear: blue, indexBuffer: .init(buffer: indices, format: .uint16),
                           draw: .init(indexCount: 3), vertexBuffers: [.init(buffer: vertices)],
                           uniforms: [slots.uniform: .init(buffer: tint)], textures: [slots.image: image],
                           samplers: [slots.sampler: sampler]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        // The sampled texel is (255, 128, 0, 255) in rgba order.
        #expect(snapshots[0][3 * 64 + 3] == 0xff00_80ff)
        #expect(snapshots[1][3 * 64 + 3] == 0xff00_80ff)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12DepthTestsRejectFartherFragmentsAndResolvesMultisampledTargets() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
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
        let compiler = IndependentShaderCompiler()
        func stage(_ entry: String) throws -> RenderStageArtifact {
            try compiler.compileRenderStage(.wgsl(source), entryPoint: entry, to: .hlsl)
        }
        let less = RenderDepthState(format: .depth32Float, compare: .less)
        func pipeline(_ vertex: String, _ fragment: String, samples: UInt32) async throws -> RenderPipeline {
            try await device.createRenderPipeline(vertex: stage(vertex), fragment: stage(fragment), colorFormat: .rgba8Unorm,
                                                  depthState: less, sampleCount: samples)
        }
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let read = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))

        // Single-sampled: the far draw loses the depth test and depth keeps 0.25.
        let near1 = try await pipeline("near", "green", samples: 1)
        let far1 = try await pipeline("far", "red", samples: 1)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource]))
        let depth = try await device.createTexture(.init(width: 8, height: 8, format: .depth32Float,
                                                         usage: [.renderAttachment, .copySource]))
        func draw(_ pipeline: RenderPipeline, first: Bool) -> BufferCommand {
            .render(pipeline: pipeline, target: target, colorLoad: first ? .clear(blue) : .load, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: first ? .clear(1) : .load))
        }
        let first = try await device.submitCommands([
            draw(near1, first: true), draw(far1, first: false),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
            .copyTextureToBuffer(source: depth, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        let snapshots = try await first.wait()
        try #require(snapshots.count == 2)
        #expect(snapshots[0][3 * 64 + 3] == 0xff00_ff00)
        #expect(snapshots[1][3 * 64 + 3] == Float(0.25).bitPattern)

        // 4x multisampled with depth, resolved into a single-sampled texture.
        let near4 = try await pipeline("near", "green", samples: 4)
        let color4 = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment], sampleCount: 4))
        let depth4 = try await device.createTexture(.init(width: 8, height: 8, format: .depth32Float, usage: [.renderAttachment], sampleCount: 4))
        let resolved = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let second = try await device.submitCommands([
            .render(pipeline: near4, target: color4, clear: blue, vertexCount: 3,
                    depth: .init(texture: depth4, loadAction: .clear(1)), resolve: .init(texture: resolved)),
            .copyTextureToBuffer(source: resolved, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        #expect(try await second.wait()[0][3 * 64 + 3] == 0xff00_ff00)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12MapsBuffersForHostWritesAndReads() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
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

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12OcclusionAndTimestampQueriesResolveToBuffers() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let compiler = IndependentShaderCompiler()
        let pipeline = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .hlsl), colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment]))
        let occlusion = try await device.createQuerySet(.init(type: .occlusion, count: 2))
        let timestamps = try await device.createQuerySet(.init(type: .timestamp, count: 2))
        let resolved = try await device.createBuffer(.init(size: 1024, usage: [.queryResolve, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func pass(vertexCount: UInt32, query: UInt32) -> BufferCommand {
            .occlusionQueried(pass: .render(pipeline: pipeline, target: target, clear: blue, vertexCount: vertexCount),
                              query: .init(querySet: occlusion, queryIndex: query))
        }
        let ticket = try await device.submitCommands([
            pass(vertexCount: 3, query: 0),
            pass(vertexCount: 0, query: 1),
            .resolveQuerySet(querySet: occlusion, firstQuery: 0, queryCount: 2, destination: resolved),
            .copy(source: resolved, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
            .timestamped(pass: .render(pipeline: pipeline, target: target, clear: blue, vertexCount: 3),
                         writes: .init(querySet: timestamps, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1)),
            .resolveQuerySet(querySet: timestamps, firstQuery: 0, queryCount: 2, destination: resolved, destinationOffset: 256),
            .copy(source: resolved, sourceOffset: 256, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(snapshots[0][0] != 0 || snapshots[0][1] != 0)  // visible samples
        #expect(snapshots[0][2] == 0 && snapshots[0][3] == 0)  // nothing drawn
        let begin = UInt64(snapshots[1][1]) << 32 | UInt64(snapshots[1][0])
        let end = UInt64(snapshots[1][3]) << 32 | UInt64(snapshots[1][2])
        #expect(begin != 0 && end >= begin)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12DrawsWithGPUAuthoredArguments() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
            """
        let compiler = IndependentShaderCompiler()
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .hlsl),
            colorFormat: .rgba8Unorm, vertexBuffers: [layout])
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
        let arguments = try await device.createBuffer(.init(size: 64, usage: [.indirect, .copyDestination]))
        let read = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))
        func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        func draw(offset: UInt64, indexed: Bool) -> BufferCommand {
            .renderIndirect(pipeline: pipeline, target: target, colorLoad: .clear(blue), indirectBuffer: arguments,
                            indirectOffset: offset, indexBuffer: indexed ? .init(buffer: indices, format: .uint16) : nil,
                            vertexBuffers: [.init(buffer: vertices)])
        }
        let snapshot: [BufferCommand] = [
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256), .read(buffer: read, byteCount: 1824)]
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: bits([-1, -1, 3, -1, -1, 3])),
            .write(buffer: indices, offset: 0, words: [0x0001_0000, 0x0000_0002]),
            // [vertexCount, instanceCount, firstVertex, firstInstance], then the indexed layout at 16.
            .write(buffer: arguments, offset: 0, words: [3, 1, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0]),
            draw(offset: 0, indexed: false),
        ] + snapshot + [draw(offset: 16, indexed: true)] + snapshot)
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 2)
        #expect(snapshots[0][3 * 64 + 3] == 0xff00_ff00)
        #expect(snapshots[1][3 * 64 + 3] == 0xff00_ff00)
        await device.close()
    } catch { await device.close(); throw error }
}

/// A window owned by its own thread so Direct3D can present into it.
private final class D3D12TestWindow: @unchecked Sendable {
    private(set) var handle: HWND?
    private let ready = DispatchSemaphore(value: 0)
    private let wake = DispatchSemaphore(value: 0)
    private let done = DispatchSemaphore(value: 0)

    init() {
        let thread = Thread { [self] in
            let module = GetModuleHandleW(nil)
            handle = "STATIC".withCString(encodedAs: UTF16.self) { windowClass in
                "SAGA".withCString(encodedAs: UTF16.self) { title in
                    CreateWindowExW(0, windowClass, title, DWORD(WS_OVERLAPPEDWINDOW), 0, 0, 200, 150, nil, nil, module, nil)
                }
            }
            ready.signal()
            wake.wait()
            if let handle { DestroyWindow(handle) }
            done.signal()
        }
        thread.start()
        ready.wait()
    }

    func close() { wake.signal(); done.wait() }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12PresentsToAWindowAndRebuildsTheSwapChainOnResize() async throws {
    let window = D3D12TestWindow()
    defer { window.close() }
    try #require(window.handle != nil)
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories,
                                              windowHandle: UInt(bitPattern: window.handle))
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
            """
        let compiler = IndependentShaderCompiler()
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .hlsl),
            colorFormat: .bgra8Unorm, vertexBuffers: [layout])
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        func frame(_ size: UInt32) async throws -> ResourceID {
            let texture = try await device.createTexture(.init(width: size, height: size, format: .bgra8Unorm,
                                                               usage: [.renderAttachment, .copySource]))
            _ = try await device.submitCommands([
                .write(buffer: vertices, offset: 0, words: [-1, -1, 3, -1, -1, 3].map { Float($0).bitPattern }),
                .render(pipeline: pipeline, target: texture, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3,
                        vertexBuffers: [.init(buffer: vertices)])])
            return texture
        }
        let small = try await frame(64)
        try await device.present(small).wait()
        try await device.present(small).wait()
        #expect(await backend.presentationRebuildsForTesting() == 1)
        let large = try await frame(96)
        try await device.present(large).wait()
        #expect(await backend.presentationRebuildsForTesting() == 2)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12TextureViewsSelectMipLevelsAndArrayLayers() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
            """
        let compiler = IndependentShaderCompiler()
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .hlsl),
            colorFormat: .rgba8Unorm, vertexBuffers: [layout])
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let mips = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm,
                                                        usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let layers = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
                                                          usage: [.renderAttachment, .copySource], arrayLayerCount: 2))
        let level1 = try await device.createTextureView(texture: mips, .init(baseMipLevel: 1, mipLevelCount: 1))
        let layer1 = try await device.createTextureView(texture: layers, .init(dimension: .d2, baseArrayLayer: 1, arrayLayerCount: 1))
        let read = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))
        let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [-1, -1, 3, -1, -1, 3].map { Float($0).bitPattern }),
            .render(pipeline: pipeline, target: level1, clear: blue, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
            .copyTextureToBuffer(source: mips, destination: read, bytesPerRow: 256, mipLevel: 1),
            .read(buffer: read, byteCount: 784),
            .copyTextureToBuffer(source: mips, destination: read, bytesPerRow: 256, mipLevel: 0),
            .read(buffer: read, byteCount: 1824),
            .render(pipeline: pipeline, target: layer1, clear: blue, vertexCount: 3, vertexBuffers: [.init(buffer: vertices)]),
            .copyTextureToBuffer(source: layers, destination: read, bytesPerRow: 256, arrayLayer: 0),
            .read(buffer: read, byteCount: 784),
            .copyTextureToBuffer(source: layers, destination: read, bytesPerRow: 256, arrayLayer: 1),
            .read(buffer: read, byteCount: 784),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.count == 4)
        #expect(snapshots[0][1 * 64 + 1] == 0xff00_ff00)  // mip 1 was rendered
        #expect(snapshots[1][1 * 64 + 1] == 0)  // mip 0 was not
        #expect(snapshots[2][1 * 64 + 1] == 0)  // layer 0 was not
        #expect(snapshots[3][1 * 64 + 1] == 0xff00_ff00)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12BindsBufferAndTextureArrays() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            enable wgpu_binding_array;
            struct Data { values: array<u32, 4> }
            @group(0) @binding(0) var<storage, read_write> destination: array<u32, 4>;
            @group(0) @binding(1) var<storage, read> sources: binding_array<Data, 2>;
            @group(0) @binding(4) var images: binding_array<texture_2d<f32>, 3>;
            @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) i: u32) {
                destination[i] = sources[0].values[i] * 10u + sources[1].values[i] + u32(textureLoad(images[2], vec2<i32>(0, 0), 0).x);
            }
            """
        let pipeline = try await device.createComputePipeline(
            try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl))
        let first = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let second = try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        var images: [ResourceID] = []
        for _ in 0..<3 { images.append(try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination]))) }
        var bindings: [ComputeBindingSlot: ResourceID] = [
            ComputeBindingSlot(binding: 0): output,
            ComputeBindingSlot(binding: 1, arrayElement: 0): first,
            ComputeBindingSlot(binding: 1, arrayElement: 1): second,
        ]
        _ = bindings.removeValue(forKey: ComputeBindingSlot(binding: 9))
        let ticket = try await device.submitCommands([
            .write(buffer: first, offset: 0, words: [1, 2, 3, 4]),
            .write(buffer: second, offset: 0, words: [5, 6, 7, 8]),
            .writeTexture(destination: images[2], data: [255, 255, 255, 255], region: .init(width: 1, height: 1)),
            .dispatchCompute(pipeline: pipeline, bindings: bindings, workgroups: .init(x: 1), textures: [
                ComputeBindingSlot(binding: 4, arrayElement: 0): images[0],
                ComputeBindingSlot(binding: 4, arrayElement: 1): images[1],
                ComputeBindingSlot(binding: 4, arrayElement: 2): images[2],
            ]),
            .copy(source: output, destination: read, byteCount: 16),
            .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[16, 27, 38, 49]])
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12RenderShadersUseBindingArrays() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let source = """
            enable wgpu_binding_array;
            struct Tint { value: vec4f }
            @group(0) @binding(0) var<storage, read> tints: binding_array<Tint, 2>;
            @group(0) @binding(1) var images: binding_array<texture_2d<f32>, 2>;
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f {
                return tints[1].value * textureLoad(images[1], vec2<i32>(0, 0), 0);
            }
            """
        let compiler = IndependentShaderCompiler()
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let pipeline = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .hlsl),
            colorFormat: .rgba8Unorm, vertexBuffers: [layout])
        let target = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let tints = [try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination])),
                     try await device.createBuffer(.init(size: 16, usage: [.storage, .copyDestination]))]
        var images: [ResourceID] = []
        for _ in 0..<2 { images.append(try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination]))) }
        let read = try await device.createBuffer(.init(size: 2048, usage: [.copyDestination, .mapRead]))
        func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: bits([-1, -1, 3, -1, -1, 3])),
            .write(buffer: tints[0], offset: 0, words: bits([1, 0, 0, 1])),
            .write(buffer: tints[1], offset: 0, words: bits([0, 1, 0, 1])),
            .writeTexture(destination: images[1], data: [255, 255, 255, 255], region: .init(width: 1, height: 1)),
            .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)],
                    textures: [ComputeBindingSlot(binding: 1, arrayElement: 0): images[0], ComputeBindingSlot(binding: 1, arrayElement: 1): images[1]],
                    storageBuffers: [ComputeBindingSlot(binding: 0, arrayElement: 0): .init(buffer: tints[0]),
                                     ComputeBindingSlot(binding: 0, arrayElement: 1): .init(buffer: tints[1])]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 1824),
        ])
        #expect(try await ticket.wait()[0][3 * 64 + 3] == 0xff00_ff00)
        await device.close()
    } catch { await device.close(); throw error }
}

@Test(.enabled(if: d3d12Available), .gpuExclusive) func d3d12SamplesDepthTexturesAndDepthAspectViews() async throws {
    let backend = try D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories)
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let drawSource = """
            @vertex fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.5, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
            """
        let layout = VertexBufferLayout(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])
        let draw = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(drawSource), entryPoint: "vs", to: .hlsl),
            fragment: try compiler.compileRenderStage(.wgsl(drawSource), entryPoint: "fs", to: .hlsl),
            colorFormat: .rgba8Unorm, vertexBuffers: [layout],
            depthState: RenderDepthState(format: .depth24PlusStencil8, compare: .always, writeEnabled: true))
        let reader = try await device.createComputePipeline(try compiler.compileCompute(.wgsl("""
            @group(0) @binding(0) var depthImage: texture_depth_2d;
            @group(0) @binding(3) var<storage, read_write> destination: array<f32, 1>;
            @compute @workgroup_size(1) fn run() { destination[0] = textureLoad(depthImage, vec2<i32>(3, 3), 0); }
            """), to: .hlsl))
        let color = try await device.createTexture(.init(width: 8, height: 8, format: .rgba8Unorm, usage: [.renderAttachment]))
        let depth = try await device.createTexture(.init(width: 8, height: 8, format: .depth24PlusStencil8,
                                                         usage: [.renderAttachment, .textureBinding]))
        let depthOnly = try await device.createTextureView(texture: depth, .init(aspect: .depthOnly))
        let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [-1, -1, 3, -1, -1, 3].map { Float($0).bitPattern }),
            .render(pipeline: draw, target: color, clear: .init(red: 0, green: 0, blue: 0, alpha: 1), vertexCount: 3,
                    vertexBuffers: [.init(buffer: vertices)],
                    depth: .init(texture: depth, loadAction: .clear(1), stencil: .init(loadAction: .clear(0)))),
            .dispatchCompute(pipeline: reader, bindings: [ComputeBindingSlot(binding: 3): output], workgroups: .init(x: 1),
                             textures: [ComputeBindingSlot(binding: 0): depthOnly]),
            .copy(source: output, destination: read, byteCount: 4),
            .read(buffer: read, byteCount: 4),
        ])
        let value = Float(bitPattern: try await ticket.wait()[0][0])
        #expect(abs(value - 0.5) < 0.001)  // the draw wrote depth 0.5
        await device.close()
    } catch { await device.close(); throw error }
}
#endif
