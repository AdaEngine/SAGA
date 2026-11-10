#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshIndirectRejectsGPUAuthoredGroupsAboveMeshLimit() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let limit = device.capabilities.maxMeshWorkgroupsPerDimension
        // A broken sanitizer must not accidentally launch a huge mesh workload.
        let checkInvalid = limit < 4096
        let invalidCount = checkInvalid ? limit + 1 : 0
        let compiler = IndependentShaderCompiler()
        let producer = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
            @group(0) @binding(0) var<storage, read_write> args: array<u32, 6>;
            @compute @workgroup_size(1) fn main() {
                args[0] = \(invalidCount)u; args[1] = 1u; args[2] = 1u;
                args[3] = 1u; args[4] = 1u; args[5] = 1u;
            }
            """), to: .msl))
        let mesh = try compiler.compileReflectedMeshStage(.wgsl("""
            enable wgpu_mesh_shader;
            struct V { @builtin(position) position: vec4f, }
            struct P { @builtin(triangle_indices) indices: vec3u, }
            struct Output {
                @builtin(vertices) vertices: array<V, 3>,
                @builtin(primitives) primitives: array<P, 1>,
                @builtin(vertex_count) vertex_count: u32,
                @builtin(primitive_count) primitive_count: u32,
            }
            var<workgroup> output: Output;
            @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
            @mesh(output) @workgroup_size(1) fn mesh() {
                atomicAdd(&counter, 1u);
                output.vertex_count = 0u;
                output.primitive_count = 0u;
            }
            """), entryPoint: "mesh", to: .msl)
        let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4u { return vec4u(0u); }"), entryPoint: "fs", to: .msl)
        let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Uint)
        let arguments = try await device.createBuffer(.init(size: 24, usage: [.storage, .indirect]))
        let counter = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint, usage: .renderAttachment))
        var commands: [BufferCommand] = [
            .write(buffer: counter, offset: 0, words: [0]),
            .dispatchCompute(pipeline: producer, bindings: [.init(binding: 0): arguments], workgroups: .init(x: 1)),
        ]
        if checkInvalid {
            commands += [
                .drawMeshIndirect(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack),
                                  indirectBuffer: arguments, meshBuffers: [.init(binding: 0): .init(buffer: counter)]),
                .copy(source: counter, destination: readback, byteCount: 4),
                .read(buffer: readback, byteCount: 4),
            ]
        }
        commands += [
            .drawMeshIndirect(pipeline: pipeline, target: target, colorLoad: .load,
                              indirectBuffer: arguments, indirectOffset: 12,
                              meshBuffers: [.init(binding: 0): .init(buffer: counter)]),
            .copy(source: counter, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]
        let result = try await device.submitCommands(commands).wait()
        #expect(result == (checkInvalid ? [[0], [1]] : [[1]]))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
