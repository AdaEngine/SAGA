import Testing
import SAGACompiler

@Test func metalMutableRayDescriptorWithUnmaskedFlags() throws {
    _ = try IndependentShaderCompiler().compileCompute(.wgsl("""
    enable wgpu_ray_query;
    @group(0) @binding(0) var<storage, read_write> values: array<u32>;
    @group(0) @binding(1) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        var desc = RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0));
        desc.flags = values[0];
        rayQueryInitialize(&query, scene, desc);
        while rayQueryProceed(&query) {}
    }
    """), to: .msl)
}

#if canImport(Metal)
import Metal
import SAGATypes
import SAGACore
import SAGACompiler
@testable import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalMutableRayDescriptorOnGPU() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable wgpu_ray_query;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 10>;
        @group(0) @binding(1) var scene: acceleration_structure;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
            var query: ray_query;
            let flags = select(select(select(select(select(select(select(select(select(0u, 16u, id.x == 1u), 32u, id.x == 2u), 256u, id.x == 3u), 512u, id.x == 4u), 48u, id.x == 5u), 272u, id.x == 6u), 1024u, id.x == 7u), 16u, id.x == 8u), 32u, id.x == 9u);
            let origin = select(vec3f(0.25, 0.25, 0.0), vec3f(0.25, 0.25, 8.0), id.x >= 8u);
            let direction = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0), id.x >= 8u);
            var descriptor = RayDesc(0u, 255u, 0.1, 10.0, origin, direction);
            descriptor.flags = flags;
            rayQueryInitialize(&query, scene, descriptor);
            while rayQueryProceed(&query) { rayQueryConfirmIntersection(&query); }
            output[id.x] = rayQueryGetCommittedIntersection(&query).kind;
        }
        """), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let vertices = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 40, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 40, usage: [.copyDestination, .mapRead]))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [0, 0, Float(4).bitPattern, Float(1).bitPattern, 0, Float(4).bitPattern, 0, Float(1).bitPattern, Float(4).bitPattern]),
        .buildAccelerationStructure(destination: bottom, geometry: geometry),
        .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: bottom)]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(group: 0, binding: 0): output], workgroups: .init(x: 10), accelerationStructures: [.init(group: 0, binding: 1): top]),
        .copy(source: output, destination: readback, byteCount: 40),
        .read(buffer: readback, byteCount: 40)
    ])
    let result = try await ticket.wait()
    #expect(result == [[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]])
    await device.close()
}
#endif
