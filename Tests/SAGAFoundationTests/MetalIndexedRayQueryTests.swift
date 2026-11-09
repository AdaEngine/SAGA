#if canImport(Metal)
import Metal
import Testing
import SAGATypes
import SAGACore
import SAGACompiler
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalWGSLRayQueryReadsIndexedTriangles() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable wgpu_ray_query;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 1>;
        @group(0) @binding(1) var scene: acceleration_structure;
        @compute @workgroup_size(1) fn main() {
            var q: ray_query;
            rayQueryInitialize(&q, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.25, 0.25, 0.0), vec3f(0.0, 0.0, 1.0)));
            while rayQueryProceed(&q) { rayQueryConfirmIntersection(&q); }
            output[0] = rayQueryGetCommittedIntersection(&q).kind;
        }
        """), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let vertices = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 12, usage: [.index, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    for format in [IndexFormat.uint16, .uint32] {
        let words: [UInt32] = format == .uint16 ? [0x0002_0001, 0] : [1, 2, 0]
        let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3,
            indexBuffer: .init(buffer: indices, format: format), indexCount: 3)
        let ticket = try await device.submitCommands([
            .write(buffer: vertices, offset: 0, words: [0, 0, Float(4).bitPattern, Float(1).bitPattern, 0, Float(4).bitPattern, 0, Float(1).bitPattern, Float(4).bitPattern]),
            .write(buffer: indices, offset: 0, words: words),
            .buildAccelerationStructure(destination: bottom, geometry: geometry),
            .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: bottom)]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1),
                             accelerationStructures: [.init(binding: 1): top]),
            .copy(source: output, destination: readback, byteCount: 4), .read(buffer: readback, byteCount: 4)
        ])
        #expect(try await ticket.wait() == [[1]])
    }
    await device.close()
}
#endif
