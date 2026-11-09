#if canImport(Metal)
import Metal
import Testing
import SAGATypes
import SAGACore
import SAGACompiler
@testable import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalWGSLRayQueryUsesBoundTopLevelStructure() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable wgpu_ray_query;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 21>;
        @group(0) @binding(1) var scene: acceleration_structure;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
            var query: ray_query;
            let origin = select(vec3f(2.25, 3.25, 0.0), vec3f(0.25, 0.25, 0.0), id.x == 1u);
            let mask = select(2u, 1u, id.x == 2u);
            rayQueryInitialize(&query, scene, RayDesc(select(select(select(select(select(0u, 4u, id.x == 0u), 1u, id.x == 3u), 2u, id.x == 4u), 64u, id.x == 5u), 128u, id.x == 6u) & 199u, mask, 0.1, 10.0,
                origin, vec3f(0.0, 0.0, 1.0)));
            while rayQueryProceed(&query) { rayQueryConfirmIntersection(&query); }
            let hit = rayQueryGetCommittedIntersection(&query);
            output[id.x * 3u] = hit.kind;
            output[id.x * 3u + 1u] = bitcast<u32>(hit.t);
            output[id.x * 3u + 2u] = hit.instance_custom_data;
        }
        """), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let vertices = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 84, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 84, usage: [.copyDestination, .mapRead]))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [0, 0, 0, Float(1).bitPattern, 0, 0, 0, Float(1).bitPattern, 0]),
        .buildAccelerationStructure(destination: bottom, geometry: geometry),
        .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: bottom,
            transform: [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4], mask: 2, instanceID: 73)]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(group: 0, binding: 0): output], workgroups: .init(x: 7),
                         accelerationStructures: [.init(group: 0, binding: 1): top]),
        .copy(source: output, destination: readback, byteCount: 84),
        .read(buffer: readback, byteCount: 84)
    ])
    let result = try await ticket.wait()
    #expect(result == [[1, Float(4).bitPattern, 73, 0, 0, 0, 0, 0, 0,
                        1, Float(4).bitPattern, 73, 1, Float(4).bitPattern, 73,
                        1, Float(4).bitPattern, 73, 0, 0, 0]])
    await device.close()
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalBuildsTriangleAccelerationStructure() throws {
    guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else { return }
    let vertices: [Float] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    let buffer = vertices.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count) }!
    let queue = device.makeCommandQueue()!
    let commands = queue.makeCommandBuffer()!
    let build = try MetalAccelerationStructureBuild.encode(device: device, commandBuffer: commands,
        vertexBuffer: buffer, vertexOffset: 0, vertexStride: 12, vertexCount: 3)
    commands.commit()
    commands.waitUntilCompleted()
    #expect(commands.status == .completed)
    #expect(build.structure.size > 0)
}
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalDeviceBuildsTriangleAccelerationStructure() async throws {
    guard let native = MTLCreateSystemDefaultDevice(), native.supportsRaytracing else { return }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let vertices = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
    let structure = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let words: [UInt32] = [0, 0, 0, Float(1).bitPattern, 0, 0, 0, Float(1).bitPattern, 0]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: words),
        .buildAccelerationStructure(destination: structure, geometry: geometry),
        .buildAccelerationStructure(destination: structure, geometry: geometry)
    ])
    _ = try await ticket.wait()
    try await device.destroyAccelerationStructure(structure)
    await device.close()
}
#endif

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalBuildsTopLevelAccelerationStructure() throws {
    guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else { return }
    let vertices: [Float] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    let buffer = vertices.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count) }!
    let commands = device.makeCommandQueue()!.makeCommandBuffer()!
    let bottom = try MetalAccelerationStructureBuild.encode(device: device, commandBuffer: commands,
        vertexBuffer: buffer, vertexOffset: 0, vertexStride: 12, vertexCount: 3)
    let top = try MetalAccelerationStructureBuild.encodeTopLevel(device: device, commandBuffer: commands,
        instances: [(.init(accelerationStructure: .init(device: 1, serial: 1),
                           transform: [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4], mask: 255, instanceID: 7), bottom)])
    let persistent = top.persistent
    #expect(top.scratch != nil && top.instanceBuffer != nil)
    #expect(persistent.scratch == nil && persistent.instanceBuffer == nil)
    #expect(persistent.dependencies.count == 1)
    #expect(persistent.dependencies[0].scratch == nil)
    #expect(persistent.dependencies[0].structure === bottom.structure)
    commands.commit()
    commands.waitUntilCompleted()
    #expect(commands.status == .completed)
    #expect(top.structure.size > 0)
}
#endif

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalDeviceBuildsAndRebuildsTopLevelAccelerationStructure() async throws {
    guard let native = MTLCreateSystemDefaultDevice(), native.supportsRaytracing else { return }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let vertices = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let instance = AccelerationStructureInstance(accelerationStructure: bottom, instanceID: 17)
    let words: [UInt32] = [0, 0, 0, Float(1).bitPattern, 0, 0, 0, Float(1).bitPattern, 0]
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: words),
        .buildAccelerationStructure(destination: bottom, geometry: geometry),
        .buildTopLevelAccelerationStructure(destination: top, instances: [instance]),
        .buildAccelerationStructure(destination: bottom, geometry: geometry),
        .buildTopLevelAccelerationStructure(destination: top, instances: [instance])
    ])
    _ = try await ticket.wait()
    try await device.destroyAccelerationStructure(bottom)
    try await device.destroyAccelerationStructure(top)
    await device.close()
}
#endif

#if canImport(Metal)
/// Exercises the Metal TLAS directly; this does not enable the public shader binding path.
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalTopLevelTraversalReadsHitMissTransformMaskAndUserID() throws {
    guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else { return }
    let source = """
    #include <metal_stdlib>
    #include <metal_raytracing>
    using namespace metal;
    using namespace metal::raytracing;
    kernel void trace(instance_acceleration_structure scene [[buffer(0)]],
                      device uint *output [[buffer(1)]], uint index [[thread_position_in_grid]]) {
        float3 origin = index == 1 ? float3(0.25f, 0.25f, 0.0f) : float3(2.25f, 3.25f, 0.0f);
        uint mask = index == 2 ? 1u : 2u;
        intersection_query<instancing, triangle_data> query;
        intersection_params params;
        query.reset(ray(origin, float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f), scene, mask, params);
        while (query.next()) {
            if (query.get_candidate_intersection_type() == intersection_type::triangle)
                query.commit_triangle_intersection();
        }
        bool hit = query.get_committed_intersection_type() == intersection_type::triangle;
        output[index * 3] = hit ? 1u : 0u;
        output[index * 3 + 1] = hit ? as_type<uint>(query.get_committed_distance()) : 0u;
        output[index * 3 + 2] = hit ? query.get_committed_user_instance_id() : 0u;
    }
    """
    let library = try device.makeLibrary(source: source, options: nil)
    let pipeline = try device.makeComputePipelineState(function: #require(library.makeFunction(name: "trace")))
    let vertices: [Float] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    let vertexBuffer = try #require(vertices.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)
    })
    let output = try #require(device.makeBuffer(length: 9 * MemoryLayout<UInt32>.stride, options: .storageModeShared))
    let commands = try #require(device.makeCommandQueue()?.makeCommandBuffer())
    let bottom = try MetalAccelerationStructureBuild.encode(device: device, commandBuffer: commands,
        vertexBuffer: vertexBuffer, vertexOffset: 0, vertexStride: 12, vertexCount: 3)
    let instance = AccelerationStructureInstance(accelerationStructure: .init(device: 1, serial: 1),
        transform: [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4], mask: 2, instanceID: 73)
    let top = try MetalAccelerationStructureBuild.encodeTopLevel(device: device, commandBuffer: commands,
        instances: [(instance, bottom)])
    let encoder = try #require(commands.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setAccelerationStructure(top.structure, bufferIndex: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.useResource(bottom.structure, usage: .read)
    encoder.dispatchThreads(MTLSize(width: 3, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 3, height: 1, depth: 1))
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    #expect(commands.status == .completed, "\(String(describing: commands.error))")
    let values = output.contents().assumingMemoryBound(to: UInt32.self)
    #expect(values[0] == 1)
    #expect(Float(bitPattern: values[1]) == 4)
    #expect(values[2] == 73)
    #expect(values[3] == 0) // ray misses the translated triangle
    #expect(values[6] == 0) // mask excludes the otherwise intersected instance
}
#endif

#if canImport(Metal)
@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalSanitizesGPUGeneratedTriangleIndices() throws {
    guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else { return }
    let vertices: [Float] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    let vertexBuffer = vertices.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count) }!
    let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void generate(device uint* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            if (i == 0) { data[0] = 0u; data[1] = 0x00020001u; data[2] = 0x0001ffffu; data[3] = 2u; }
            if (i == 1) { data[4] = 0u; data[5] = 1u; data[6] = 2u; data[7] = 0u; data[8] = 3u; data[9] = 2u; }
        }
        """, options: nil)
    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "generate")!)
    let source = device.makeBuffer(length: 40, options: .storageModePrivate)!
    let commands = device.makeCommandQueue()!.makeCommandBuffer()!
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.dispatchThreads(.init(width: 2, height: 1, depth: 1), threadsPerThreadgroup: .init(width: 2, height: 1, depth: 1))
    encoder.endEncoding()
    var builds: [MetalAccelerationStructureBuild] = []
    var outputs: [any MTLBuffer] = []
    for (format, offset) in [(IndexFormat.uint16, 2), (.uint32, 16)] {
        let binding = IndexBufferBinding(buffer: .init(device: 1, serial: 1), format: format, offset: UInt64(offset))
        let build = try MetalAccelerationStructureBuild.encode(device: device, commandBuffer: commands,
            vertexBuffer: vertexBuffer, vertexOffset: 0, vertexStride: 12, vertexCount: 3,
            indices: (source, binding, 6))
        let output = device.makeBuffer(length: 24, options: .storageModeShared)!
        let blit = commands.makeBlitCommandEncoder()!
        blit.copy(from: build.indexScratch!, sourceOffset: 0, to: output, destinationOffset: 0, size: 24)
        blit.endEncoding()
        builds.append(build)
        outputs.append(output)
    }
    commands.commit()
    commands.waitUntilCompleted()
    #expect(commands.status == .completed)
    for output in outputs {
        let words = output.contents().bindMemory(to: UInt32.self, capacity: 6)
        #expect(Array(UnsafeBufferPointer(start: words, count: 6)) == [0, 1, 2, 0, 0, 0])
    }
    #expect(builds.allSatisfy { $0.structure.size > 0 })
}
#endif
