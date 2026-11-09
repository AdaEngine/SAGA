import Testing
import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import SAGABackends

@Test func metalRayQueryMapsFaceAndGeometryFlags() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 1>;
    @group(0) @binding(1) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(output[0] & 1015u, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
        output[0] = rayQueryGetCommittedIntersection(&query).kind;
    }
    """))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    let generated = try #require(ComputeCodeGenerator(.msl)).generateCompute(validated)
    guard case .text(.msl, let code) = generated.artifact else { throw ShaderFailure.invalidHandle }
    #expect(code.contains("triangle_cull_mode::front"))
    #expect(code.contains("triangle_cull_mode::back"))
    #expect(code.contains("geometry_cull_mode::triangle"))
    #expect(code.contains("geometry_cull_mode::bounding_box"))
    #expect(code.contains("set_triangle_front_facing_winding"))
    #expect(code.contains("popcount"))
}

#if canImport(Metal)
import Metal
import SAGATypes
import SAGACore
import SAGACompiler
@testable import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true))
@MainActor func metalRayQueryFaceAndGeometryFlagsOnGPU() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        enable wgpu_ray_query;
        @group(0) @binding(0) var<storage, read_write> output: array<u32, 11>;
        @group(0) @binding(1) var scene: acceleration_structure;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
            var query: ray_query;
            let flags = select(select(select(select(select(select(select(select(select(select(0u, 16u, id.x == 1u), 32u, id.x == 2u), 256u, id.x == 3u), 512u, id.x == 4u), 48u, id.x == 5u), 768u, id.x == 6u), 3u, id.x == 7u), 16u, id.x == 8u), 32u, id.x == 9u), 8u, id.x == 10u) & 1023u;
            let origin = select(vec3f(0.25, 0.25, 0.0), vec3f(0.25, 0.25, 8.0), id.x == 8u || id.x == 9u);
            let direction = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0), id.x == 8u || id.x == 9u);
            rayQueryInitialize(&query, scene, RayDesc(flags, 255u, 0.1, 10.0, origin, direction));
            while rayQueryProceed(&query) { rayQueryConfirmIntersection(&query); }
            output[id.x] = rayQueryGetCommittedIntersection(&query).kind;
        }
        """), to: .msl)
    let pipeline = try await device.createComputePipeline(shader)
    let vertices = try await device.createBuffer(.init(size: 36, usage: [.vertex, .copyDestination]))
    let output = try await device.createBuffer(.init(size: 44, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 44, usage: [.copyDestination, .mapRead]))
    let bottom = try await device.createAccelerationStructure(.init(maxTriangleCount: 1))
    let top = try await device.createAccelerationStructure(.init(maxInstanceCount: 1))
    let geometry = AccelerationStructureTriangleGeometry(vertexBuffer: vertices, vertexStride: 12, vertexCount: 3)
    let ticket = try await device.submitCommands([
        .write(buffer: vertices, offset: 0, words: [0, 0, Float(4).bitPattern, Float(1).bitPattern, 0, Float(4).bitPattern, 0, Float(1).bitPattern, Float(4).bitPattern]),
        .buildAccelerationStructure(destination: bottom, geometry: geometry),
        .buildTopLevelAccelerationStructure(destination: top, instances: [.init(accelerationStructure: bottom)]),
        .dispatchCompute(pipeline: pipeline, bindings: [.init(group: 0, binding: 0): output], workgroups: .init(x: 11), accelerationStructures: [.init(group: 0, binding: 1): top]),
        .copy(source: output, destination: readback, byteCount: 44),
        .read(buffer: readback, byteCount: 44)
    ])
    let result = try await ticket.wait()
    #expect(result == [[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]])
    await device.close()
}
#endif

@Test func metalRayQueryValidatesUnsupportedFlagBitAtRuntime() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @group(0) @binding(0) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(1024u, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
    }
    """))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    let generator = try #require(ComputeCodeGenerator(.msl))
    let shader = try generator.generateCompute(validated)
    guard case .text(.msl, let source) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(source.contains("~1023u"))
}
