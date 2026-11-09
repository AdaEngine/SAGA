import Testing
import SAGACompiler
import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
import SAGABackends

@Test func metalRayQueryCompilesAndReflectsResources() throws {
    let source = """
    enable wgpu_ray_query;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 1>;
    @group(0) @binding(1) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(RAY_FLAG_NONE, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
        output[0] = rayQueryGetCommittedIntersection(&query).kind;
    }
    """
    let artifact = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
    #expect(artifact.accelerationStructures.count == 1)
    #expect(artifact.accelerationStructures[0].binding == .init(group: 0, binding: 1))
    #expect(artifact.accelerationStructures[0].flatBindingIndex != artifact.storageBindings[0].flatBindingIndex)
}

@Test func metalRayQueryMapsDynamicFlags() throws {
    let source = """
    enable wgpu_ray_query;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 1>;
    @group(0) @binding(1) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(output[0] & 199u, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
        output[0] = rayQueryGetCommittedIntersection(&query).kind;
    }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl(source))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    let generator = try #require(ComputeCodeGenerator(.msl))
    let generated = try generator.generateCompute(validated)
    guard case .text(.msl, let code) = generated.artifact else { throw ShaderFailure.invalidHandle }
    #expect(code.contains("opacity_cull_mode::opaque"))
    #expect(code.contains("opacity_cull_mode::non_opaque"))
    #expect(code.contains("forced_opacity::opaque"))
    #expect(code.contains("forced_opacity::non_opaque"))
    #expect(code.contains("accept_any_intersection"))
}

@Test func metalRayQueryValidatesUnboundedDynamicFlagsAtRuntime() throws {
    let source = """
    enable wgpu_ray_query;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 1>;
    @group(0) @binding(1) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(output[0], 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
        output[0] = rayQueryGetCommittedIntersection(&query).kind;
    }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl(source))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    let generator = try #require(ComputeCodeGenerator(.msl))
    let shader = try generator.generateCompute(validated)
    guard case .text(.msl, let source) = shader.artifact else { throw ShaderFailure.invalidHandle }
    #expect(source.contains("~1023u"))
}
