import Testing
import SAGAShaderIR
import SAGAFrontends
import SAGACompiler
import SAGAValidation
@testable import SAGABackends

@Test func glslRayQueryEmitsTraversalAndPrimitiveCullingCapability() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @group(0) @binding(0) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
        while rayQueryProceed(&query) { rayQueryConfirmIntersection(&query); }
        rayQueryTerminate(&query);
    }
    """))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    let artifact = try #require(ComputeCodeGenerator(.glsl)).generateCompute(validated).artifact
    guard case .text(.glsl, let source) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("#version 460 core"))
    #expect(source.contains("layout(primitive_culling);"))
    #expect(source.contains("rayQueryInitializeEXT"))
    #expect(source.contains("rayQueryProceedEXT"))
    #expect(source.contains("rayQueryConfirmIntersectionEXT"))
    #expect(source.contains("rayQueryTerminateEXT"))
}

@Test func glslRayQueryHelpersUseExtensionOperations() {
    #expect(glslRayQueryProceed(query: "q", state: "s", result: "r").contains { $0.contains("rayQueryProceedEXT(q)") })
    #expect(glslRayQueryConfirm(query: "q", state: "s").contains("rayQueryConfirmIntersectionEXT(q)"))
    #expect(glslRayQueryTerminate(query: "q", state: "s").contains("rayQueryTerminateEXT(q)"))
    #expect(glslRayQueryTerminate(query: "q", state: "s").contains("(s & 2u) != 0u"))
}

@Test func glslRayIntersectionUsesStateAndTypedAccessors() {
    let fields = (0..<11).map { "f\($0)" }
    let candidate = glslRayIntersectionLines(query: "q", name: "hit", type: "Hit", fields: fields, committed: false).joined(separator: "\n")
    #expect(candidate.contains("(q_state & 2u) != 0u"))
    #expect(candidate.contains("gl_RayQueryCandidateIntersectionTriangleEXT"))
    #expect(candidate.contains("rayQueryGetIntersectionBarycentricsEXT(q, false)"))
    #expect(candidate.contains("rayQueryGetIntersectionObjectToWorldEXT(q, false)"))
    let committed = glslRayIntersectionLines(query: "q", name: "hit", type: "Hit", fields: fields, committed: true).joined(separator: "\n")
    #expect(committed.contains("(q_state & 4u) != 0u && (q_state & 2u) != 0u"))
    #expect(committed.contains("gl_RayQueryCommittedIntersectionNoneEXT"))
    #expect(committed.contains("rayQueryGetIntersectionTEXT(q, true)"))
}

@Test func glslRayQueryGenerateChecksHitRange() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryGenerateIntersection(&query, 1.0);
    }
    """))
    let function = try module.functions.value(for: #require(module.entryPoints.first).function)
    var writer = ShaderBodyWriter(module: module, function: function, arguments: [], storage: [], outputFormat: .glsl)
    let source = try writer.write().joined(separator: "\n")
    #expect(source.contains("rayQueryGetRayTMinEXT"))
    #expect(source.contains("rayQueryGetIntersectionTEXT"))
    #expect(source.contains("saga_hit_t <= saga_hit_limit"))
}

@Test func glslRayQueryStateRetainsProceedCompletion() {
    let source = glslRayQueryProceed(query: "q", state: "s", result: "r").joined(separator: "\n")
    #expect(source.contains("s |= r ? 2u : 6u"))
}

@Test func glslRayFlagsMatchNagaValidation() {
    let condition = glslRayFlagsValid("flags")
    #expect(!condition.contains("~1023u"))
    #expect(condition.contains("flags & 195u"))
    #expect(condition.contains("flags & 304u"))
    #expect(condition.contains("flags & 768u"))
    #expect(!condition.contains("1015u"))
}

@Test func glslRayQueryEmitsCandidateAndCommittedFields() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @group(0) @binding(0) var scene: acceleration_structure;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    @compute @workgroup_size(1) fn main() {
        var q: ray_query;
        rayQueryInitialize(&q, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
        while rayQueryProceed(&q) {
            let c = rayQueryGetCandidateIntersection(&q);
            output[0] = c.kind + c.instance_custom_data + c.geometry_index + c.primitive_index;
            rayQueryConfirmIntersection(&q);
            rayQueryGenerateIntersection(&q, 1.0);
        }
        let c = rayQueryGetCommittedIntersection(&q);
        output[0] = c.kind + c.instance_custom_data + c.geometry_index + c.primitive_index;
    }
    """))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    let artifact = try #require(ComputeCodeGenerator(.glsl)).generateCompute(validated).artifact
    guard case .text(.glsl, let source) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(source.contains("rayQueryGetIntersectionBarycentricsEXT"))
    #expect(source.contains("rayQueryGetIntersectionObjectToWorldEXT"))
    #expect(source.contains("rayQueryGenerateIntersectionEXT"))
    #expect(source.contains("rayQueryGetRayTMinEXT"))
}

@Test func glslRayQueryVertexPositionsUsesPositionFetch() throws {
    let source = """
    enable wgpu_ray_query, wgpu_ray_query_vertex_return;
    @group(0) @binding(0) var scene: acceleration_structure<vertex_return>;
    @group(0) @binding(1) var<storage, read_write> result: array<vec4f>;
    fn first(p: array<vec3f, 3>) -> vec3f { return p[0] + p[1] + p[2]; }
    @compute @workgroup_size(1) fn main() {
        var q: ray_query<vertex_return>;
        rayQueryInitialize(&q, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
        while rayQueryProceed(&q) {
            let p = getCandidateHitVertexPositions(&q);
            result[0] = vec4f(first(p), 1.0);
            rayQueryConfirmIntersection(&q);
        }
        let p = getCommittedHitVertexPositions(&q);
        result[1] = vec4f(first(p), 1.0);
    }
    """
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery, .rayHitVertexPosition])).parse(.wgsl(source))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery, .rayHitVertexPosition]))
    let artifact = try #require(ComputeCodeGenerator(.glsl)).generateCompute(validated).artifact
    guard case .text(.glsl, let code) = artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(code.contains("#extension GL_EXT_ray_tracing_position_fetch : require"))
    #expect(code.contains("rayQueryGetIntersectionTriangleVertexPositionsEXT"))
    #expect(code.contains(".elements);"))
    #expect(code.contains("gl_RayQueryCandidateIntersectionTriangleEXT"))
    #expect(code.contains("gl_RayQueryCommittedIntersectionTriangleEXT"))
    let publicResult = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .glsl)
    guard case .text(.glsl, let publicCode) = publicResult.artifact else { throw ShaderFailure.incompatibleArtifact }
    #expect(publicCode.contains("rayQueryGetIntersectionTriangleVertexPositionsEXT"))
}
