import Testing
import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@testable import SAGABackends

@Test func glslRayQueryFailsClosedBeforeTraversalIsVerified() throws {
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
    #expect(throws: ShaderFailure.unsupported("Compute code generation: GLSL ray queries require validated traversal semantics and compiler verification")) {
        try #require(ComputeCodeGenerator(.glsl)).generateCompute(validated)
    }
}

@Test func glslRayQueryHelpersUseExtensionOperations() {
    #expect(glslRayQueryProceed(query: "q", state: "s", result: "r").contains { $0.contains("rayQueryProceedEXT(q)") })
    #expect(glslRayQueryConfirm(query: "q", state: "s").contains("rayQueryConfirmIntersectionEXT(q)"))
    #expect(glslRayQueryTerminate(query: "q", state: "s").contains("rayQueryTerminateEXT(q)"))
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
    #expect(condition.contains("flags & 195u"))
    #expect(condition.contains("flags & 304u"))
    #expect(condition.contains("flags & 768u"))
    #expect(!condition.contains("1015u"))
}
