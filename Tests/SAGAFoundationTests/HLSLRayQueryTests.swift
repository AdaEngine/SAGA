import Testing
import SAGAShaderIR
import SAGAFrontends
import SAGAValidation
@testable import SAGABackends

@Test func hlslRayQueryFailsClosedBeforeTraversalIsVerified() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @group(1) @binding(2) var scene: acceleration_structure;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(RAY_FLAG_NONE, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
        rayQueryTerminate(&query);
    }
    """))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    #expect(throws: ShaderFailure.unsupported("Compute code generation: HLSL ray queries require validated traversal semantics and compiler verification")) {
        try #require(ComputeCodeGenerator(.hlsl)).generateCompute(validated)
    }
}

@Test func hlslRayQueryIntersectionFailsClosed() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @group(0) @binding(0) var scene: acceleration_structure;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryInitialize(&query, scene, RayDesc(RAY_FLAG_NONE, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
        while rayQueryProceed(&query) {}
        output[0] = rayQueryGetCommittedIntersection(&query).kind;
    }
    """))
    let validated = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: [.rayQuery]))
    #expect(throws: ShaderFailure.self) {
        try #require(ComputeCodeGenerator(.hlsl)).generateCompute(validated)
    }
}

@Test func hlslRayQueryTerminationUsesInitializationState() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryTerminate(&query);
    }
    """))
    let entry = try #require(module.entryPoints.first)
    let function = try module.functions.value(for: entry.function)
    var writer = ShaderBodyWriter(module: module, function: function, arguments: [], storage: [], outputFormat: .hlsl)
    let source = try writer.write().joined(separator: "\n")
    #expect(source.contains("if ((saga_local_0_state & 1u) != 0u) { saga_local_0.Abort(); }"))
}

@Test func hlslRayQueryCommitOperationsCheckCandidateAndDistance() throws {
    let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
    enable wgpu_ray_query;
    @compute @workgroup_size(1) fn main() {
        var query: ray_query;
        rayQueryConfirmIntersection(&query);
        rayQueryGenerateIntersection(&query, 1.0);
    }
    """))
    let entry = try #require(module.entryPoints.first)
    let function = try module.functions.value(for: entry.function)
    var writer = ShaderBodyWriter(module: module, function: function, arguments: [], storage: [], outputFormat: .hlsl)
    let source = try writer.write().joined(separator: "\n")
    #expect(source.contains("CANDIDATE_NON_OPAQUE_TRIANGLE"))
    #expect(source.contains(".CommitNonOpaqueTriangleHit()"))
    #expect(source.contains("CANDIDATE_PROCEDURAL_PRIMITIVE"))
    #expect(source.contains(".RayTMin()"))
    #expect(source.contains(".CommittedRayT()"))
    #expect(source.contains(".CommitProceduralPrimitiveHit("))
}

@Test func hlslIntersectionTracksCandidateAndCommittedState() {
    let layout = ShaderStructure(typeName: "Intersection", constructor: "makeIntersection")
    let candidate = hlslRayIntersectionLines(query: "rq", name: "hit", type: "Intersection", layout: layout, committed: false).joined(separator: "\n")
    #expect(candidate.contains("(rq_state & 2u) != 0u && (rq_state & 4u) == 0u"))
    #expect(candidate.contains("CANDIDATE_NON_OPAQUE_TRIANGLE"))
    #expect(candidate.contains("hit.saga_member_0 = 3u"))
    #expect(candidate.contains("rq.CandidateTriangleRayT()"))
    #expect(candidate.contains("rq.CandidateInstanceContributionToHitGroupIndex()"))
    #expect(candidate.contains("rq.CandidateWorldToObject4x3()"))
    let committed = hlslRayIntersectionLines(query: "rq", name: "hit", type: "Intersection", layout: layout, committed: true).joined(separator: "\n")
    #expect(committed.contains("if ((rq_state & 4u) != 0u)"))
    #expect(committed.contains("saga_type != COMMITTED_NOTHING"))
    #expect(committed.contains("saga_type == COMMITTED_TRIANGLE_HIT"))
    #expect(committed.contains("rq.CommittedRayT()"))
    #expect(committed.contains("rq.CommittedObjectToWorld4x3()"))
}

@Test func hlslRayQueryValidatesDynamicFlagCombinations() {
    let condition = hlslRayFlagsValidation("flags")
    #expect(condition.contains("(flags & ~1023u) == 0u"))
    #expect(condition.contains("countbits(flags & 195u) <= 1u"))
    #expect(condition.contains("countbits(flags & 304u) <= 1u"))
    #expect(condition.contains("(flags & 768u) != 768u"))
}

@Test func hlslRayDescriptorMatchesNagaNaNAndExtentChecks() {
    let condition = hlslRayDescriptorValidation(fields: ["flags", "mask", "tmin", "tmax", "origin", "direction"])
    #expect(condition.contains("!any(isnan(origin))"))
    #expect(condition.contains("!any(isnan(direction))"))
    #expect(condition.contains("!isnan(tmin)"))
    #expect(condition.contains("!isnan(tmax)"))
    #expect(condition.contains("tmin >= 0.0f && tmin <= tmax"))
    #expect(!condition.contains("isfinite"))
    #expect(!condition.contains("direction !="))
}

@Test func hlslRayTerminationDoesNotChangeTraversalTracker() {
    let line = hlslRayTerminate(query: "rq", state: "state")
    #expect(line.contains("(state & 1u) != 0u"))
    #expect(line.contains("rq.Abort()"))
    #expect(!line.contains("state |="))
    #expect(!line.contains("state & 4u"))
}
