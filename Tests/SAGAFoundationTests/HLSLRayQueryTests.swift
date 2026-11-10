import Testing
import SAGAShaderIR
import SAGAFrontends
import SAGACompiler
import SAGAValidation
@testable import SAGABackends

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
    #expect(!condition.contains("~1023u"))
    #expect(condition.contains("countbits(flags & 195u) <= 1u"))
    #expect(condition.contains("countbits(flags & 304u) <= 1u"))
    #expect(condition.contains("(flags & 768u) != 768u"))
}

@Test func hlslRayDescriptorMatchesNagaNaNAndExtentChecks() {
    let condition = hlslRayDescriptorValidation(fields: ["flags", "mask", "tmin", "tmax", "origin", "direction"])
    for value in ["origin", "direction", "tmin", "tmax"] {
        #expect(condition.contains("asuint(\(value))"))
    }
    #expect(condition.contains("0x7f800000u"))
    #expect(condition.contains("0x007fffffu"))
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

@Test func dynamicPublicRayQueryReflectsHLSLBindings() throws {
 let source = """
 enable wgpu_ray_query;
 @group(1) @binding(2) var scene: acceleration_structure;
 @group(1) @binding(3) var<storage, read_write> result: array<u32>;
 @compute @workgroup_size(1) fn main() {
  var q: ray_query;
  rayQueryInitialize(&q, scene, RayDesc(result[0], 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
  while rayQueryProceed(&q) { let c = rayQueryGetCandidateIntersection(&q); result[1] = c.kind; rayQueryConfirmIntersection(&q); rayQueryGenerateIntersection(&q, 1.0); }
  result[2] = rayQueryGetCommittedIntersection(&q).kind;
  rayQueryTerminate(&q);
 }
 """
 let result = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl)
 #expect(result.hlslProfile == "cs_6_5")
 #expect(result.accelerationStructures.first?.binding == .init(group: 1, binding: 2))
 #expect(result.storageBindings.first?.binding == .init(group: 1, binding: 3))
 guard case .text(.hlsl, let code) = result.artifact else { throw ShaderFailure.invalidHandle }
 #expect(code.contains("register(t2, space1)"))
 #expect(code.contains("register(u3, space1)"))

}
@Test func publicHLSLRejectsRayVertexReturn() {
 for getter in ["getCandidateHitVertexPositions", "getCommittedHitVertexPositions"] {
 let source = """
 enable wgpu_ray_query, wgpu_ray_query_vertex_return;
 @group(0) @binding(0) var scene: acceleration_structure<vertex_return>;
 @compute @workgroup_size(1) fn main() {
  var q: ray_query<vertex_return>;
  rayQueryInitialize(&q, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
  let positions = \(getter)(&q);
 }
 """
 #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl) }
 }
}

@Test func publicHLSLRejectsRayVertexReturnWithoutPositionGetter() {
 let source = """
 enable wgpu_ray_query, wgpu_ray_query_vertex_return;
 @group(0) @binding(0) var scene: acceleration_structure<vertex_return>;
 @compute @workgroup_size(1) fn main() {
  var q: ray_query<vertex_return>;
  rayQueryInitialize(&q, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
  while rayQueryProceed(&q) {}
 }
 """
 #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl) }
}

@Test func hlslRayDescriptorChecksNaNByBits() throws {
 let source = """
 enable wgpu_ray_query;
 @group(0) @binding(0) var scene: acceleration_structure;
 @compute @workgroup_size(1) fn main() {
  var q: ray_query;
  rayQueryInitialize(&q, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
  while rayQueryProceed(&q) {}
 }
 """
 let result = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl)
 guard case .text(.hlsl, let code) = result.artifact else { throw ShaderFailure.invalidHandle }
 #expect(code.contains("asuint("))
 #expect(!code.contains("isnan("))
}

@Test func hlslRayQueryAcceptsAccelerationStructureHelperArgument() throws {
    let source = """
    enable wgpu_ray_query;
    @group(0) @binding(0) var scene: acceleration_structure;
    fn trace(geometry: acceleration_structure) -> u32 {
        var q: ray_query;
        rayQueryInitialize(&q, geometry, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
        while rayQueryProceed(&q) {}
        return rayQueryGetCommittedIntersection(&q).kind;
    }
    @group(0) @binding(1) var<storage, read_write> result: array<u32>;
    @compute @workgroup_size(1) fn main() { result[0] = trace(scene); }
    """
    let result = try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .hlsl)
    guard case .text(.hlsl, let code) = result.artifact else { throw ShaderFailure.invalidHandle }
    #expect(code.contains("RaytracingAccelerationStructure"))
    #expect(code.contains("TraceRayInline"))
}
