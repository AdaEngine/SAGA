import SAGAFrontends
import SAGAShaderIR
import SAGAValidation
import Testing

@Suite struct WGSLRayQueryFrontendTests {
    @Test func lowersRayQueryOperations() throws {
        let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("""
        enable wgpu_ray_query;
        @group(0) @binding(0) var scene: acceleration_structure;
        @compute @workgroup_size(1) fn main() {
            var query: ray_query;
            rayQueryInitialize(&query, scene, RayDesc(RAY_FLAG_NONE, 255u, 0.1, 10.0, vec3<f32>(0.0), vec3<f32>(1.0)));
            while rayQueryProceed(&query) {}
            let hit = rayQueryGetCommittedIntersection(&query);
            rayQueryTerminate(&query);
        }
        """))
        #expect(module.specialTypes.rayDescriptor != nil)
        #expect(module.specialTypes.rayIntersection != nil)
        #expect(module.functions.values.contains { $0.statements.values.contains { if case .rayQuery = $0.kind { true } else { false } } })
    }
    @Test func requiresCapabilityAndDirective() {
        #expect(throws: (any Error).self) { try WGSLFrontend(options: .init(capabilities: [])).parse(.wgsl("enable wgpu_ray_query;")) }
        #expect(throws: (any Error).self) { try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl("alias Q = ray_query;")) }
    }

    @Test func loopLocalRayQueryResetsOnEveryIteration() throws {
        let source = """
        enable wgpu_ray_query;
        @compute @workgroup_size(1) fn main() {
            var outer: ray_query;
            for (var i = 0u; i < 3u; i++) {
                var query: ray_query;
                if i == 1u { var nested: ray_query; }
                _ = rayQueryProceed(&query);
            }
        }
        """
        let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl(source))
        let function = try #require(module.functions.values.first)
        let resets = function.statements.values.compactMap { statement -> IRHandle<IRExpression>? in
            if case .rayQuery(let query, .begin) = statement.kind { return query }
            return nil
        }
        #expect(resets.count == 2)
        let resetNames = try resets.map { handle -> String? in
            guard case .localVariable(let local) = try function.expressions.value(for: handle).kind else { return nil }
            return try function.localVariables.value(for: local).name
        }
        #expect(Set(resetNames.compactMap { $0 }) == Set(["query", "nested"]))
        try ShaderSemanticValidator(capabilities: [.rayQuery]).validate(module)
    }

    @Test func rayQueryResultsMayBeDiscardedAsCallStatements() throws {
        let source = """
        enable wgpu_ray_query;
        @compute @workgroup_size(1) fn main() {
            var query: ray_query;
            rayQueryProceed(&query);
            rayQueryGetCandidateIntersection(&query);
            rayQueryGetCommittedIntersection(&query);
        }
        """
        let module = try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl(source))
        try ShaderSemanticValidator(capabilities: [.rayQuery]).validate(module)
        let function = try #require(module.functions.values.first)
        #expect(function.statements.values.filter { if case .rayQuery(_, .proceed) = $0.kind { true } else { false } }.count == 1)
    }

    @Test func vertexReturnRequiresItsExtensionAndCapability() throws {
        func shader(vertexExtension: String = ", wgpu_ray_query_vertex_return", acceleration: String = "acceleration_structure<vertex_return>", query: String = "ray_query<vertex_return>") -> String { """
        enable wgpu_ray_query\(vertexExtension);
        @group(0) @binding(0) var scene: \(acceleration);
        @compute @workgroup_size(1) fn main() {
            var query: \(query);
            rayQueryInitialize(&query, scene, RayDesc(0u, 255u, 0.1, 10.0, vec3f(0.0), vec3f(1.0)));
            let committed = getCommittedHitVertexPositions(&query);
            let candidate = getCandidateHitVertexPositions(&query);
            let first = committed[0] + candidate[1];
        }
        """ }
        let source = shader()
        let capabilities: ShaderCapabilities = [.rayQuery, .rayHitVertexPosition]
        let module = try WGSLFrontend(options: .init(capabilities: capabilities)).parse(.wgsl(source))
        let result = try #require(module.specialTypes.rayVertexReturn)
        guard case .array(let base, .constant(3), 16) = try module.types.value(for: result).kind else {
            Issue.record("Expected three aligned vertex positions")
            return
        }
        #expect(try module.types.value(for: base).kind == .vector(size: .three, scalar: .f32))
        _ = try ValidatedModule.checking(module, with: ShaderSemanticValidator(capabilities: capabilities))
        #expect(throws: (any Error).self) {
            try WGSLFrontend(options: .init(capabilities: [.rayQuery])).parse(.wgsl(source))
        }
        #expect(throws: (any Error).self) {
            try WGSLFrontend(options: .init(capabilities: capabilities)).parse(.wgsl(shader(vertexExtension: "")))
        }
        for invalid in [
            shader(acceleration: "acceleration_structure"),
            shader(query: "ray_query")
        ] {
            let invalidModule = try WGSLFrontend(options: .init(capabilities: capabilities)).parse(.wgsl(invalid))
            #expect(throws: ShaderFailure.self) {
                try ValidatedModule.checking(invalidModule, with: ShaderSemanticValidator(capabilities: capabilities))
            }
        }
    }
}
