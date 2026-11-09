import SAGAShaderIR

extension WGSLParser {
    static let rayQueryValueBuiltins: Set<String> = ["rayQueryProceed", "rayQueryGetCommittedIntersection", "rayQueryGetCandidateIntersection", "getCommittedHitVertexPositions", "getCandidateHitVertexPositions"]
    static let rayQueryStatementBuiltins: Set<String> = ["rayQueryInitialize", "rayQueryGenerateIntersection", "rayQueryConfirmIntersection", "rayQueryTerminate"]

    func installRayQueryTypes(at token: WGSLToken) throws {
        guard rayDescriptorType == nil else { return }
        let u = try intern(.scalar(.u32)), f = try intern(.scalar(.f32)), b = try intern(.scalar(.bool))
        let v2 = try intern(.vector(size: .two, scalar: .f32))
        let v3 = try intern(.vector(size: .three, scalar: .f32))
        let m = try intern(.matrix(columns: .four, rows: .three, scalar: .f32))
        func structure(_ name: String, _ fields: [(String, WType, UInt32)], span: UInt32) throws -> WType {
            var members: [IRStructMember] = []
            for (name, type, offset) in fields {
                try node()
                members.append(.init(name: name, type: type, offset: offset))
            }
            let handle = try intern(.structure(members: members, span: span), name: name, at: token)
            structureAlignments[handle] = 16
            typeNames[wgslKey(name)] = handle
            return handle
        }
        rayDescriptorType = try structure("RayDesc", [("flags", u, 0), ("cull_mask", u, 4), ("t_min", f, 8), ("t_max", f, 12), ("origin", v3, 16), ("dir", v3, 32)], span: 48)
        rayIntersectionType = try structure("RayIntersection", [("kind", u, 0), ("t", f, 4), ("instance_custom_data", u, 8), ("instance_index", u, 12), ("sbt_record_offset", u, 16), ("geometry_index", u, 20), ("primitive_index", u, 24), ("barycentrics", v2, 32), ("front_face", b, 40), ("object_to_world", m, 48), ("world_to_object", m, 112)], span: 176)
    }

    func rayQueryConstant(_ name: String) -> UInt32? {
        switch name {
        case "RAY_FLAG_NONE": 0
        case "RAY_FLAG_FORCE_OPAQUE": 1
        case "RAY_FLAG_FORCE_NO_OPAQUE": 2
        case "RAY_FLAG_TERMINATE_ON_FIRST_HIT": 4
        case "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER": 8
        case "RAY_FLAG_CULL_BACK_FACING": 16
        case "RAY_FLAG_CULL_FRONT_FACING": 32
        case "RAY_FLAG_CULL_OPAQUE": 64
        case "RAY_FLAG_CULL_NO_OPAQUE": 128
        case "RAY_FLAG_SKIP_TRIANGLES": 256
        case "RAY_FLAG_SKIP_AABBS": 512
        case "RAY_QUERY_INTERSECTION_NONE": 0
        case "RAY_QUERY_INTERSECTION_TRIANGLE": 1
        case "RAY_QUERY_INTERSECTION_GENERATED": 2
        case "RAY_QUERY_INTERSECTION_AABB": 3
        default: nil
        }
    }

    func rayQueryPointer(at token: WGSLToken) throws -> WGSLValue {
        guard rayQueryEnabled else { throw wgslError(.unsupportedCapability, "Ray query requires enable wgpu_ray_query", token) }
        let pointer = try dereference(expression(), at: token)
        guard case .rayQuery = types[pointer.type.index].kind, try pointerSpace(pointer) == .function else {
            throw wgslError(.typeMismatch, "Expected function ray_query pointer", token)
        }
        return pointer
    }

    func rayQueryValue(at token: WGSLToken) throws -> WGSLValue {
        let query = try rayQueryPointer(at: token)
        _ = try eat(","); try expect(")")
        if token.text == "rayQueryProceed" {
            let result = try expressionNode(.rayQueryProceedResult, type: intern(.scalar(.bool)), at: token, emit: false)
            try statement(.rayQuery(query: query.handle, function: .proceed(result: result.handle)), at: token)
            return result
        }
        if token.text == "getCommittedHitVertexPositions" || token.text == "getCandidateHitVertexPositions" {
            guard rayQueryVertexReturnEnabled else {
                throw wgslError(.unsupportedCapability, "Vertex positions require enable wgpu_ray_query_vertex_return", token)
            }
            if rayVertexReturnType == nil {
                let vector = try intern(.vector(size: .three, scalar: .f32))
                rayVertexReturnType = try intern(.array(base: vector, size: .constant(3), stride: 16))
            }
            guard let type = rayVertexReturnType else { throw ShaderFailure.invalidHandle }
            return try expressionNode(.rayQueryVertexPositions(query: query.handle, committed: token.text == "getCommittedHitVertexPositions"), type: type, at: token)
        }
        guard let type = rayIntersectionType else { throw ShaderFailure.invalidHandle }
        return try expressionNode(.rayQueryGetIntersection(query: query.handle, committed: token.text == "rayQueryGetCommittedIntersection"), type: type, at: token)
    }

    func rayQueryStatement(at token: WGSLToken, end: String) throws {
        _ = try take(); try expect("(")
        let query = try rayQueryPointer(at: token)
        let operation: IRRayQueryFunction
        switch token.text {
        case "rayQueryInitialize":
            try expect(",")
            let acceleration = try loaded(expression(), at: token)
            guard case .accelerationStructure = types[acceleration.type.index].kind, let descriptor = rayDescriptorType else {
                throw wgslError(.typeMismatch, "Expected acceleration structure", token)
            }
            try expect(",")
            let desc = try coerce(expression(), to: descriptor, at: token)
            operation = .initialize(accelerationStructure: acceleration.handle, descriptor: desc.handle)
        case "rayQueryGenerateIntersection":
            try expect(",")
            let hit = try coerce(expression(), to: intern(.scalar(.f32)), at: token)
            operation = .generateIntersection(hitT: hit.handle)
        case "rayQueryConfirmIntersection": operation = .confirmIntersection
        default: operation = .terminate
        }
        _ = try eat(","); try expect(")"); try expect(end)
        try statement(.rayQuery(query: query.handle, function: operation), at: token)
    }
}
