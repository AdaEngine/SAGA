// GL_EXT_ray_query lowering. The public generator remains gated until shader
// compilation and traversal behavior have been verified independently.
func glslRayQueryProceed(query: String, state: String, result: String) -> [String] {
    ["bool \(result) = false;",
     "if ((\(state) & 1u) != 0u && (\(state) & 4u) == 0u) {",
     "    \(result) = rayQueryProceedEXT(\(query));",
     "    \(state) |= \(result) ? 2u : 6u;",
     "}"]
}

func glslRayQueryConfirm(query: String, state: String) -> String {
    "if ((\(state) & 2u) != 0u && (\(state) & 4u) == 0u && rayQueryGetIntersectionTypeEXT(\(query), false) == gl_RayQueryCandidateIntersectionTriangleEXT) { rayQueryConfirmIntersectionEXT(\(query)); }"
}

func glslRayQueryTerminate(query: String, state: String) -> String {
    "if ((\(state) & 1u) != 0u && (\(state) & 4u) == 0u) { rayQueryTerminateEXT(\(query)); \(state) |= 4u; }"
}

// GLSL intersection getters are only valid for the corresponding hit kind.
func glslRayIntersectionLines(
    query: String, name: String, type: String, fields: [String], committed: Bool
) -> [String] {
    precondition(fields.count == 11)
    let state = "\(query)_state", choice = committed ? "true" : "false"
    let condition = committed ? "(\(state) & 4u) != 0u && (\(state) & 2u) != 0u" : "(\(state) & 2u) != 0u && (\(state) & 4u) == 0u"
    func field(_ index: Int) -> String { "\(name).\(fields[index])" }
    func getter(_ suffix: String) -> String { "rayQueryGetIntersection\(suffix)EXT(\(query), \(choice))" }
    let triangle = committed ? "gl_RayQueryCommittedIntersectionTriangleEXT" : "gl_RayQueryCandidateIntersectionTriangleEXT"
    let none = committed ? "gl_RayQueryCommittedIntersectionNoneEXT" : "gl_RayQueryCandidateIntersectionAABBEXT"
    var lines = ["\(type) \(name) = \(type)(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(0.0), mat4x3(0.0));", "if (\(condition)) {", "    uint saga_type = \(getter("Type"));", "    if (saga_type == \(triangle)) {", "        \(field(0)) = 1u;", "        \(field(1)) = \(getter("T"));", "        \(field(7)) = \(getter("Barycentrics"));", "        \(field(8)) = \(getter("FrontFace"));", "    }"]
    if committed {
        lines += ["    else if (saga_type != \(none)) {", "        \(field(0)) = 2u;", "        \(field(1)) = \(getter("T"));", "    }", "    if (saga_type != \(none)) {"]
    } else {
        lines += ["    else if (saga_type == \(none)) { \(field(0)) = 3u; }", "    if (saga_type == \(triangle) || saga_type == \(none)) {"]
    }
    for (index, method) in [(2, "InstanceCustomIndex"), (3, "InstanceId"), (4, "InstanceShaderBindingTableRecordOffset"), (5, "GeometryIndex"), (6, "PrimitiveIndex"), (9, "ObjectToWorld"), (10, "WorldToObject")] {
        let value = index <= 6 ? "uint(\(getter(method)))" : getter(method)
        lines.append("        \(field(index)) = \(value);")
    }
    lines += ["    }", "}"]
    return lines
}

func glslRayFlagsValid(_ flags: String) -> String {
    "bitCount(\(flags) & 195u) < 2 && bitCount(\(flags) & 304u) < 2 && bitCount(\(flags) & 768u) < 2"
}
