// The result layout follows Naga's RayIntersection special type.
func hlslRayIntersectionLines(
    query: String, name: String, type: String, layout: ShaderStructure, committed: Bool
) -> [String] {
    let state = "\(query)_state"
    let prefix = committed ? "Committed" : "Candidate"
    let condition = committed ? "(\(state) & 4u) != 0u" : "(\(state) & 2u) != 0u && (\(state) & 4u) == 0u"
    func field(_ index: Int) -> String { "\(name).\(layout.memberName(index))" }
    var lines = ["\(type) \(name) = (\(type))0;", "if (\(condition)) {"]
    if committed {
        lines += ["    COMMITTED_STATUS saga_type = \(query).CommittedStatus();",
                  "    \(field(0)) = uint(saga_type);",
                  "    if (saga_type != COMMITTED_NOTHING) {",
                  "        \(field(1)) = \(query).CommittedRayT();",
                  "        if (saga_type == COMMITTED_TRIANGLE_HIT) {",
                  "            \(field(7)) = \(query).CommittedTriangleBarycentrics();",
                  "            \(field(8)) = \(query).CommittedTriangleFrontFace();",
                  "        }"]
    } else {
        lines += ["    CANDIDATE_TYPE saga_type = \(query).CandidateType();",
                  "    if (saga_type == CANDIDATE_NON_OPAQUE_TRIANGLE) {",
                  "        \(field(0)) = 1u;",
                  "        \(field(1)) = \(query).CandidateTriangleRayT();",
                  "        \(field(7)) = \(query).CandidateTriangleBarycentrics();",
                  "        \(field(8)) = \(query).CandidateTriangleFrontFace();",
                  "    } else { \(field(0)) = 3u; }"]
    }
    let indent = committed ? "        " : "    "
    for (index, method) in [(2, "InstanceID"), (3, "InstanceIndex"),
                            (4, "InstanceContributionToHitGroupIndex"), (5, "GeometryIndex"),
                            (6, "PrimitiveIndex"), (9, "ObjectToWorld4x3"), (10, "WorldToObject4x3")] {
        lines.append("\(indent)\(field(index)) = \(query).\(prefix)\(method)();")
    }
    if committed { lines.append("    }") }
    lines.append("}")
    return lines
}

// Reject unsupported bits and mutually exclusive DXR ray flags before traversal.
func hlslRayFlagsValidation(_ flags: String) -> String {
    "(\(flags) & ~1023u) == 0u && countbits(\(flags) & 195u) <= 1u && countbits(\(flags) & 304u) <= 1u && (\(flags) & 768u) != 768u"
}

func hlslRayDescriptorValidation(fields: [String]) -> String {
    let tmin = fields[2], tmax = fields[3], origin = fields[4], direction = fields[5]
    return "!isnan(\(tmin)) && !isnan(\(tmax)) && !any(isnan(\(origin))) && !any(isnan(\(direction))) && \(tmin) >= 0.0f && \(tmin) <= \(tmax)"
}

func hlslRayTerminate(query: String, state: String) -> String {
    "if ((\(state) & 1u) != 0u) { \(query).Abort(); }"
}
