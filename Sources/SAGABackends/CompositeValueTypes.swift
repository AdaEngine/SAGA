import SAGAShaderIR

extension IRTypeKind {
    var isShaderCompositeValue: Bool {
        switch self {
        case .array(_, .constant, _), .structure: return true
        default: return false
        }
    }
}

extension ShaderOutputFormat {
    var wrapsShaderValueArrays: Bool { self == .hlsl || self == .glsl }

    func valueArrayIndex(_ index: String) -> String {
        let field = wrapsShaderValueArrays ? ".elements" : ""
        return "\(field)[\(index)]"
    }
}

func compositeValueDeclarations(
    module: IRModule, format: ShaderOutputFormat, roots: Set<IRHandle<IRType>>, constructed: Set<IRHandle<IRType>>,
    structures: inout [IRHandle<IRType>: ShaderStructure], budget: inout ShaderBudget
) throws -> [String] {
    var pending = Array(roots)
    var types: Set<IRHandle<IRType>> = []
    while let handle = pending.popLast() {
        try budget.consume(.work)
        guard structures[handle] == nil, types.insert(handle).inserted else { continue }
        switch try module.types.value(for: handle).kind {
        case .structure(let members, _):
            try budget.consume(.work, amount: members.count)
            pending.append(contentsOf: members.map(\.type))
        case .array(let element, .constant, _):
            pending.append(element)
        default:
            types.remove(handle)
        }
    }
    var lines: [String] = []
    // Validation ensures child type handles precede their containers.
    for handle in types.sorted(by: { $0.index < $1.index }) {
        try budget.consume(.work)
        let kind = try module.types.value(for: handle).kind
        switch kind {
        case .structure(let members, _):
            let name = "SagaValue\(handle.index)"
            let structure = ShaderStructure(typeName: name,
                constructor: format == .hlsl ? "saga_make_value_\(handle.index)" : name)
            let memberTypes = try members.map {
                try budget.consume(.work)
                return try compositeMemberType($0.type, module: module, format: format, structures: structures)
            }
            let declaration = structure.declarations(memberTypes: memberTypes, in: format)
            for line in declaration { try budget.consume(.sourceBytes, amount: line.utf8.count + 1) }
            lines += declaration
            structures[handle] = structure
        case .array(let element, .constant(let count), _):
            let type = try compositeMemberType(element, module: module, format: format, structures: structures)
            let structure: ShaderStructure
            if format.wrapsShaderValueArrays {
                let name = "SagaArray\(handle.index)"
                let constructor = "saga_make_array_\(handle.index)"
                let declaration = "struct \(name) { \(type) elements[\(count)]; };"
                try budget.consume(.sourceBytes, amount: declaration.utf8.count + 1)
                lines.append(declaration)
                if constructed.contains(handle) {
                    try appendArrayConstructor(name: name, constructor: constructor, element: type, count: count,
                                               to: &lines, budget: &budget)
                }
                if format == .glsl {
                    let zero = [
                        "\(name) saga_zero_array_\(handle.index)(\(type) zero) {",
                        "    \(name) result;",
                        "    for (uint i = 0u; i < \(count)u; ++i) { result.elements[i] = zero; }",
                        "    return result;", "}",
                    ]
                    for line in zero { try budget.consume(.sourceBytes, amount: line.utf8.count + 1) }
                    lines += zero
                }
                structure = .init(typeName: name, constructor: constructor)
            } else {
                let name = "array<\(type), \(count)>"
                try budget.consume(.sourceBytes, amount: name.utf8.count)
                structure = .init(typeName: name, constructor: name)
            }
            structures[handle] = structure
        default: throw ShaderFailure.invalidHandle
        }
    }
    return lines
}

private func compositeMemberType(
    _ type: IRHandle<IRType>, module: IRModule, format: ShaderOutputFormat,
    structures: [IRHandle<IRType>: ShaderStructure]
) throws -> String {
    switch try module.types.value(for: type).kind {
    case .atomic(.i32):
        return format == .msl ? "atomic_int" : format == .wgsl ? "atomic<i32>" : "int"
    case .atomic(.u32), .atomic(.f32):
        return format == .msl ? "atomic_uint" : format == .wgsl ? "atomic<u32>" : "uint"
    case .atomic(let scalar) where scalar == .i64 || scalar == .u64:
        guard format != .msl else { throw ShaderFailure.unsupported("Metal does not support 64-bit workgroup atomics") }
        let lane = try shaderVectorType(1, scalar: scalar, in: format)
        return format == .wgsl ? "atomic<\(lane)>" : lane
    default:
        return try shaderValueType(type, module: module, format: format, structures: structures)
    }
}

private func appendArrayConstructor(
    name: String, constructor: String, element: String, count: UInt32,
    to lines: inout [String], budget: inout ShaderBudget
) throws {
    // A wrapper gives nested arrays value semantics in both HLSL and GLSL.
    let members = Int(count)
    try budget.consume(.work, amount: members)
    let (bytes, overflow) = members.multipliedReportingOverflow(by: element.utf8.count + 48)
    guard !overflow else { throw ShaderFailure.budgetExceeded(.sourceBytes) }
    try budget.consume(.sourceBytes, amount: bytes)
    let parameters = (0..<members).map { "\(element) arg\($0)" }.joined(separator: ", ")
    lines += ["\(name) \(constructor)(\(parameters)) {", "    \(name) result;"]
    lines += (0..<members).map { "    result.elements[\($0)] = arg\($0);" }
    lines += ["    return result;", "}"]
}
