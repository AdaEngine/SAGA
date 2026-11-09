import SAGAShaderIR

func workgroupCompositeByteSize(_ type: IRHandle<IRType>, in module: IRModule) throws -> UInt64? {
    switch try module.types.value(for: type).kind {
    case .structure(_, let span):
        return UInt64(span)
    case .array(let element, .constant(let count), let stride):
        switch try module.types.value(for: element).kind {
        case .structure, .array, .scalar(.bool), .vector(_, .bool):
            return UInt64(count) * UInt64(stride)
        default: return nil
        }
    case .scalar(.bool): return 4
    case .vector(let size, .bool): return UInt64(size.rawValue) * 4
    default: return nil
    }
}

func initializeCompositeWorkgroup(
    _ type: IRHandle<IRType>, reference: String, indexPrefix: String, lane: String, threads: UInt32, module: IRModule,
    format: ShaderOutputFormat, structures: [IRHandle<IRType>: ShaderStructure], budget: inout ShaderBudget
) throws -> [String] {
    var lines: [String] = []
    var nextIndex = 0

    func append(_ line: String, indent: String) throws {
        try budget.consume(.sourceBytes, amount: indent.utf8.count + line.utf8.count + 1)
        lines.append(indent + line)
    }

    func initialize(
        _ type: IRHandle<IRType>, at reference: String, indent: String, parallel: Bool
    ) throws {
        try budget.consume(.work)
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let kind = try module.types.value(for: type).kind
        switch kind {
        case .array(let element, .constant(let count), _):
            let index = "\(indexPrefix)_\(nextIndex)"
            nextIndex += 1
            let start = parallel ? lane : "0u"
            let stride = parallel ? threads : 1
            try append("for (uint \(index) = \(start); \(index) < \(count)u; \(index) += \(stride)u) {", indent: indent)
            let path = format.valueArrayIndex(index)
            try initialize(element, at: reference + path, indent: indent + "    ", parallel: false)
            try append("}", indent: indent)
        case .structure(let members, _):
            guard let structure = structures[type] else { throw ShaderFailure.invalidHandle }
            for (index, member) in members.enumerated() {
                try initialize(member.type, at: reference + "." + structure.memberName(index),
                               indent: indent, parallel: parallel)
            }
        default:
            let store: String
            if case .atomic(let scalar) = kind {
                let zero: String = switch scalar {
                case .i64: signed64Literal(0, in: format)
                case .u64: unsigned64Literal(0, in: format)
                case .i32: "0"
                default: "0u"
                }
                store = format == .msl ? "atomic_store_explicit(&\(reference), \(zero), memory_order_relaxed);" :
                    "\(reference) = \(zero);"
            } else {
                let name = try shaderValueType(type, module: module, format: format, structures: structures)
                let zero = format == .hlsl ? "((\(name))0)" : "\(name)(0)"
                store = "\(reference) = \(zero);"
            }
            try append(parallel ? "if (\(lane) == 0u) { \(store) }" : store, indent: indent)
        }
    }

    try initialize(type, at: reference, indent: "    ", parallel: true)
    return lines
}
