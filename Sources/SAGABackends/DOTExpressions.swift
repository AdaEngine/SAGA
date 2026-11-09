import SAGAShaderIR

extension DOTGraphWriter {
    mutating func writeFunctionExpressions(_ function: IRFunction, functionIndex: Int) throws {
        let functionID = "f\(functionIndex)"
        let names = Dictionary(grouping: function.namedExpressions, by: \.expression.index)
        for (index, expression) in function.expressions.values.enumerated() {
            try writeExpression(expression, index: index, prefix: "\(functionID)_e",
                                functionID: functionID, names: names[index] ?? [])
        }
    }

    mutating func writeGlobalExpressions(_ expressions: IRTable<IRExpression>) throws {
        for (index, expression) in expressions.values.enumerated() {
            try writeExpression(expression, index: index, prefix: "ge", functionID: nil, names: [])
        }
    }

    private mutating func writeExpression(
        _ expression: IRExpression, index: Int, prefix: String, functionID: String?, names: [IRNamedExpression]
    ) throws {
        let id = "\(prefix)\(index)"
        let label = ([expression.kind.dotLabel] + names.map(\.name)).joined(separator: "\n")
        try node(id, label: label)
        try edge(id, "t\(expression.type.index)", label: "type")
        if let typeOperand = expression.kind.dotTypeOperand, typeOperand != expression.type {
            try edge(id, "t\(typeOperand.index)", label: "type operand")
        }
        for (operandIndex, operand) in expression.kind.dotOperands.enumerated() {
            try edge(id, "\(prefix)\(operand.index)", label: "operand \(operandIndex)")
        }
        if let reference = expression.kind.dotReference(in: functionID) {
            try edge(id, reference, label: "reference")
        }
    }
}

private extension IRExpressionKind {
    var dotTypeOperand: IRHandle<IRType>? {
        switch self {
        case .zeroValue(let type), .compose(let type, _), .bitcast(_, let type), .atomicResult(let type, _),
             .workgroupUniformLoadResult(let type), .subgroupOperationResult(let type):
            type
        default:
            nil
        }
    }

    var dotOperands: [IRHandle<IRExpression>] {
        switch self {
        case .rayQueryVertexPositions(let query, _), .rayQueryGetIntersection(let query, _):
            [query]
        case .cooperativeLoad(_, _, _, let data):
            [data.pointer, data.stride]
        case .cooperativeMultiplyAdd(let a, let b, let c):
            [a, b, c]
        default:
            operands
        }
    }

    func dotReference(in functionID: String?) -> String? {
        switch self {
        case .constant(let constant): "c\(constant.index)"
        case .override(let value): "o\(value.index)"
        case .globalVariable(let global): "g\(global.index)"
        case .localVariable(let local): functionID.map { "\($0)_local\(local.index)" }
        case .functionArgument(let index): functionID.map { "\($0)_arg\(index)" }
        case .callResult(let function): "f\(function.index)"
        default: nil
        }
    }

    var dotLabel: String {
        switch self {
        case .literal(let literal): "literal \(literal.dotLabel)"
        case .constant: "constant"
        case .override: "override"
        case .runtimeValue: "runtimeValue"
        case .zeroValue: "zeroValue"
        case .compose: "compose"
        case .access: "access"
        case .accessIndex(_, let index): "accessIndex \(index)"
        case .splat(let size, _): "splat \(size.rawValue)"
        case .swizzle(let size, _, let pattern):
            "swizzle \([pattern.x, pattern.y, pattern.z, pattern.w].prefix(Int(size.rawValue)).map(\.rawValue).joined())"
        case .functionArgument: "functionArgument"
        case .globalVariable: "globalVariable"
        case .localVariable: "localVariable"
        case .load: "load"
        case .imageSample(let sample): sample.dotLabel
        case .imageLoad: "imageLoad"
        case .imageQuery(_, let query):
            switch query {
            case .size: "imageQuery size"
            case .numLevels: "imageQuery numLevels"
            case .numLayers: "imageQuery numLayers"
            case .numSamples: "imageQuery numSamples"
            }
        case .unary(let op, _): "unary \(op.rawValue)"
        case .binary(let op, _, _): "binary \(op.rawValue)"
        case .select: "select"
        case .derivative(let axis, let control, _): "derivative \(axis.rawValue) \(control.rawValue)"
        case .relational(let function, _): "relational \(function.rawValue)"
        case .math(let function, _, _, _, _): "math \(function.rawValue)"
        case .cast(_, let target): "cast \(target.kind.rawValue) \(Int(target.width) * 8)"
        case .bitcast: "bitcast"
        case .callResult: "callResult"
        case .atomicResult(_, let comparison): comparison ? "atomicResult comparison" : "atomicResult"
        case .workgroupUniformLoadResult: "workgroupUniformLoadResult"
        case .arrayLength: "arrayLength"
        case .rayQueryVertexPositions(_, let committed):
            "rayQueryVertexPositions \(committed ? "committed" : "candidate")"
        case .rayQueryProceedResult: "rayQueryProceedResult"
        case .rayQueryGetIntersection(_, let committed):
            "rayQueryGetIntersection \(committed ? "committed" : "candidate")"
        case .subgroupBallotResult: "subgroupBallotResult"
        case .subgroupOperationResult: "subgroupOperationResult"
        case .cooperativeLoad(let columns, let rows, let role, let data):
            "cooperativeLoad \(columns.rawValue)x\(rows.rawValue) \(role.rawValue) \(data.rowMajor ? "rowMajor" : "columnMajor")"
        case .cooperativeMultiplyAdd: "cooperativeMultiplyAdd"
        }
    }
}

private extension IRImageSample {
    var dotLabel: String {
        var parts = gather.map { ["imageGather", $0.rawValue] } ?? ["imageSample"]
        switch level {
        case .auto: break
        case .zero: parts.append("zeroLevel")
        case .exact: parts.append("exactLevel")
        case .bias: parts.append("bias")
        case .gradient: parts.append("gradient")
        }
        if depthReference != nil { parts.append("compare") }
        if clampToEdge { parts.append("clampToEdge") }
        return parts.joined(separator: " ")
    }
}

private extension IRLiteral {
    var dotLabel: String {
        switch self {
        case .f64(let value): String(value)
        case .f32(let value): String(value)
        case .f16Bits(let bits): "0x\(String(bits, radix: 16))"
        case .u16(let value): String(value)
        case .i16(let value): String(value)
        case .u32(let value): String(value)
        case .i32(let value): String(value)
        case .u64(let value): String(value)
        case .i64(let value): String(value)
        case .bool(let value): String(value)
        case .abstractInt(let value): String(value)
        case .abstractFloat(let value): String(value)
        }
    }
}
