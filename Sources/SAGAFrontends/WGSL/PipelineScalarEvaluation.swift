import SAGAShaderIR

package enum WGSLPipelineValue {
    case scalar(IRLiteral)
    case vector([IRLiteral])
    case matrix([[IRLiteral]])
    case composite([WGSLPipelineValue])

    package var lanes: [IRLiteral] {
        switch self {
        case .scalar(let literal): return [literal]
        case .vector(let lanes): return lanes
        case .matrix(let columns): return columns.flatMap { $0 }
        case .composite: return []
        }
    }

    fileprivate func numericBound() throws -> WGSLScalarConstant {
        switch self {
        case .scalar(let literal):
            return pipelineConstant(literal)
        case .vector(let lanes):
            let elements = lanes.map(pipelineConstant)
            guard let first = elements.first else { throw ShaderFailure.invalidHandle }
            return .vector(.init(elements: elements, element: first.kind))
        case .matrix, .composite:
            throw ShaderFailure.invalidHandle
        }
    }
}

package func evaluateWGSLPipelineScalar(
    _ root: IRHandle<IRExpression>, in module: IRModule,
    overrides: [IRHandle<IROverride>: IRLiteral], budget: inout ShaderBudget
) throws -> IRLiteral {
    guard case .scalar(let value) = try evaluateWGSLPipelineValue(root, in: module, overrides: overrides, budget: &budget)
    else { throw ShaderFailure.invalidHandle }
    return value
}

package func evaluateWGSLPipelineValue(
    _ root: IRHandle<IRExpression>, in module: IRModule,
    overrides: [IRHandle<IROverride>: IRLiteral], budget: inout ShaderBudget
) throws -> WGSLPipelineValue {
    let graph = try module.pipelineGraph(root, budget: &budget)
    let fallback = pipelineToken(try module.globalExpressions.value(for: root).location)
    let parser = WGSLParser(tokens: [], budget: budget, options: .init())
    defer { budget = parser.budget }
    var evaluator = WGSLScalarEvaluator(parser: parser, tokens: [], fallback: fallback)
    var mapped: [IRHandle<IRExpression>: Int] = [:]
    func index(_ operand: IRHandle<IRExpression>) throws -> Int {
        guard let index = mapped[operand] else { throw ShaderFailure.invalidHandle }
        return index
    }

    for handle in graph.sorted(by: { $0.index < $1.index }) {
        try parser.budget.consume(.work)
        let expression = try module.globalExpressions.value(for: handle)
        let token = pipelineToken(expression.location)
        let type = try module.types.value(for: expression.type).kind
        let kind = try pipelineKind(type, handle: expression.type, evaluator: evaluator, at: token)
        let operation: WGSLConstNode.Op
        switch expression.kind {
        case .literal(let literal):
            operation = .value(pipelineConstant(literal))
        case .zeroValue where type.isPipelineComposite:
            operation = .value(try pipelineZero(expression.type, module: module, evaluator: evaluator, at: token))
        case .zeroValue where kind.lanes > 1 || kind.isMatrix:
            operation = try pipelineConstructor(kind, arguments: [])
        case .zeroValue:
            operation = .value(try evaluator.zero(kind, at: token))
        case .compose(_, let components):
            operation = try pipelineConstructor(kind, arguments: components.map(index))
        case .splat(_, let value):
            operation = try pipelineConstructor(kind, arguments: [index(value)])
        case .swizzle(let size, let vector, let pattern):
            let lanes = [pattern.x, pattern.y, pattern.z, pattern.w].prefix(Int(size.rawValue))
            operation = try .swizzle(index(vector), lanes.map(pipelineLaneIndex))
        case .access(let base, let subscriptValue):
            operation = try .index(index(base), index(subscriptValue))
        case .accessIndex(let base, let offset):
            let subscriptValue = try evaluator.append(
                .value(.integer(.init(value: Int64(offset), kind: .uint))), .integer(.uint), token)
            operation = try .index(index(base), subscriptValue)
        case .constant(let constant):
            mapped[handle] = try index(module.constants.value(for: constant).initializer)
            continue
        case .override(let handle):
            guard let value = overrides[handle] else { throw ShaderFailure.invalidHandle }
            operation = .value(pipelineConstant(value))
        case .unary(let op, let operand):
            let symbol: String = switch op {
            case .negate: "-"
            case .logicalNot: "!"
            case .bitwiseNot: "~"
            }
            operation = try .unary(symbol, index(operand))
        case .binary(let op, let left, let right):
            guard let symbol = WGSLParser.binary.first(where: { $0.value.1 == op })?.key else {
                throw ShaderFailure.invalidHandle
            }
            operation = try .binary(symbol, index(left), index(right))
        case .select(let condition, let accept, let reject):
            operation = try .call("select", [index(reject), index(accept), index(condition)])
        case .cast(let operand, _) where kind.lanes > 1 || kind.isMatrix:
            operation = try pipelineConstructor(kind, arguments: [index(operand)], explicit: true)
        case .cast(let operand, let target):
            let name: String
            switch target {
            case .bool: name = "bool"
            case .i16: name = "i16"
            case .u16: name = "u16"
            case .i32: name = "i32"
            case .u32: name = "u32"
            case .i64: name = "i64"
            case .u64: name = "u64"
            case .f32: name = "f32"
            case .f64: name = "f64"
            case .f16: name = "f16"
            default: throw ShaderFailure.invalidHandle
            }
            operation = try .call(name, [index(operand)])
        case .bitcast(let operand, _):
            let source = try index(operand)
            operation = .bitcast(source, evaluator.nodes[source].kind)
        case .relational(let reduction, let operand) where reduction.isBooleanReduction:
            operation = try .call(reduction.rawValue, [index(operand)])
        case .math(let function, let operand, nil, nil, nil) where function.decomposesFloat:
            let argument = try index(operand)
            operation = .decomposition(function, argument, evaluator.nodes[argument].kind)
        case .math(.ldexp, let mantissa, let exponent?, nil, nil):
            let power = try index(exponent)
            operation = try .ldexp(index(mantissa), power, evaluator.nodes[power].kind)
        case .math(let function, let first, let second, let third, let fourth):
            let arguments = try ([first] + [second, third, fourth].compactMap { $0 }).map(index)
            let signature = arguments.map { evaluator.nodes[$0].kind }
            if function.numericBuiltin != nil {
                operation = .numeric(function, arguments)
            } else if function.integerBitArity != nil {
                operation = .integerBit(function, arguments, signature)
            } else if let packing = function.packingBuiltin {
                operation = .packing(packing, arguments[0])
            } else if function.geometricArity != nil {
                operation = .geometric(function, arguments, signature)
            } else {
                throw ShaderFailure.invalidHandle
            }
        default:
            throw ShaderFailure.invalidHandle
        }
        mapped[handle] = try evaluator.append(operation, kind, token)
    }
    let result = try evaluator.evaluateTyped(index(root))
    return try pipelineValue(result, budget: &parser.budget)
}

private func pipelineValue(_ result: WGSLScalarConstant, budget: inout ShaderBudget) throws -> WGSLPipelineValue {
    try budget.consume(.work)
    try budget.enterDepth()
    defer { try? budget.leaveDepth() }
    switch result {
    case .vector(let value):
        return .vector(try value.elements.map { try $0.literal })
    case .matrix(let value):
        return .matrix(try value.columns.map { try $0.elements.map { try $0.literal } })
    case .composite(_, let components):
        try budget.consume(.nodes, amount: components.count)
        return .composite(try components.map { try pipelineValue($0, budget: &budget) })
    default:
        return .scalar(try result.literal)
    }
}

private func pipelineConstructor(
    _ kind: WGSLConstKind, arguments: [Int], explicit: Bool = false
) throws -> WGSLConstNode.Op {
    switch kind {
    case .vector(let width, let element):
        return .vector(width, element, arguments, explicit)
    case .matrix(let columns, let rows, let element):
        return .matrix(columns, rows, element, arguments, explicit)
    case .composite(let type, let count) where arguments.count == count:
        return .composite(type, arguments)
    default:
        throw ShaderFailure.invalidHandle
    }
}

private func pipelineKind(
    _ kind: IRTypeKind, handle: IRHandle<IRType>, evaluator: WGSLScalarEvaluator, at token: WGSLToken
) throws -> WGSLConstKind {
    switch kind {
    case .array(_, .constant(let count), _): return .composite(handle, Int(count))
    case .structure(let members, _): return .composite(handle, members.count)
    default: return try evaluator.numericConstKind(kind, at: token)
    }
}

private func pipelineZero(
    _ type: IRHandle<IRType>, module: IRModule, evaluator: WGSLScalarEvaluator, at token: WGSLToken
) throws -> WGSLScalarConstant {
    let parser = evaluator.parser
    try parser.budget.consume(.work)
    try parser.budget.enterDepth()
    defer { try? parser.budget.leaveDepth() }
    let kind = try module.types.value(for: type).kind
    switch kind {
    case .array(let element, .constant(let count), _):
        try parser.budget.consume(.nodes, amount: Int(count))
        let value = try pipelineZero(element, module: module, evaluator: evaluator, at: token)
        return .composite(type, Array(repeating: value, count: Int(count)))
    case .structure(let members, _):
        try parser.budget.consume(.nodes, amount: members.count)
        return .composite(type, try members.map { try pipelineZero($0.type, module: module, evaluator: evaluator, at: token) })
    case .vector(let width, let scalar):
        let leaf = try WGSLConstKind.scalar(scalar, at: token)
        return .vector(.init(elements: Array(repeating: try evaluator.zero(leaf, at: token), count: Int(width.rawValue)), element: leaf))
    case .matrix(let columns, let rows, let scalar):
        let leaf = try WGSLConstKind.scalar(scalar, at: token)
        let column = WGSLVectorConstant(elements: Array(repeating: try evaluator.zero(leaf, at: token), count: Int(rows.rawValue)),
                                        element: leaf)
        return .matrix(.init(columns: Array(repeating: column, count: Int(columns.rawValue)), rows: Int(rows.rawValue), element: leaf))
    default:
        return try evaluator.zero(evaluator.numericConstKind(kind, at: token), at: token)
    }
}

private extension IRTypeKind {
    var isPipelineComposite: Bool {
        switch self {
        case .array(_, .constant, _), .structure: return true
        default: return false
        }
    }
}

private func pipelineLaneIndex(_ lane: IRSwizzleComponent) -> Int {
    switch lane {
    case .x: return 0
    case .y: return 1
    case .z: return 2
    case .w: return 3
    }
}

private func pipelineToken(_ location: ShaderLocation?) -> WGSLToken {
    if case .source(let span) = location {
        return WGSLToken(kind: .symbol, text: "override", start: span.startUTF8, end: span.endUTF8)
    }
    return WGSLToken(kind: .symbol, text: "override", start: 0, end: 0)
}

private func pipelineConstant(_ literal: IRLiteral) -> WGSLScalarConstant {
    switch literal {
    case .bool(let value): return .boolean(value)
    case .i16(let value): return .integer(.init(value: Int64(value), kind: .sint16))
    case .u16(let value): return .integer(.init(value: Int64(value), kind: .uint16))
    case .i32(let value): return .integer(.init(value: Int64(value), kind: .sint))
    case .u32(let value): return .integer(.init(value: Int64(value), kind: .uint))
    case .i64(let value): return .integer(.init(value: value, kind: .sint64))
    case .u64(let value): return .integer(.init(value: Int64(bitPattern: value), kind: .uint64))
    case .f32(let value): return .floating(.init(value: Double(value), kind: .f32))
    case .f64(let value): return .floating(.init(value: value, kind: .f64))
    case .f16Bits(let bits): return .floating(.init(value: Double(Binary16.decode(bits)), kind: .f16))
    case .abstractInt(let value): return .integer(.init(value: value, kind: .abstract))
    case .abstractFloat(let value): return .floating(.init(value: value, kind: .abstract))
    }
}

package func validateWGSLPipelineNumericBounds(
    _ function: IRMathFunction, lower: WGSLPipelineValue, upper: WGSLPipelineValue,
    at location: ShaderLocation?, budget: inout ShaderBudget
) throws {
    let token = pipelineToken(location)
    let parser = WGSLParser(tokens: [], budget: budget, options: .init())
    defer { budget = parser.budget }
    let evaluator = WGSLScalarEvaluator(parser: parser, tokens: [], fallback: token)
    try evaluator.validateNumericBounds(function, lower.numericBound(), upper.numericBound(), at: token)
}
