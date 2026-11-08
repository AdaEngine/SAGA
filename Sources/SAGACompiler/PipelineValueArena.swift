internal import SAGAFrontends
internal import SAGAShaderIR

// Folded aggregates use the same typed global constants as WGSL constant declarations.
struct PipelineValueArena {
    var types: [IRType]
    var constants: [IRConstant]
    var expressions: [IRExpression]

    mutating func replacements(
        _ values: [IRHandle<IRExpression>: PipelineBodyReplacement], in function: IRFunction,
        budget: inout ShaderBudget
    ) throws -> [IRHandle<IRExpression>: IRExpressionKind] {
        var result: [IRHandle<IRExpression>: IRExpressionKind] = [:]
        for handle in values.keys.sorted(by: { $0.index < $1.index }) {
            try budget.consume(.work)
            guard let value = values[handle] else { throw ShaderFailure.invalidHandle }
            let type = try function.expressions.value(for: handle).type
            switch value {
            case .zero:
                result[handle] = .zeroValue(type)
            case .value(.scalar(let literal)):
                result[handle] = .literal(literal)
            case .value(let aggregate):
                let initializer = try compose(aggregate, type: type, budget: &budget)
                let constant = try IRHandle<IRConstant>(index: constants.count)
                try budget.consume(.nodes)
                constants.append(.init(type: type, initializer: initializer))
                result[handle] = .constant(constant)
            }
        }
        return result
    }

    mutating func compose(
        _ value: WGSLPipelineValue, type: IRHandle<IRType>, budget: inout ShaderBudget
    ) throws -> IRHandle<IRExpression> {
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        try budget.consume(.work)
        switch (value, types[type.index].kind) {
        case (.scalar(let literal), .scalar):
            return try append(.literal(literal), type: type, budget: &budget)
        case (.composite(let values), .array(let element, .constant(let count), _)) where values.count == Int(count):
            let components = try values.map { try compose($0, type: element, budget: &budget) }
            return try append(.compose(type: type, components: components), type: type, budget: &budget)
        case (.composite(let values), .structure(let members, _)) where values.count == members.count:
            let components = try zip(values, members).map { try compose($0, type: $1.type, budget: &budget) }
            return try append(.compose(type: type, components: components), type: type, budget: &budget)
        case (.vector(let lanes), .vector(_, let scalar)):
            return try vector(lanes, type: type, scalar: scalar, budget: &budget)
        case (.matrix(let columns), .matrix(_, let rows, let scalar)):
            let columnType = try intern(.vector(size: rows, scalar: scalar), budget: &budget)
            let components = try columns.map { try vector($0, type: columnType, scalar: scalar, budget: &budget) }
            return try append(.compose(type: type, components: components), type: type, budget: &budget)
        default:
            throw ShaderFailure.invalidHandle
        }
    }

    private mutating func vector(
        _ lanes: [IRLiteral], type: IRHandle<IRType>, scalar: IRScalar, budget: inout ShaderBudget
    ) throws -> IRHandle<IRExpression> {
        let laneType = try intern(.scalar(scalar), budget: &budget)
        let components = try lanes.map { try append(.literal($0), type: laneType, budget: &budget) }
        return try append(.compose(type: type, components: components), type: type, budget: &budget)
    }

    private mutating func intern(_ kind: IRTypeKind, budget: inout ShaderBudget) throws -> IRHandle<IRType> {
        for (index, type) in types.enumerated() {
            try budget.consume(.work)
            if type.name == nil, type.kind == kind { return try IRHandle(index: index) }
        }
        try budget.consume(.nodes)
        let handle = try IRHandle<IRType>(index: types.count)
        types.append(.init(kind: kind))
        return handle
    }

    private mutating func append(
        _ kind: IRExpressionKind, type: IRHandle<IRType>, budget: inout ShaderBudget
    ) throws -> IRHandle<IRExpression> {
        try budget.consume(.nodes)
        let handle = try IRHandle<IRExpression>(index: expressions.count)
        expressions.append(.init(type: type, kind: kind))
        return handle
    }
}
