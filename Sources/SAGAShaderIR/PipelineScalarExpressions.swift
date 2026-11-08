extension IRExpressionKind {
    package func remappingPipelineOperands(
        _ map: (IRHandle<IRExpression>) throws -> IRHandle<IRExpression>
    ) rethrows -> IRExpressionKind? {
        switch self {
        case .literal, .zeroValue, .constant, .override:
            return self
        case .compose(let type, let components):
            return try .compose(type: type, components: components.map(map))
        case .splat(let size, let value):
            return try .splat(size: size, value: map(value))
        case .swizzle(let size, let vector, let pattern):
            return try .swizzle(size: size, vector: map(vector), pattern: pattern)
        case .access(let base, let index):
            return try .access(base: map(base), index: map(index))
        case .accessIndex(let base, let index):
            return try .accessIndex(base: map(base), index: index)
        case .unary(let op, let value):
            return try .unary(op: op, expression: map(value))
        case .binary(let op, let left, let right):
            return try .binary(op: op, left: map(left), right: map(right))
        case .select(let condition, let accept, let reject):
            return try .select(condition: map(condition), accept: map(accept), reject: map(reject))
        case .cast(let value, let target):
            return try .cast(expression: map(value), target: target)
        case .bitcast(let value, let target):
            return try .bitcast(expression: map(value), type: target)
        case .relational(let operation, let argument) where operation.isBooleanReduction:
            return try .relational(function: operation, argument: map(argument))
        case .math(let operation, let a, let b, let c, let d) where operation.numericBuiltin != nil || operation == .ldexp ||
            operation.integerBitArity != nil || operation.geometricArity != nil || operation.packingBuiltin != nil ||
            operation.decomposesFloat:
            return try .math(function: operation, argument: map(a), argument1: b.map(map),
                             argument2: c.map(map), argument3: d.map(map))
        default:
            return nil
        }
    }

    package var pipelineOperands: [IRHandle<IRExpression>]? {
        var operands: [IRHandle<IRExpression>] = []
        guard remappingPipelineOperands({ operand in
            operands.append(operand)
            return operand
        }) != nil else { return nil }
        return operands
    }
}

extension IRModule {
    package func pipelineGraph(
        _ root: IRHandle<IRExpression>, budget: inout ShaderBudget
    ) throws -> Set<IRHandle<IRExpression>> {
        var visited = Set([root])
        var pending = [root]
        while let handle = pending.popLast() {
            try budget.consume(.work)
            let expression = try globalExpressions.value(for: handle)
            guard try types.value(for: expression.type).kind.isPipelineValueType,
                  var operands = expression.kind.pipelineOperands else {
                throw ShaderFailure.diagnostics([.init(severity: .error, code: .unsupportedConstruct,
                    message: "Pipeline expression requires supported constant-evaluable operations", location: expression.location)])
            }
            if case .constant(let constant) = expression.kind {
                operands.append(try constants.value(for: constant).initializer)
            }
            for operand in operands where visited.insert(operand).inserted {
                pending.append(operand)
            }
        }
        return visited
    }
}


package enum IRPipelineEvaluationStage {
    case constant, pipeline, runtime
}

extension IRExpressionKind {
    package func pipelineEvaluationStage(
        type: IRTypeKind, stageOf: (IRHandle<IRExpression>) -> IRPipelineEvaluationStage
    ) -> IRPipelineEvaluationStage {
        guard type.isPipelineValueType, let operands = pipelineOperands else { return .runtime }
        if case .override = self { return .pipeline }
        var result = IRPipelineEvaluationStage.constant
        for operand in operands {
            switch stageOf(operand) {
            case .runtime: return .runtime
            case .pipeline: result = .pipeline
            case .constant: break
            }
        }
        return result
    }
}


extension IRTypeKind {
    package var isPipelineValueType: Bool {
        switch self {
        case .scalar(let lane), .vector(_, let lane):
            return [.bool, .i32, .u32, .i64, .u64, .f32, .f16, .abstractInt, .abstractFloat].contains(lane)
        case .matrix(_, _, let lane):
            return [.f32, .f16, .abstractFloat].contains(lane)
        case .array(_, .constant, _), .structure:
            return true
        default: return false
        }
    }
}
