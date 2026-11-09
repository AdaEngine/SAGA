internal import SAGAFrontends
internal import SAGAShaderIR

enum PipelineBodyReplacement {
    case value(WGSLPipelineValue)
    case zero
}

func specializePipelineBody(
    _ function: IRFunction, in module: IRModule, overrides: [IRHandle<IROverride>: IRLiteral],
    budget: inout ShaderBudget
) throws -> [IRHandle<IRExpression>: PipelineBodyReplacement] {
    let expressions = function.expressions.values
    var stages: [IRPipelineEvaluationStage] = []
    var pipelineNodes: Set<IRHandle<IRExpression>> = []
    for (index, expression) in expressions.enumerated() {
        try budget.consume(.work)
        let type = try module.types.value(for: expression.type).kind
        let stage = expression.kind.pipelineEvaluationStage(type: type) { operand in
            stages[operand.index]
        }
        stages.append(stage)
        if stage == .pipeline { pipelineNodes.insert(try IRHandle(index: index)) }
    }
    guard !pipelineNodes.isEmpty else { return [:] }

    // Expressions consumed only by another pipeline expression are evaluated lazily with that parent.
    // Actual runtime uses and unused top-level expressions are roots. Emit statements are scheduling metadata.
    var roots = pipelineNodes
    var runtimeUses: Set<IRHandle<IRExpression>> = []
    for (index, expression) in expressions.enumerated() {
        try budget.consume(.work)
        let operands = expression.kind.operands
        if stages[index] == .pipeline {
            roots.subtract(operands)
        } else {
            for operand in operands where stages[operand.index] == .pipeline { runtimeUses.insert(operand) }
        }
    }
    for statement in function.statements.values {
        try budget.consume(.work)
        for operand in statement.kind.pipelineValueUses where stages[operand.index] == .pipeline {
            runtimeUses.insert(operand)
        }
    }
    roots.formUnion(runtimeUses)

    // Reuse the typed WGSL evaluator with function-local values remapped into a temporary arena.
    try budget.consume(.work, amount: module.globalExpressions.count)
    var arena = module.globalExpressions.values
    var remap: [IRHandle<IRExpression>?] = Array(repeating: nil, count: expressions.count)
    for (index, expression) in expressions.enumerated() where stages[index] != .runtime {
        try budget.consume(.nodes)
        guard let kind = try expression.kind.remappingPipelineOperands({ operand in
            guard let mapped = remap[operand.index] else { throw ShaderFailure.invalidHandle }
            return mapped
        }) else { throw ShaderFailure.invalidHandle }
        remap[index] = try IRHandle(index: arena.count)
        arena.append(.init(type: expression.type, kind: kind, location: expression.location))
    }
    let pipelineModule = IRModule(types: module.types, constants: module.constants,
                                overrides: module.overrides, globalExpressions: .init(arena))
    var evaluated: [IRHandle<IRExpression>: WGSLPipelineValue] = [:]
    func value(_ handle: IRHandle<IRExpression>) throws -> WGSLPipelineValue? {
        if stages[handle.index] == .runtime { return nil }
        if let value = evaluated[handle] { return value }
        guard let mapped = remap[handle.index] else { throw ShaderFailure.invalidHandle }
        let value = try evaluateWGSLPipelineValue(mapped, in: pipelineModule, overrides: overrides, budget: &budget)
        evaluated[handle] = value
        return value
    }
    for root in roots.sorted(by: { $0.index < $1.index }) { _ = try value(root) }

    // Some runtime operations have mandatory checks on a pipeline-time operand.
    func integerWidth(_ type: IRHandle<IRType>) throws -> UInt32 {
        switch try module.types.value(for: type).kind {
        case .scalar(let scalar), .vector(_, let scalar): return UInt32(scalar.width) * 8
        default: throw ShaderFailure.invalidHandle
        }
    }
    for (index, expression) in expressions.enumerated() where stages[index] == .runtime {
        try budget.consume(.work)
        switch expression.kind {
        case .binary(let op, _, let right) where op == .divide || op == .modulo:
            if stages[right.index] == .pipeline,
               try value(right)?.lanes.contains(where: { [.i32(0), .u32(0), .i64(0), .u64(0)].contains($0) }) == true {
                throw pipelineFailure("Pipeline integer divisor must be nonzero", at: expression.location)
            }
        case .binary(let op, _, let right) where op == .shiftLeft || op == .shiftRight:
            guard stages[right.index] == .pipeline else { break }
            let bitWidth = try integerWidth(expression.type)
            if try value(right)?.lanes.contains(where: { lane in
                if case .u32(let count) = lane { return count >= bitWidth }
                return false
            }) == true {
                throw pipelineFailure("Pipeline shift count must be below \(bitWidth)", at: expression.location)
            }
        case .math(.ldexp, _, let exponent?, nil, nil):
            guard stages[exponent.index] == .pipeline,
                  let shape = try module.types.value(for: expression.type).kind.concreteFloatShape else { break }
            let limit = shape.scalar == .f16 ? 16 : 128
            if try value(exponent)?.lanes.contains(where: { lane in
                if case .i32(let power) = lane { return power > limit }
                return false
            }) == true {
                throw pipelineFailure("Pipeline ldexp exponent exceeds the floating-point format limit", at: expression.location)
            }
        case .math(let op, let first, let second?, let third?, nil) where op == .clamp || op == .smoothStep:
            let (low, high) = op == .clamp ? (second, third) : (first, second)
            guard stages[low.index] == .pipeline || stages[high.index] == .pipeline,
                  let lower = try value(low), let upper = try value(high) else { break }
            try validateWGSLPipelineNumericBounds(op, lower: lower, upper: upper, at: expression.location, budget: &budget)
        case .math(.extractBits, _, let offset?, let count?, nil),
             .math(.insertBits, _, _, let offset?, let count?):
            guard stages[offset.index] == .pipeline || stages[count.index] == .pipeline,
                  case .scalar(.u32(let low))? = try value(offset), case .scalar(.u32(let width))? = try value(count) else { break }
            let bitWidth = try integerWidth(expression.type)
            if UInt64(low) + UInt64(width) > UInt64(bitWidth) {
                throw pipelineFailure("Pipeline bit range exceeds \(bitWidth) bits", at: expression.location)
            }
        default:
            break
        }
    }

    var replacements: [IRHandle<IRExpression>: PipelineBodyReplacement] = [:]
    for handle in pipelineNodes {
        try budget.consume(.work)
        // An unreferenced child belongs to a folded parent and may have been skipped by short-circuiting.
        replacements[handle] = evaluated[handle].map(PipelineBodyReplacement.value) ?? .zero
    }
    return replacements
}

private func pipelineFailure(_ message: String, at location: ShaderLocation?) -> ShaderFailure {
    .diagnostics([.init(severity: .error, code: .invalidType, message: message, location: location)])
}

private extension IRStatementKind {
    var pipelineValueUses: [IRHandle<IRExpression>] {
        switch self {
        case .store(let pointer, let value): return [pointer, value]
        case .ifCondition(let condition, _, _), .switchValue(let condition, _): return [condition]
        case .returnValue(let value?), .loop(_, _, let value?): return [value]
        case .call(_, let arguments, _), .debugPrintf(_, let arguments): return arguments
        case .imageStore(let image, let coordinate, let layer, let value),
             .imageAtomic(let image, let coordinate, let layer, _, let value):
            return [image, coordinate, value] + [layer].compactMap { $0 }
        case .atomic(let pointer, let function, let value, _):
            if case .exchange(let comparison?, _) = function { return [pointer, value, comparison] }
            return [pointer, value]
        case .workgroupUniformLoad(let pointer, _): return [pointer]
        default: return []
        }
    }
}
