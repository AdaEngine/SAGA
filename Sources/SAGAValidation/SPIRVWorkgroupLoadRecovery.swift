import SAGAShaderIR

struct SPIRVWorkgroupLoadRecovery {
    struct Candidate {
        let before: IRHandle<IRStatement>
        let load: IRHandle<IRStatement>
        let after: IRHandle<IRStatement>
        let pointer: IRHandle<IRExpression>
    }

    typealias Candidates = [Int: [IRHandle<IRExpression>: Candidate]]

    static func collect(in module: IRModule, budget: inout ShaderBudget) throws -> Candidates {
        guard module.entryPoints.contains(where: {
            if case .binary? = $0.location { return true }
            return false
        }), module.globalVariables.values.contains(where: { $0.space == .workgroup }) else { return [:] }

        var result: Candidates = [:]
        for (index, function) in module.functions.values.enumerated() {
            try budget.consume(.work)
            for block in function.blocks.values {
                try budget.consume(.work)
                let statements = block.statements
                var offset = 0
                while statements.count - offset >= 3 {
                    try budget.consume(.work)
                    let before = try function.statements.value(for: statements[offset])
                    let load = try function.statements.value(for: statements[offset + 1])
                    let after = try function.statements.value(for: statements[offset + 2])
                    if before.kind == .controlBarrier(.workgroup), after.kind == .controlBarrier(.workgroup),
                       case .binary(let beforeSpan)? = before.location, case .binary(let loadSpan)? = load.location,
                       case .binary(let afterSpan)? = after.location,
                       beforeSpan.endWord == loadSpan.startWord, loadSpan.endWord == afterSpan.startWord,
                       case .emit(let expressions) = load.kind, expressions.count == 1,
                       case .load(let pointer) = try function.expressions.value(for: expressions[0]).kind,
                       case .pointer(_, .workgroup) = try module.types.value(for: function.expressions.value(for: pointer).type).kind {
                        try budget.consume(.nodes)
                        result[index, default: [:]][expressions[0]] = .init(
                            before: statements[offset], load: statements[offset + 1],
                            after: statements[offset + 2], pointer: pointer)
                        offset += 3
                    } else {
                        offset += 1
                    }
                }
            }
        }
        return result
    }

    static func rewrite(
        _ module: IRModule, candidates: Candidates, budget: inout ShaderBudget
    ) throws -> IRModule {
        var functions = module.functions.values
        try budget.consume(.work, amount: functions.count)
        for (index, loads) in candidates {
            let function = functions[index]
            var expressions = function.expressions.values
            var statements = function.statements.values
            try budget.consume(.work, amount: expressions.count)
            try budget.consume(.work, amount: statements.count)
            for (handle, candidate) in loads {
                try budget.consume(.work)
                let old = expressions[handle.index]
                expressions[handle.index] = .init(type: old.type, kind: .workgroupUniformLoadResult(type: old.type),
                                                 location: old.location)
                statements[candidate.load.index] = .init(kind: .workgroupUniformLoad(pointer: candidate.pointer, result: handle),
                                                          location: statements[candidate.load.index].location)
                // Preserve statement handles and ownership. The uniform load already includes both barriers.
                for barrier in [candidate.before, candidate.after] {
                    statements[barrier.index] = .init(kind: .emit([]), location: statements[barrier.index].location)
                }
            }
            functions[index] = .init(name: function.name, arguments: function.arguments, result: function.result,
                                     localVariables: function.localVariables, expressions: .init(expressions),
                                     namedExpressions: function.namedExpressions, statements: .init(statements),
                                     blocks: function.blocks, body: function.body,
                                     diagnosticFilter: function.diagnosticFilter, location: function.location)
        }
        return .init(types: module.types, specialTypes: module.specialTypes, constants: module.constants,
                     overrides: module.overrides, globalVariables: module.globalVariables,
                     globalExpressions: module.globalExpressions, functions: .init(functions), entryPoints: module.entryPoints,
                     diagnosticFilters: module.diagnosticFilters, diagnosticFilter: module.diagnosticFilter,
                     documentation: module.documentation, allowsPointerSubviews: module.allowsPointerSubviews,
                     uniformBufferStandardLayout: module.uniformBufferStandardLayout,
                     requiresSubgroupUniformity: module.requiresSubgroupUniformity)
    }
}
