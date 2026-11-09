import SAGAShaderIR

extension WGSLParser {
    // The RHS is emitted inside a conditional block, never evaluated eagerly.
    func shortCircuit(
        _ left: WGSLValue, op: WGSLToken, precedence: Int, expressionStart: Int, minimum: Int
    ) throws -> WGSLValue {
        let lhs = try loaded(left,at:op)
        guard types[lhs.type.index].kind == .scalar(.bool) else { throw wgslError(.typeMismatch,"Logical operand must be bool",op) }
        if body.pipelineStages[lhs.handle.index] == .constant {
            let end = try logicalExpressionEnd(minimum: minimum)
            if case .boolean(let result)? = try constantLogicalValue(expressionStart..<end, at: op) {
                cursor = end
                return try expressionNode(.literal(.bool(result)), type: lhs.type, at: op, emit: false)
            }
        }
        let saved = body.current
        body.current = []
        let rhs = try loaded(expression(precedence + 1), at: op)
        guard rhs.type == lhs.type else {
            throw wgslError(.typeMismatch, "Logical operand must be bool", op)
        }
        let rightStatements = body.current
        body.current = saved
        if pipelineExpressionEnd != nil ||
           (hasOverrideDeclarations && body.pipelineStages[lhs.handle.index] != .runtime &&
            body.pipelineStages[rhs.handle.index] != .runtime) {
            body.current.append(contentsOf: rightStatements)
            return try expressionNode(.binary(op: op.text == "&&" ? .logicalAnd : .logicalOr,
                                              left: lhs.handle, right: rhs.handle), type: lhs.type, at: op)
        }
        try node()
        let local = try IRHandle<IRLocalVariable>(index: body.locals.count)
        body.locals.append(.init(type: lhs.type))
        let ptr = try expressionNode(.localVariable(local), type: intern(.pointer(base: lhs.type, space: .function)),
                                     at: op, emit: false)
        try statement(.store(pointer: ptr.handle, value: lhs.handle), at: op)
        let outer = body.current
        body.current = rightStatements
        try statement(.store(pointer: ptr.handle, value: rhs.handle), at: op)
        let rightBlock = try saveBlock(body.current, at: op)
        let empty = try saveBlock([], at: op)
        body.current = outer
        try statement(.ifCondition(condition: lhs.handle, accept: op.text == "&&" ? rightBlock : empty,
                                   reject: op.text == "&&" ? empty : rightBlock), at: op)
        return try expressionNode(.load(pointer:ptr.handle),type:lhs.type,at:op)
    }

    private func logicalExpressionEnd(minimum: Int) throws -> Int {
        var depth = 0
        var end = cursor
        while end != pipelineExpressionEnd {
            try budget.consume(.work)
            let next = tokens[end]
            if next.kind == .end || (depth == 0 && (
                next.closesGroup || [";", ",", "{", "}", ":"].contains(next.text) ||
                (Self.binary[next.text]?.0 ?? Int.max) < minimum
            )) {
                break
            }
            if next.opensGroup { depth += 1 }
            else if next.closesGroup { depth -= 1 }
            end += 1
        }
        return end
    }

    private func constantLogicalValue(_ range: Range<Int>, at op: WGSLToken) throws -> WGSLScalarConstant? {
        // Check the full typed expression before evaluating it. Parsing the RHS as a standalone
        // expression would fold value-only errors even when the logical operator skips it.
        do {
            return try knownConstantValue(range, at: op)
        } catch ShaderFailure.diagnostics(let diagnostics)
            where diagnostics.allSatisfy({ $0.code == .unsupportedConstruct }) {
            return nil
        } catch ShaderFailure.budgetExceeded(.depth) {
            // The constant evaluator has a smaller recursion limit than normal expression parsing.
            return nil
        }
    }
    // Checked scalar constant folding. No host trap on malformed untrusted literals.
    func fold(_ op: IRBinaryOperator, left: WGSLValue, right: WGSLValue, at t: WGSLToken) throws -> IRLiteral? {
        try budget.consume(.work)
        guard case .literal(let l) = body.expressions[left.handle.index].kind,
              case .literal(let r) = body.expressions[right.handle.index].kind else { return nil }
        if case .abstractInt(let a) = l, case .abstractInt(let b) = r {
            let result: (partialValue: Int64, overflow: Bool)
            switch op {
            case .add: result = a.addingReportingOverflow(b)
            case .subtract: result = a.subtractingReportingOverflow(b)
            case .multiply: result = a.multipliedReportingOverflow(by:b)
            case .divide,.modulo:
                guard b != 0, !(a == Int64.min && b == -1) else { throw wgslError(.invalidType,"Invalid constant division",t) }
                return .abstractInt(op == .divide ? a/b : a%b)
            case .and:return .abstractInt(a & b)
            case .inclusiveOr:return .abstractInt(a | b)
            case .exclusiveOr:return .abstractInt(a ^ b)
            case .equal:return .bool(a == b)
            case .notEqual:return .bool(a != b)
            case .less:return .bool(a < b)
            case .lessEqual:return .bool(a <= b)
            case .greater:return .bool(a > b)
            case .greaterEqual:return .bool(a >= b)
            default:return nil
            }
            guard !result.overflow else { throw wgslError(.invalidType,"Abstract integer overflow",t) }
            return .abstractInt(result.partialValue)
        }
        if types[left.type.index].kind == .scalar(.abstractFloat),
           let a = try pureCompositeConstant(left, at: t), let b = try pureCompositeConstant(right, at: t) {
            let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: t)
            return try evaluator.floatingBinary(t.text, a, b, at: t).literal
        }
        return nil
    }
}
