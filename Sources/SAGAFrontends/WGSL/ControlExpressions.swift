import SAGAShaderIR

extension WGSLParser {
    // The RHS is emitted inside a conditional block, never evaluated eagerly.
    func shortCircuit(_ left: WGSLValue, op: WGSLToken, precedence: Int) throws -> WGSLValue {
        let lhs = try loaded(left,at:op)
        guard types[lhs.type.index].kind == .scalar(.bool) else { throw wgslError(.typeMismatch,"Logical operand must be bool",op) }
        try node(); let local = try IRHandle<IRLocalVariable>(index:body.locals.count)
        body.locals.append(IRLocalVariable(type:lhs.type))
        let ptr = try expressionNode(.localVariable(local),type:intern(.pointer(base:lhs.type,space:.function)),at:op,emit:false)
        try statement(.store(pointer:ptr.handle,value:lhs.handle),at:op)
        let saved = body.current; body.current = []
        let rhs = try loaded(expression(precedence+1),at:op)
        guard rhs.type == lhs.type else { throw wgslError(.typeMismatch,"Logical operand must be bool",op) }
        try statement(.store(pointer:ptr.handle,value:rhs.handle),at:op)
        let rightBlock = try saveBlock(body.current,at:op), empty = try saveBlock([],at:op)
        body.current = saved
        try statement(.ifCondition(condition:lhs.handle,accept:op.text == "&&" ? rightBlock : empty,reject:op.text == "&&" ? empty : rightBlock),at:op)
        return try expressionNode(.load(pointer:ptr.handle),type:lhs.type,at:op)
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
        if case .abstractFloat(let a) = l, case .abstractFloat(let b) = r {
            let n: Double
            switch op {
            case .add:n = a+b
            case .subtract:n = a-b
            case .multiply:n = a*b
            case .divide:n = a/b
            case .equal:return .bool(a == b)
            case .notEqual:return .bool(a != b)
            case .less:return .bool(a < b)
            case .lessEqual:return .bool(a <= b)
            case .greater:return .bool(a > b)
            case .greaterEqual:return .bool(a >= b)
            default:return nil
            }
            guard n.isFinite else { throw wgslError(.invalidType,"Invalid constant float arithmetic",t) }; return .abstractFloat(n)
        }
        return nil
    }
}
