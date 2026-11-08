import SAGAShaderIR

extension GLSLParser {
    func switchStatement(at start: GLSLToken) throws -> Bool {
        try expect("(")
        let selector = try rvalue(expression(), at: start)
        let selectorType = kind(selector.type)
        guard selectorType == .scalar(.i32) || selectorType == .scalar(.u32) else {
            throw glslError(start, .typeMismatch, "Switch selector requires a scalar int or uint")
        }
        try expect(")")
        try expect("{")
        let c = try requireContext()
        var outer = c.current
        c.current = []
        c.scopes.append([:])
        c.switches.append((c.loopDepth, false))
        defer {
            c.switches.removeLast()
            c.scopes.removeLast()
            c.current = outer
        }

        var cases: [IRSwitchCase] = []
        var seen = Set<IRSwitchValue>()
        var terminal = false
        while text != "}" {
            guard text == "case" || text == "default" else {
                throw glslError(token, .unexpectedToken, "Expected a switch label")
            }
            var labels: [IRSwitchValue] = []
            repeat {
                let label = try advance()
                let value: IRSwitchValue
                if label.text == "default" {
                    value = .default
                } else {
                    let constant = try integerConstant(at: label)
                    value = selectorType == .scalar(.u32) ? .u32(constant.bits) : .i32(Int32(bitPattern: constant.bits))
                }
                guard seen.insert(value).inserted else {
                    throw glslError(label, .duplicateName, "Duplicate switch label")
                }
                try expect(":")
                labels.append(value)
            } while text == "case" || text == "default"
            guard text != "}" else {
                throw glslError(token, .unexpectedToken, "Last switch label requires a statement")
            }
            terminal = false
            while text != "case" && text != "default" && text != "}" {
                if text.isEmpty { throw glslError(token, .unexpectedToken, "Unterminated switch") }
                if terminal { throw glslError(token, .unsupportedConstruct, "Unreachable statements not lowered") }
                terminal = try statement()
            }
            let body = try block(c.current)
            c.current = []
            for label in labels {
                try budget.consume(.nodes)
                cases.append(.init(value: label, body: body, fallThrough: !terminal && text != "}"))
            }
        }
        try expect("}")
        let hasDefault = seen.contains(.default)
        if !hasDefault { cases.append(.init(value: .default, body: try block([]))) }
        let handle = try statementNode(.switchValue(selector: selector.expression, cases: cases), at: start)
        let exits = hasDefault && terminal && c.switches.last?.hasBreak == false
        outer.append(handle)
        return exits
    }

    func integerConstant(at start: GLSLToken) throws -> (type: IRHandle<IRType>, bits: UInt32, literal: IRLiteral) {
        let c = try requireContext()
        let outer = c.current
        let expressions = c.expressions.count, statements = c.statements.count
        let blocks = c.blocks.count, locals = c.locals.count
        c.current = []
        defer {
            c.current = outer
            c.expressions.removeSubrange(expressions...)
            c.statements.removeSubrange(statements...)
            c.blocks.removeSubrange(blocks...)
            c.locals.removeSubrange(locals...)
        }
        let value = try rvalue(expression(), at: start)
        let bits = try integerConstantBits(value.expression, at: start)
        let literal: IRLiteral = kind(value.type) == .scalar(.i32) ? .i32(Int32(bitPattern: bits)) : .u32(bits)
        return (value.type, bits, literal)
    }

    private func integerConstantBits(_ handle: IRHandle<IRExpression>, at start: GLSLToken) throws -> UInt32 {
        try enter()
        defer { depth -= 1 }
        let expression = try requireContext().expressions[handle.index]
        let type = kind(expression.type)
        guard type == .scalar(.i32) || type == .scalar(.u32) else {
            throw glslError(start, .typeMismatch, "Expected a scalar integer constant")
        }
        switch expression.kind {
        case .literal(.i32(let value)): return UInt32(bitPattern: value)
        case .literal(.u32(let value)): return value
        case .cast(let operand, _): return try integerConstantBits(operand, at: start)
        case .unary(.negate, let operand): return try 0 &- integerConstantBits(operand, at: start)
        case .unary(.bitwiseNot, let operand): return try ~integerConstantBits(operand, at: start)
        case .binary(let op, let lhs, let rhs):
            let left = try integerConstantBits(lhs, at: start)
            let right = try integerConstantBits(rhs, at: start)
            let signed = type == .scalar(.i32)
            switch op {
            case .add: return left &+ right
            case .subtract: return left &- right
            case .multiply: return left &* right
            case .and: return left & right
            case .inclusiveOr: return left | right
            case .exclusiveOr: return left ^ right
            case .shiftLeft, .shiftRight:
                guard right < 32 else { throw glslError(start, .invalidType, "Invalid constant shift") }
                if op == .shiftLeft { return left << right }
                return signed ? UInt32(bitPattern: Int32(bitPattern: left) >> right) : left >> right
            case .divide, .modulo:
                guard right != 0 else { throw glslError(start, .invalidType, "Constant division by zero") }
                if signed {
                    let a = Int64(Int32(bitPattern: left)), b = Int64(Int32(bitPattern: right))
                    return UInt32(truncatingIfNeeded: op == .divide ? a / b : a % b)
                }
                return op == .divide ? left / right : left % right
            default: break
            }
        default: break
        }
        throw glslError(start, .invalidType, "Switch label or const initializer requires a constant expression")
    }

    /// RHS evaluation and side effects occur only in the selected block, not eagerly.
    func shortCircuit(_ lhs:GLSLValue,op:GLSLToken,precedence:Int) throws -> GLSLValue {
        let c=try requireContext();let l=try rvalue(lhs,at:op);try requireBool(l,at:op)
        try budget.consume(.nodes);let local=try IRHandle<IRLocalVariable>(index:c.locals.count)
        c.locals.append(.init(type:l.type))
        let pointer=try expressionNode(addType(.pointer(base:l.type,space:.function)),.localVariable(local),at:op,emit:false,pointer:true,writable:true)
        try append(.store(pointer:pointer.expression,value:l.expression),at:op)
        let outer=c.current;c.current=[]
        let right=try rvalue(expression(precedence+1),at:op);try requireBool(right,at:op)
        try append(.store(pointer:pointer.expression,value:right.expression),at:op)
        let rhs=try block(c.current);c.current=outer
        let empty=try block([])
        try append(.ifCondition(condition:l.expression,accept:op.text=="&&" ? rhs:empty,reject:op.text=="&&" ? empty:rhs),at:op)
        return try rvalue(pointer,at:op)
    }
}
