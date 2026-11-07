import SAGAShaderIR

extension GLSLParser {
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
