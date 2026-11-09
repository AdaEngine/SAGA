import SAGAShaderIR

extension WGSLParser {
    func subgroupElect(at name: WGSLToken) throws -> WGSLValue {
        try requireSubgroups(at: name)
        guard !(try localConstantNameExists(name)), try arguments().isEmpty else {
            throw wgslError(.typeMismatch, "Subgroup election does not take operands", name)
        }
        // The first active lane is the one with no active predecessors.
        let type = try intern(.scalar(.u32))
        let one = try expressionNode(.literal(.u32(1)), type: type, at: name, emit: false)
        let zero = try expressionNode(.literal(.u32(0)), type: type, at: name, emit: false)
        let preceding = try expressionNode(.subgroupOperationResult(type: type), type: type, at: name, emit: false)
        try statement(.subgroupCollectiveOperation(op: .add, collective: .exclusiveScan,
                                                   argument: one.handle, result: preceding.handle), at: name)
        return try expressionNode(.binary(op: .equal, left: preceding.handle, right: zero.handle),
                                  type: intern(.scalar(.bool)), at: name)
    }

    func subgroupBallot(at name: WGSLToken) throws -> WGSLValue {
        try requireSubgroups(at: name)
        let args = try arguments()
        guard !(try localConstantNameExists(name)), args.count <= 1,
              args.first.map({ types[$0.type.index].kind == .scalar(.bool) }) ?? true else {
            throw wgslError(.typeMismatch, "Subgroup ballot requires a bool predicate", name)
        }
        let type = try intern(.vector(size: .four, scalar: .u32))
        let result = try expressionNode(.subgroupBallotResult, type: type, at: name, emit: false)
        try statement(.subgroupBallot(result: result.handle, predicate: args.first?.handle), at: name)
        return result
    }

    func subgroupCollective(
        _ operation: IRSubgroupOperation, collective: IRCollectiveOperation, at name: WGSLToken
    ) throws -> WGSLValue {
        try requireSubgroups(at: name)
        let args = try arguments()
        guard !(try localConstantNameExists(name)), args.count == 1, let first = args.first else {
            throw wgslError(.typeMismatch, "Subgroup collective requires one operand", name)
        }
        let argument = try defaultMaterialize(first, at: name)
        let operand = try operation.isBitwise ? argument :
            subgroupWorkingOperand(argument, useUnsignedArithmetic: operation == .add || operation == .mul, at: name)
        guard operation.accepts(types[operand.type.index].kind, collective: collective) else {
            throw wgslError(.typeMismatch, "Invalid subgroup collective operand type", name)
        }
        let result = try expressionNode(.subgroupOperationResult(type: operand.type), type: operand.type,
                                        at: name, emit: false)
        try statement(.subgroupCollectiveOperation(op: operation, collective: collective,
                                                   argument: operand.handle, result: result.handle), at: name)
        return try subgroupSourceResult(result, type: argument.type, at: name)
    }
}
