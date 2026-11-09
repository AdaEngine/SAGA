import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeSubgroupBallot(
        predicate: IRHandle<IRExpression>?, result: IRHandle<IRExpression>, in state: inout SPIRVFunctionState
    ) throws {
        let scope = try subgroupScope(requiring: 64) // GroupNonUniformBallot
        let vote = try predicate.map { try value($0, in: &state) } ?? literalID(.bool(true))
        let type = try state.function.expressions.value(for: result).type
        let id = try valueInstruction(339, type: type, operands: [scope, vote]) // OpGroupNonUniformBallot
        state.bind(result, to: id)
    }

    mutating func writeSubgroupGather(
        _ mode: IRGatherMode, argument: IRHandle<IRExpression>, result: IRHandle<IRExpression>,
        in state: inout SPIRVFunctionState
    ) throws {
        let (opcode, capability): (UInt32, UInt32) = switch mode {
        case .broadcastFirst: (338, 64)
        case .broadcast: (337, 64)
        case .shuffle: (345, 65)
        case .shuffleXor: (346, 65)
        case .shuffleUp: (347, 66)
        case .shuffleDown: (348, 66)
        case .quadBroadcast, .quadBroadcastDynamic: (365, 68)
        case .quadSwap: (366, 68)
        }
        let scope = try subgroupScope(requiring: capability)
        let type = try state.function.expressions.value(for: result).type
        let argumentID = try value(argument, in: &state)
        let id: UInt32
        switch mode {
        case .quadBroadcastDynamic(let index):
            let selector = try value(index, in: &state)
            id = try writeDynamicQuadBroadcast(argumentID, index: selector, scope: scope, type: type)
        case .quadSwap(let direction):
            let lane: UInt32 = switch direction {
            case .x: 0
            case .y: 1
            case .diagonal: 2
            }
            id = try valueInstruction(opcode, type: type, operands: [scope, argumentID, unsignedConstantID(lane)])
        default:
            var operands = [scope, argumentID]
            if let index = mode.index { operands.append(try subgroupSelector(index, in: &state)) }
            id = try valueInstruction(opcode, type: type, operands: operands)
        }
        state.bind(result, to: id)
    }

    mutating func writeSubgroupCollective(
        _ operation: IRSubgroupOperation, _ collective: IRCollectiveOperation,
        argument: IRHandle<IRExpression>, result: IRHandle<IRExpression>, in state: inout SPIRVFunctionState
    ) throws {
        let type = try state.function.expressions.value(for: result).type
        guard let shape = IRBitcastShape(try module.types.value(for: type).kind) else {
            throw ShaderFailure.invalidHandle
        }
        let scope = try subgroupScope(requiring: operation.isVote ? 62 : 63)
        let argumentID = try value(argument, in: &state)
        let opcode: UInt32 = switch (operation, shape.scalar.kind) {
        case (.all, _): 334
        case (.any, _): 335
        case (.add, .float): 350
        case (.add, _): 349
        case (.mul, .float): 352
        case (.mul, _): 351
        case (.min, .sint): 353
        case (.min, .uint): 354
        case (.min, _): 355
        case (.max, .sint): 356
        case (.max, .uint): 357
        case (.max, _): 358
        case (.and, _): 359
        case (.or, _): 360
        case (.xor, _): 361
        }
        let groupOperation: UInt32 = switch collective {
        case .reduce: 0
        case .inclusiveScan: 1
        case .exclusiveScan: 2
        }
        let operands = operation.isVote ? [scope, argumentID] : [scope, groupOperation, argumentID]
        let id = try valueInstruction(opcode, type: type, operands: operands)
        state.bind(result, to: id)
    }

    private mutating func subgroupScope(requiring capability: UInt32) throws -> UInt32 {
        try requireCapability(61) // GroupNonUniform
        try requireCapability(capability)
        return try unsignedConstantID(3) // Subgroup scope
    }

    private mutating func subgroupSelector(
        _ index: IRHandle<IRExpression>, in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let id = try value(index, in: &state)
        let type = try state.function.expressions.value(for: index).type
        guard try module.types.value(for: type).kind == .scalar(.i32) else { return id }
        // Constant broadcast selectors must remain constants in SPIR-V 1.3.
        if let bits = constantWords[id] { return try unsignedConstantID(bits) }
        return try rawValueInstruction(124, type: scalarTypeID(.u32), operands: [id]) // OpBitcast
    }

    private mutating func writeDynamicQuadBroadcast(
        _ argument: UInt32, index: UInt32, scope: UInt32, type: IRHandle<IRType>
    ) throws -> UInt32 {
        guard let shape = IRBitcastShape(try module.types.value(for: type).kind) else {
            throw ShaderFailure.invalidHandle
        }
        let bool = try scalarTypeID(.bool)
        let predicateType = try scalarOrVectorTypeID(.bool, size: shape.size)
        // SPIR-V 1.3 requires constant quad indices. Keep the reads in the same
        // control flow so all participating invocations execute each broadcast.
        var selected = try valueInstruction(365, type: type, operands: [scope, argument, unsignedConstantID(0)])
        for lane in 1..<UInt32(4) {
            let constant = try unsignedConstantID(lane)
            let broadcast = try valueInstruction(365, type: type, operands: [scope, argument, constant])
            var test = try rawValueInstruction(170, type: bool, operands: [index, constant]) // OpIEqual
            if let size = shape.size {
                test = try rawValueInstruction(80, type: predicateType, operands: Array(repeating: test, count: Int(size.rawValue)))
            }
            selected = try valueInstruction(169, type: type, operands: [test, broadcast, selected]) // OpSelect
        }
        return selected
    }
}
