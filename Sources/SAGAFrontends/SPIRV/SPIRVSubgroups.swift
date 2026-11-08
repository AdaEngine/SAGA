import SAGAShaderIR

enum SPVSubgroup {
    static let capabilities: Set<UInt32> = [61, 62, 63, 64, 65, 66, 68]

    static func capability(for opcode: UInt32) -> UInt32? {
        switch opcode {
        case 334, 335: 62 // GroupNonUniformVote
        case 349...364: 63 // GroupNonUniformArithmetic
        case 337...339: 64 // GroupNonUniformBallot
        case 345, 346: 65 // GroupNonUniformShuffle
        case 347, 348: 66 // GroupNonUniformShuffleRelative
        case 365, 366: 68 // GroupNonUniformQuad
        default: nil
        }
    }
}

extension SPVFunctionLowerer {
    mutating func lowerSubgroup(
        _ instruction: SPVInstruction, capability: UInt32, _ statements: inout [IRHandle<IRStatement>]
    ) throws {
        let opcode = instruction.op
        let a = instruction.a
        try instruction.count([334, 335, 338, 339].contains(opcode) ? 4 : 5)
        guard module.capabilities.contains(capability) else {
            throw instruction.fail("Subgroup opcode \(opcode) requires capability \(capability)", code: .unsupportedCapability)
        }
        guard module.constantMap[a[2]]?.nonnegativeInteger == 3 else {
            throw instruction.fail("Subgroup operation requires Subgroup execution scope", code: .unsupportedConstruct)
        }
        let type = try module.type(a[0], instruction)
        let kind = module.types[type.index].kind
        let result: IRHandle<IRExpression>
        let statement: IRStatementKind
        switch opcode {
        case 362...364:
            try lowerLogicalSubgroup(instruction, type: type, &statements)
            return
        case 339:
            let predicate = try value(a[3], instruction, &statements)
            guard kind == .vector(size: .four, scalar: .u32),
                  try module.kind(predicate.type, instruction) == .scalar(.bool) else {
                throw instruction.fail("Subgroup ballot requires a bool predicate and vec4<u32> result", code: .typeMismatch)
            }
            result = try expr(type, .subgroupBallotResult, instruction)
            statement = .subgroupBallot(result: result, predicate: predicate.expression)
        case 334, 335, 349...361:
            let vote = opcode == 334 || opcode == 335
            let (operation, collective) = try subgroupCollective(instruction, type: kind)
            let argument = try typed(a[vote ? 3 : 4], a[0], instruction, &statements)
            result = try expr(type, .subgroupOperationResult(type: type), instruction)
            statement = .subgroupCollectiveOperation(op: operation, collective: collective, argument: argument, result: result)
        default:
            let argument = try typed(a[3], a[0], instruction, &statements)
            let mode = try subgroupGatherMode(instruction, &statements)
            result = try expr(type, .subgroupOperationResult(type: type), instruction)
            statement = .subgroupGather(mode: mode, argument: argument, result: result)
        }
        let anchor = try stmt(statement, instruction)
        statements.append(anchor)
        defineValue(a[1], .init(expression: result, type: a[0], anchor: anchor))
    }

    private func subgroupCollectiveMode(_ instruction: SPVInstruction) throws -> IRCollectiveOperation {
        switch instruction.a[3] {
        case 0: .reduce
        case 1: .inclusiveScan
        case 2: .exclusiveScan
        default:
            throw instruction.fail("Unsupported subgroup group operation", code: .unsupportedConstruct)
        }
    }

    private func subgroupCollective(
        _ instruction: SPVInstruction, type: IRTypeKind
    ) throws -> (IRSubgroupOperation, IRCollectiveOperation) {
        let opcode = instruction.op
        let collective: IRCollectiveOperation = opcode == 334 || opcode == 335
            ? .reduce : try subgroupCollectiveMode(instruction)
        let scalar: IRScalar
        switch type {
        case .scalar(let value), .vector(_, let value): scalar = value
        default: throw instruction.fail("Expected a subgroup scalar or vector", code: .typeMismatch)
        }
        let operation: IRSubgroupOperation
        switch opcode {
        case 334: operation = .all
        case 335: operation = .any
        case 349 where scalar.kind == .sint || scalar.kind == .uint,
             350 where scalar.kind == .float: operation = .add
        case 351 where scalar.kind == .sint || scalar.kind == .uint,
             352 where scalar.kind == .float: operation = .mul
        case 353 where scalar.kind == .sint, 354 where scalar.kind == .uint,
             355 where scalar.kind == .float: operation = .min
        case 356 where scalar.kind == .sint, 357 where scalar.kind == .uint,
             358 where scalar.kind == .float: operation = .max
        case 359: operation = .and
        case 360: operation = .or
        case 361: operation = .xor
        default: throw instruction.fail("Subgroup opcode and scalar type are incompatible", code: .typeMismatch)
        }
        guard operation.wgslName(collective) != nil else {
            throw instruction.fail("Only subgroup add and multiply support scans", code: .unsupportedConstruct)
        }
        guard operation.accepts(type, collective: collective) else {
            throw instruction.fail("Invalid subgroup collective type", code: .typeMismatch)
        }
        return (operation, collective)
    }

    private mutating func lowerLogicalSubgroup(
        _ instruction: SPVInstruction, type: IRHandle<IRType>, _ statements: inout [IRHandle<IRStatement>]
    ) throws {
        let collective = try subgroupCollectiveMode(instruction)
        let size: IRVectorSize?
        switch module.types[type.index].kind {
        case .scalar(.bool): size = nil
        case .vector(let width, .bool): size = width
        default:
            throw instruction.fail("Logical subgroup operation requires a Boolean scalar or vector", code: .typeMismatch)
        }
        let a = instruction.a
        let predicate = try typed(a[4], a[0], instruction, &statements)
        let scalarType = try module.internType(.scalar(.u32))
        var wordType = scalarType
        var zero = try expr(scalarType, .literal(.u32(0)), instruction)
        var one = try expr(scalarType, .literal(.u32(1)), instruction)
        if let size {
            wordType = try module.internType(.vector(size: size, scalar: .u32))
            zero = try emitValue(wordType, .splat(size: size, value: zero), &statements, instruction)
            one = try emitValue(wordType, .splat(size: size, value: one), &statements, instruction)
        }

        // Count false lanes for AND and true lanes for OR and XOR. The zero
        // sum of an empty exclusive prefix gives the appropriate identity.
        let countsFalse = instruction.op == 362
        let countable = try emitValue(wordType, .select(condition: predicate, accept: countsFalse ? zero : one,
                                                         reject: countsFalse ? one : zero), &statements, instruction)
        let sum = try expr(wordType, .subgroupOperationResult(type: wordType), instruction)
        statements.append(try stmt(.subgroupCollectiveOperation(op: .add, collective: collective,
                                                                 argument: countable, result: sum), instruction))
        let count = instruction.op == 364
            ? try emitValue(wordType, .binary(op: .and, left: sum, right: one), &statements, instruction)
            : sum
        try result(instruction, .binary(op: countsFalse ? .equal : .notEqual, left: count, right: zero), &statements)
    }

    private mutating func subgroupGatherMode(
        _ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRGatherMode {
        let opcode = instruction.op
        if opcode == 338 { return .broadcastFirst }
        let selectorID = instruction.a[4]
        let selector = try value(selectorID, instruction, &statements)
        guard try module.kind(selector.type, instruction) == .scalar(.u32) else {
            throw instruction.fail("Subgroup selector must be u32", code: .typeMismatch)
        }
        let constant = module.constantMap[selectorID]?.nonnegativeInteger
        if opcode == 366 {
            switch constant {
            case 0: return .quadSwap(.x)
            case 1: return .quadSwap(.y)
            case 2: return .quadSwap(.diagonal)
            default: throw instruction.fail("Quad swap direction must be a constant in 0...2")
            }
        }
        let index: IRHandle<IRExpression>
        if let constant {
            let limit: UInt32 = opcode == 365 ? 4 : 128
            guard constant < limit else { throw instruction.fail("Subgroup selector is out of range") }
            index = try expr(module.type(selector.type, instruction), .literal(.u32(constant)), instruction)
        } else {
            if opcode == 337 || opcode == 365 {
                guard module.version >= 0x00010500 else {
                    throw instruction.fail("Dynamic broadcast selectors require SPIR-V 1.5", code: .unsupportedConstruct)
                }
                // A shuffle has the same value as broadcast when the source lane is dynamically uniform.
                return opcode == 337 ? .shuffle(selector.expression) : .quadBroadcastDynamic(selector.expression)
            }
            index = selector.expression
        }
        switch opcode {
        case 337: return .broadcast(index)
        case 345: return .shuffle(index)
        case 346: return .shuffleXor(index)
        case 347: return .shuffleUp(index)
        case 348: return .shuffleDown(index)
        case 365: return .quadBroadcast(index)
        default: throw instruction.fail("Unsupported subgroup gather opcode")
        }
    }
}
