import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeIntegerShift(
        _ operation: IRBinaryOperator, base: UInt32, count: UInt32, shape: IRBitcastShape,
        type: IRHandle<IRType>
    ) throws -> UInt32 {
        let word = try scalarOrVectorTypeID(.u32, size: shape.size)
        let mask = try splatLiteralID(.u32(UInt32(shape.scalar.width) * 8 - 1), type: word, size: shape.size)
        let amount = try rawValueInstruction(199, type: word, operands: [count, mask]) // OpBitwiseAnd
        let opcode: UInt32
        if operation == .shiftLeft { opcode = 196 } // OpShiftLeftLogical
        else { opcode = shape.scalar.kind == .sint ? 195 : 194 } // OpShiftRightArithmetic / Logical
        return try valueInstruction(opcode, type: type, operands: [base, amount])
    }

    mutating func writeIntegerBitBuiltin(
        _ builtin: IRMathFunction, arguments: [UInt32], shape: IRBitcastShape, type: IRHandle<IRType>
    ) throws -> UInt32 {
        guard let arity = builtin.integerBitArity, arguments.count == arity else { throw ShaderFailure.invalidHandle }
        let result = try typeID(type)
        if shape.scalar.width == 4 {
            return try writeBitWord(builtin, arguments: arguments, shape: shape, width: 32, type: result)
        }
        if shape.scalar.width == 8 {
            if arity == 1 { return try writeWideBitUnary(builtin, operand: arguments[0], shape: shape, type: result) }
            return try writeWideBitField(builtin, arguments: arguments, shape: shape, type: result)
        }

        // Vulkan's bitfield, reverse, count and find operations use 32-bit words.
        // Sign-extend only when the operation observes signedness; otherwise preserve the low 16 bits.
        let signed = shape.scalar == .i16 && (builtin == .firstLeadingBit || builtin == .extractBits)
        let working = IRBitcastShape(size: shape.size, scalar: signed ? .i32 : .u32)
        let word = try scalarOrVectorTypeID(working.scalar, size: shape.size)
        let widen: UInt32 = signed ? 114 : 113 // OpSConvert / OpUConvert
        var widened = arguments
        widened[0] = try rawValueInstruction(widen, type: word, operands: [arguments[0]])
        if builtin == .insertBits {
            widened[1] = try rawValueInstruction(widen, type: word, operands: [arguments[1]])
        }
        let value = try writeBitWord(builtin, arguments: widened, shape: working, width: 16, type: word)
        return try rawValueInstruction(shape.scalar == .i16 ? 114 : 113, type: result, operands: [value])
    }

    private mutating func writeWideBitUnary(
        _ builtin: IRMathFunction, operand: UInt32, shape: IRBitcastShape, type result: UInt32
    ) throws -> UInt32 {
        let wide = try scalarOrVectorTypeID(.u64, size: shape.size)
        let word = try scalarOrVectorTypeID(.u32, size: shape.size)
        let half = try splatLiteralID(.u32(32), type: word, size: shape.size)
        var bits = try shape.scalar == .i64 ? rawValueInstruction(124, type: wide, operands: [operand]) : operand
        if shape.scalar == .i64, builtin == .firstLeadingBit {
            let last = try splatLiteralID(.u32(63), type: word, size: shape.size)
            let negative = try rawValueInstruction(194, type: wide, operands: [bits, last])
            let mask = try rawValueInstruction(130, type: wide, operands: [zeroID(typeID: wide), negative])
            bits = try rawValueInstruction(198, type: wide, operands: [bits, mask]) // Complement negative inputs.
        }
        let low = try rawValueInstruction(113, type: word, operands: [bits])
        let upperBits = try rawValueInstruction(194, type: wide, operands: [bits, half])
        let high = try rawValueInstruction(113, type: word, operands: [upperBits])
        let wordShape = IRBitcastShape(size: shape.size, scalar: .u32)
        let lowResult = try writeBitWord(builtin, arguments: [low], shape: wordShape, width: 32, type: word)
        let highResult = try writeBitWord(builtin, arguments: [high], shape: wordShape, width: 32, type: word)
        let value: UInt32
        if builtin == .reverseBits {
            let lower = try rawValueInstruction(113, type: wide, operands: [highResult])
            let upper = try rawValueInstruction(113, type: wide, operands: [lowResult])
            let shifted = try rawValueInstruction(196, type: wide, operands: [upper, half])
            value = try rawValueInstruction(197, type: wide, operands: [lower, shifted])
        } else {
            let combined: UInt32
            switch builtin {
            case .countOneBits:
                combined = try rawValueInstruction(128, type: word, operands: [lowResult, highResult])
            case .countLeadingZeros:
                let continued = try rawValueInstruction(128, type: word, operands: [lowResult, half])
                combined = try selectForNonzeroWord(high, accept: highResult, reject: continued, shape: wordShape, type: word)
            case .countTrailingZeros:
                let continued = try rawValueInstruction(128, type: word, operands: [highResult, half])
                combined = try selectForNonzeroWord(low, accept: lowResult, reject: continued, shape: wordShape, type: word)
            case .firstLeadingBit:
                let continued = try rawValueInstruction(128, type: word, operands: [highResult, half])
                combined = try selectForNonzeroWord(high, accept: continued, reject: lowResult, shape: wordShape, type: word)
            case .firstTrailingBit:
                let continued = try rawValueInstruction(128, type: word, operands: [highResult, half])
                let upper = try selectForNonzeroWord(high, accept: continued, reject: highResult, shape: wordShape, type: word)
                combined = try selectForNonzeroWord(low, accept: lowResult, reject: upper, shape: wordShape, type: word)
            default: throw unsupported("does not lower this wide integer bit builtin")
            }
            // Sign extension preserves the all-ones search sentinel at 64 bits.
            let search = builtin == .firstLeadingBit || builtin == .firstTrailingBit
            value = try rawValueInstruction(search ? 114 : 113, type: wide, operands: [combined])
        }
        return try shape.scalar == .i64 ? rawValueInstruction(124, type: result, operands: [value]) : value
    }

    private mutating func selectForNonzeroWord(
        _ value: UInt32, accept: UInt32, reject: UInt32, shape: IRBitcastShape, type: UInt32
    ) throws -> UInt32 {
        let boolean = try scalarOrVectorTypeID(.bool, size: shape.size)
        let nonzero = try rawValueInstruction(171, type: boolean, operands: [value, zeroID(typeID: type)])
        return try rawValueInstruction(169, type: type, operands: [nonzero, accept, reject])
    }

    private mutating func writeBitWord(
        _ builtin: IRMathFunction, arguments: [UInt32], shape: IRBitcastShape, width: UInt32, type result: UInt32
    ) throws -> UInt32 {
        let operand = arguments[0]
        let signed = shape.scalar == .i32
        switch builtin {
        case .countOneBits:
            return try rawValueInstruction(205, type: result, operands: [operand]) // OpBitCount
        case .reverseBits:
            let reversed = try rawValueInstruction(204, type: result, operands: [operand]) // OpBitReverse
            if width == 32 { return reversed }
            let amount = try splatLiteralID(.u32(16), type: result, size: shape.size)
            return try rawValueInstruction(194, type: result, operands: [reversed, amount]) // OpShiftRightLogical
        case .extractBits, .insertBits:
            let insert = builtin == .insertBits
            let range = insert ? 2 : 1
            let u32 = try scalarTypeID(.u32)
            let bool = try scalarTypeID(.bool)
            let limit = try unsignedConstantID(width)
            let offset = try unsignedMinimum(arguments[range], limit, type: u32, predicate: bool)
            let remaining = try rawValueInstruction(130, type: u32, operands: [limit, offset])
            let count = try unsignedMinimum(arguments[range + 1], remaining, type: u32, predicate: bool)
            let opcode: UInt32 = insert ? 201 : signed ? 202 : 203
            return try rawValueInstruction(opcode, type: result, operands: Array(arguments.prefix(range)) + [offset, count])
        case .firstLeadingBit:
            let find: UInt32 = signed ? 74 : 75 // FindSMsb / FindUMsb
            return try rawValueInstruction(12, type: result, operands: [glslInstructionSetID(), find, operand])
        case .firstTrailingBit, .countLeadingZeros, .countTrailingZeros:
            let word = try scalarOrVectorTypeID(.u32, size: shape.size)
            let bits = try signed ? rawValueInstruction(124, type: word, operands: [operand]) : operand
            let find: UInt32 = builtin == .countLeadingZeros ? 75 : 73 // FindUMsb / FindILsb
            let first = try rawValueInstruction(12, type: word, operands: [glslInstructionSetID(), find, bits])
            let value: UInt32
            if builtin == .countLeadingZeros {
                // FindUMsb returns all ones for zero, so subtracting it wraps to the word width.
                let last = try splatLiteralID(.u32(width - 1), type: word, size: shape.size)
                value = try rawValueInstruction(130, type: word, operands: [last, first])
            } else if builtin == .countTrailingZeros {
                let limit = try splatLiteralID(.u32(width), type: word, size: shape.size)
                let predicate = try scalarOrVectorTypeID(.bool, size: shape.size)
                value = try unsignedMinimum(first, limit, type: word, predicate: predicate)
            } else {
                value = first
            }
            return try signed ? rawValueInstruction(124, type: result, operands: [value]) : value
        default: throw unsupported("does not lower this integer bit builtin")
        }
    }

    private mutating func unsignedMinimum(_ value: UInt32, _ limit: UInt32, type: UInt32, predicate: UInt32) throws -> UInt32 {
        let above = try rawValueInstruction(172, type: predicate, operands: [value, limit])
        return try rawValueInstruction(169, type: type, operands: [above, limit, value])
    }
}
