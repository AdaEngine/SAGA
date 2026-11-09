import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeWideBitField(
        _ builtin: IRMathFunction, arguments: [UInt32], shape: IRBitcastShape, type result: UInt32
    ) throws -> UInt32 {
        let insert = builtin == .insertBits
        let rangeIndex = insert ? 2 : 1
        let u32 = try scalarTypeID(.u32)
        let u64 = try scalarTypeID(.u64)
        let word = try scalarOrVectorTypeID(.u64, size: shape.size)
        let shiftType = try scalarOrVectorTypeID(.u32, size: shape.size)
        let width = try unsignedConstantID(64)
        let offset = try glslMathInstruction(38, type: u32, operands: [arguments[rangeIndex], width]) // UMin
        let remaining = try rawValueInstruction(130, type: u32, operands: [width, offset])
        let count = try glslMathInstruction(38, type: u32, operands: [arguments[rangeIndex + 1], remaining])

        // A full-width field needs an explicit all-ones mask. Every shift uses a count below 64.
        let shiftMask = try unsignedConstantID(63)
        let offsetBits = try rawValueInstruction(199, type: u32, operands: [offset, shiftMask])
        let amount = try splatBitFieldValue(offsetBits, type: shiftType, size: shape.size)
        let countBits = try rawValueInstruction(199, type: u32, operands: [count, shiftMask])
        let one = try literalID(.u64(1))
        let power = try rawValueInstruction(196, type: u64, operands: [one, countBits])
        let partialMask = try rawValueInstruction(130, type: u64, operands: [power, one])
        let fullWidth = try rawValueInstruction(170, type: scalarTypeID(.bool), operands: [count, width])
        let fullMask = try literalID(.u64(UInt64.max))
        let lowMask = try rawValueInstruction(169, type: u64, operands: [fullWidth, fullMask, partialMask])
        let mask = try splatBitFieldValue(lowMask, type: word, size: shape.size)

        let signed = shape.scalar == .i64
        let base = try signed ? rawValueInstruction(124, type: word, operands: [arguments[0]]) : arguments[0]
        let value: UInt32
        if insert {
            let payload = try signed ? rawValueInstruction(124, type: word, operands: [arguments[1]]) : arguments[1]
            let fieldMask = try rawValueInstruction(196, type: word, operands: [mask, amount])
            let keepMask = try rawValueInstruction(200, type: word, operands: [fieldMask])
            let preserved = try rawValueInstruction(199, type: word, operands: [base, keepMask])
            let lowPayload = try rawValueInstruction(199, type: word, operands: [payload, mask])
            let shiftedPayload = try rawValueInstruction(196, type: word, operands: [lowPayload, amount])
            value = try rawValueInstruction(197, type: word, operands: [preserved, shiftedPayload])
        } else {
            let shifted = try rawValueInstruction(194, type: word, operands: [base, amount])
            let extracted = try rawValueInstruction(199, type: word, operands: [shifted, mask])
            if signed {
                // Flipping and subtracting the sign bit extends it, including an empty field (zero).
                let last = try rawValueInstruction(130, type: u32, operands: [count, unsignedConstantID(1)])
                let signIndex = try rawValueInstruction(199, type: u32, operands: [last, shiftMask])
                let signBit = try rawValueInstruction(196, type: u64, operands: [one, signIndex])
                let sign = try splatBitFieldValue(signBit, type: word, size: shape.size)
                let flipped = try rawValueInstruction(198, type: word, operands: [extracted, sign])
                value = try rawValueInstruction(130, type: word, operands: [flipped, sign])
            } else {
                value = extracted
            }
        }
        return try signed ? rawValueInstruction(124, type: result, operands: [value]) : value
    }

    private mutating func splatBitFieldValue(_ value: UInt32, type: UInt32, size: IRVectorSize?) throws -> UInt32 {
        guard let size else { return value }
        return try rawValueInstruction(80, type: type, operands: Array(repeating: value, count: Int(size.rawValue)))
    }
}
