extension SPIRVBinaryWriter {
    mutating func writePackedDot(
        _ left: UInt32, _ right: UInt32, signed: Bool, type result: UInt32
    ) throws -> UInt32 {
        // Each packed operand contains four 8-bit lanes. The products and sum fit in 32 bits.
        var sum = try packedLaneProduct(left, right, index: 0, signed: signed, type: result)
        for lane in 1..<UInt32(4) {
            let product = try packedLaneProduct(left, right, index: lane, signed: signed, type: result)
            sum = try rawValueInstruction(128, type: result, operands: [sum, product]) // OpIAdd
        }
        return sum
    }

    private mutating func packedLaneProduct(
        _ left: UInt32, _ right: UInt32, index: UInt32, signed: Bool, type result: UInt32
    ) throws -> UInt32 {
        let a = try packedIntegerLane(left, index: index, signed: signed)
        let b = try packedIntegerLane(right, index: index, signed: signed)
        return try rawValueInstruction(132, type: result, operands: [a, b]) // OpIMul
    }
}
