import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func declareArrayValueSpills(in state: inout SPIRVFunctionState) throws {
        for expression in state.function.expressions.values {
            try charge()
            guard case .access(let base, _) = expression.kind else { continue }
            let type = try state.function.expressions.value(for: base).type
            guard state.arrayValueSpills[type] == nil,
                  case .array(_, .constant, _) = try module.types.value(for: type).kind else { continue }
            let pointer = try pointerTypeID(storageClass: 7, pointee: typeID(type))
            let spill = try allocateID()
            try emit(59, [pointer, spill, 7]) // OpVariable Function
            state.arrayValueSpills[type] = spill
        }
    }

    mutating func writeArrayValueAccess(
        _ array: UInt32, index: UInt32, signed: Bool, count: UInt32, spill: UInt32, type: IRHandle<IRType>
    ) throws -> UInt32 {
        let result = try typeID(type)
        if let position = constantWords[index] {
            guard position < count else { return try zeroID(typeID: result) }
            return try rawValueInstruction(81, type: result, operands: [array, position]) // OpCompositeExtract
        }
        let u32 = try scalarTypeID(.u32)
        let position = try signed ? rawValueInstruction(124, type: u32, operands: [index]) : index
        let valid = try rawValueInstruction(176, type: scalarTypeID(.bool),
                                            operands: [position, unsignedConstantID(count)]) // OpULessThan
        return try valueWhen(valid, type: result) { writer in
            // SPIR-V cannot extract an array value with a dynamic index. Copy the value to
            // function memory, then use an in-range pointer without touching the source buffer.
            try writer.emit(62, [spill, array]) // OpStore
            let pointer = try writer.pointerTypeID(storageClass: 7, pointee: result)
            let element = try writer.rawValueInstruction(65, type: pointer, operands: [spill, position])
            return try writer.rawValueInstruction(61, type: result, operands: [element])
        }
    }
}
