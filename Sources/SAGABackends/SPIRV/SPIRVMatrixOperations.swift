import SAGAShaderIR

extension IRTypeKind {
    var isSPIRVMatrix: Bool {
        if case .matrix = self { return true }
        return false
    }
}

extension SPIRVBinaryWriter {
    mutating func writeMatrixFloatCast(
        _ value: UInt32, columns: IRVectorSize, rows: IRVectorSize, source: IRScalar, target: IRScalar,
        type: IRHandle<IRType>
    ) throws -> UInt32 {
        let sourceColumn = try vectorTypeID(size: rows, scalar: source)
        let targetColumn = try vectorTypeID(size: rows, scalar: target)
        var converted: [UInt32] = []
        for index in 0..<UInt32(columns.rawValue) {
            let column = try rawValueInstruction(81, type: sourceColumn, operands: [value, index])
            converted.append(try rawValueInstruction(115, type: targetColumn, operands: [column])) // OpFConvert
        }
        return try valueInstruction(80, type: type, operands: converted)
    }

    mutating func writeMatrixBinary(
        _ operation: IRBinaryOperator, lhs: (id: UInt32, kind: IRTypeKind), rhs: (id: UInt32, kind: IRTypeKind),
        type: IRHandle<IRType>
    ) throws -> UInt32 {
        switch (operation, lhs.kind, rhs.kind) {
        case (.multiply, .matrix, .matrix):
            return try valueInstruction(146, type: type, operands: [lhs.id, rhs.id])
        case (.multiply, .matrix, .vector):
            return try valueInstruction(145, type: type, operands: [lhs.id, rhs.id])
        case (.multiply, .vector, .matrix):
            return try valueInstruction(144, type: type, operands: [lhs.id, rhs.id])
        case (.multiply, .matrix, .scalar):
            return try valueInstruction(143, type: type, operands: [lhs.id, rhs.id])
        case (.multiply, .scalar, .matrix):
            return try valueInstruction(143, type: type, operands: [rhs.id, lhs.id])
        case (.add, .matrix(let columns, let rows, let scalar), .matrix),
             (.subtract, .matrix(let columns, let rows, let scalar), .matrix):
            let columnType = try vectorTypeID(size: rows, scalar: scalar)
            let opcode: UInt32 = operation == .add ? 129 : 131 // OpFAdd / OpFSub
            var result: [UInt32] = []
            for index in 0..<UInt32(columns.rawValue) {
                let left = try rawValueInstruction(81, type: columnType, operands: [lhs.id, index])
                let right = try rawValueInstruction(81, type: columnType, operands: [rhs.id, index])
                result.append(try rawValueInstruction(opcode, type: columnType, operands: [left, right]))
            }
            return try valueInstruction(80, type: type, operands: result) // OpCompositeConstruct
        default: throw unsupported("does not lower this matrix operator")
        }
    }

    mutating func writeMatrixColumn(
        _ matrix: UInt32, index: UInt32, signed: Bool, columns: IRVectorSize, rows: IRVectorSize,
        type: IRHandle<IRType>
    ) throws -> UInt32 {
        // SPIR-V has no dynamic matrix extraction. Select a fixed column with a vector condition.
        let columnType = try typeID(type)
        let bool = try scalarTypeID(.bool)
        let vector = try vectorTypeID(size: rows, scalar: .bool)
        var result = try zeroID(typeID: columnType)
        for position in 0..<UInt32(columns.rawValue) {
            let offset = try signed ? literalID(.i32(Int32(position))) : unsignedConstantID(position)
            let matches = try rawValueInstruction(170, type: bool, operands: [index, offset]) // OpIEqual
            let test = try rawValueInstruction(80, type: vector, operands: Array(repeating: matches, count: Int(rows.rawValue)))
            let column = try rawValueInstruction(81, type: columnType, operands: [matrix, position])
            result = try rawValueInstruction(169, type: columnType, operands: [test, column, result]) // OpSelect
        }
        return result
    }
}
