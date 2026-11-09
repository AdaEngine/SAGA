import SAGAShaderIR

enum SPVMatrixMemoryLayout {
    static let rowMajor: UInt32 = 4
    static let columnMajor: UInt32 = 5
    static let stride: UInt32 = 7
}

extension SPVDecoder {
    mutating func validateMatrixMemberLayouts(_ instruction: SPVInstruction) throws -> Bool {
        let decorations = members[instruction.a[0]] ?? [:]
        var complete = true
        for (index, id) in instruction.a.dropFirst().enumerated() {
            try charge()
            let member = decorations[UInt32(index)] ?? [:]
            let rowMajor = member[SPVMatrixMemoryLayout.rowMajor] != nil
            let columnMajor = member[SPVMatrixMemoryLayout.columnMajor] != nil
            let stride = member[SPVMatrixMemoryLayout.stride]?.first
            var element = try type(id, instruction)
            while case .array(let base, _, _) = types[element.index].kind {
                try charge()
                element = base
            }
            guard case .matrix = types[element.index].kind else {
                guard !rowMajor, !columnMajor, stride == nil else {
                    throw instruction.fail("Matrix layout decorations require a matrix or array of matrices")
                }
                continue
            }
            if !rowMajor, !columnMajor, stride == nil {
                complete = false
                continue
            }
            guard !rowMajor, columnMajor, let stride,
                  UInt64(stride) == (try valueLayout(element, instruction)).alignment else {
                throw instruction.fail("Matrix memory layout requires ColMajor and the natural column stride",
                                       code: .unsupportedConstruct)
            }
        }
        return complete
    }
}
