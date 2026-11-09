import SAGAShaderIR

extension WGSLParser {
    static func cooperativeMatrixShape(_ name: String) -> (IRCooperativeSize, IRCooperativeSize)? {
        guard name.hasPrefix("coop_mat") else { return nil }
        let dimensions = name.dropFirst("coop_mat".count).split(separator: "x")
        guard dimensions.count == 2, let columns = UInt8(dimensions[0]), let rows = UInt8(dimensions[1]),
              let c = IRCooperativeSize(rawValue: columns), let r = IRCooperativeSize(rawValue: rows) else { return nil }
        return (c, r)
    }
    static func isCooperativeMatrixName(_ name: String) -> Bool { cooperativeMatrixShape(name) != nil }
    static let cooperativeLoads: Set<String> = ["coopLoad", "coopLoadT"]
    static let cooperativeStores: Set<String> = ["coopStore", "coopStoreT"]

    func cooperativeMatrixType(_ columns: IRCooperativeSize, _ rows: IRCooperativeSize, at name: WGSLToken) throws -> WType {
        try requireCooperativeMatrix(at: name)
        try expect("<")
        let element = try type()
        guard case .scalar(let scalar) = types[element.index].kind, scalar == .f16 || scalar == .f32 else {
            throw wgslError(.invalidType, "Cooperative matrices require f16 or f32 elements", name)
        }
        try expect(",")
        let roleToken = try ident()
        let role: IRCooperativeRole
        switch roleToken.text {
        case "A": role = .a
        case "B": role = .b
        case "C": role = .c
        default: throw wgslError(.invalidType, "Expected cooperative matrix role A, B, or C", roleToken)
        }
        _ = try eat(",")
        try expect(">")
        return try intern(.cooperativeMatrix(columns: columns, rows: rows, scalar: scalar, role: role), at: name)
    }

    func cooperativeLoad(at name: WGSLToken) throws -> WGSLValue {
        try requireCooperativeMatrix(at: name)
        try expect("<")
        let matrix = try type()
        _ = try eat(",")
        try expect(">")
        guard case .cooperativeMatrix(let columns, let rows, let scalar, let role) = types[matrix.index].kind else {
            throw wgslError(.typeMismatch, "Cooperative load requires a cooperative matrix type", name)
        }
        try expect("(")
        let args = try arguments()
        guard args.count == 1 || args.count == 2 else {
            throw wgslError(.typeMismatch, "Cooperative load requires an element pointer and optional stride", name)
        }
        let rowMajor = name.text == "coopLoadT"
        let data = try cooperativeData(pointer: args[0], stride: args.count == 2 ? args[1] : nil,
                                       scalar: scalar, defaultStride: rowMajor ? columns : rows,
                                       rowMajor: rowMajor, access: .load, at: name)
        return try expressionNode(.cooperativeLoad(columns: columns, rows: rows, role: role, data: data),
                                  type: matrix, at: name)
    }

    func cooperativeMultiplyAdd(at name: WGSLToken) throws -> WGSLValue {
        try requireCooperativeMatrix(at: name)
        let args = try arguments()
        guard args.count == 3, case .cooperativeMatrix = types[args[2].type.index].kind else {
            throw wgslError(.typeMismatch, "coopMultiplyAdd requires matrices A, B, and C", name)
        }
        return try expressionNode(.cooperativeMultiplyAdd(a: args[0].handle, b: args[1].handle, c: args[2].handle),
                                  type: args[2].type, at: name)
    }

    func cooperativeStore(at name: WGSLToken, end: String) throws {
        try requireCooperativeMatrix(at: name)
        _ = try take()
        try expect("(")
        let args = try arguments()
        guard args.count == 2 || args.count == 3 else {
            throw wgslError(.typeMismatch, "Cooperative store requires a matrix, element pointer, and optional stride", name)
        }
        guard case .cooperativeMatrix(let columns, let rows, let scalar, _) = types[args[0].type.index].kind else {
            throw wgslError(.typeMismatch, "Cooperative store requires a cooperative matrix value", name)
        }
        let rowMajor = name.text == "coopStoreT"
        let data = try cooperativeData(pointer: args[1], stride: args.count == 3 ? args[2] : nil,
                                       scalar: scalar, defaultStride: rowMajor ? columns : rows,
                                       rowMajor: rowMajor, access: .store, at: name)
        try expect(end)
        try statement(.cooperativeStore(target: args[0].handle, data: data), at: name)
    }

    private func requireCooperativeMatrix(at name: WGSLToken) throws {
        guard cooperativeMatrixEnabled else {
            throw wgslError(.unsupportedCapability, "Cooperative matrices require enable wgpu_cooperative_matrix", name)
        }
    }

    private func cooperativeData(pointer: WGSLValue, stride: WGSLValue?, scalar: IRScalar,
                                 defaultStride: IRCooperativeSize, rowMajor: Bool, access: IRStorageAccess,
                                 at name: WGSLToken) throws -> IRCooperativeData {
        guard case .pointer(let pointee, let space) = types[pointer.type.index].kind,
              types[pointee.index].kind == .scalar(scalar) else {
            throw wgslError(.typeMismatch, "Cooperative matrix memory requires a matching scalar element pointer", name)
        }
        let available: IRStorageAccess
        switch space {
        case .storage(let mode): available = mode
        case .function, .private, .workgroup: available = [.load, .store]
        case .uniform, .immediate: available = .load
        default: available = []
        }
        guard available.contains(access) else {
            throw wgslError(.invalidAddressSpace, "Cooperative matrix pointer does not permit this memory operation", name)
        }
        let strideValue: WGSLValue
        if let stride {
            switch types[stride.type.index].kind {
            case .scalar(.abstractInt): strideValue = try coerce(stride, to: intern(.scalar(.u32)), at: name)
            case .scalar(.u32), .scalar(.i32): strideValue = stride
            default: throw wgslError(.typeMismatch, "Cooperative matrix stride must be a scalar integer", name)
            }
        } else {
            strideValue = try expressionNode(.literal(.u32(UInt32(defaultStride.rawValue))),
                                             type: intern(.scalar(.u32)), at: name, emit: false)
        }
        return .init(pointer: pointer.handle, stride: strideValue.handle, rowMajor: rowMajor)
    }
}
