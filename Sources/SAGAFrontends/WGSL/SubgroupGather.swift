import SAGAShaderIR

extension WGSLParser {
    static let subgroupGatherBuiltins: Set<String> = [
        "subgroupBroadcastFirst", "subgroupBroadcast", "subgroupShuffle",
        "subgroupShuffleDown", "subgroupShuffleUp", "subgroupShuffleXor",
        "quadBroadcast", "quadSwapX", "quadSwapY", "quadSwapDiagonal",
    ]

    func requireSubgroups(at token: WGSLToken) throws {
        guard subgroupsEnabled else {
            throw wgslError(.unsupportedCapability, "Subgroup operations require enable subgroups", token)
        }
    }

    func subgroupGather(at name: WGSLToken) throws -> WGSLValue {
        try requireSubgroups(at: name)
        guard !(try localConstantNameExists(name)) else {
            throw wgslError(.typeMismatch, "Local value is not callable", name)
        }
        let argument = try defaultMaterialize(loaded(expression(), at: name), at: name)
        switch types[argument.type.index].kind {
        case .scalar(let scalar), .vector(_, let scalar):
            guard [.i32, .u32, .f32, .f16].contains(scalar) else {
                throw wgslError(.typeMismatch, "Subgroup gather requires a concrete numeric scalar or vector", name)
            }
        default:
            throw wgslError(.typeMismatch, "Subgroup gather requires a concrete numeric scalar or vector", name)
        }
        let mode: IRGatherMode
        switch name.text {
        case "subgroupBroadcastFirst":
            mode = .broadcastFirst
        case "quadSwapX":
            mode = .quadSwap(.x)
        case "quadSwapY":
            mode = .quadSwap(.y)
        case "quadSwapDiagonal":
            mode = .quadSwap(.diagonal)
        default:
            try expect(",")
            let start = cursor
            let selector = try loaded(expression(), at: name)
            let constant = try knownIntegerConstant(start..<cursor, at: name)
            let indexLimit: Int64 = name.text == "quadBroadcast" ? 4 : 128
            if let constant, !(0..<indexLimit).contains(constant.value) {
                throw wgslError(.invalidType, "Constant subgroup selector is outside 0..<\(indexLimit)", name)
            }
            switch name.text {
            case "subgroupBroadcast", "quadBroadcast":
                guard let constant else {
                    throw wgslError(.invalidType, "Broadcast index must be a constant expression", name)
                }
                let index = try expressionNode(.literal(.u32(UInt32(constant.value))), type: intern(.scalar(.u32)),
                                               at: name, emit: false)
                mode = name.text == "quadBroadcast" ? .quadBroadcast(index.handle) : .broadcast(index.handle)
            case "subgroupShuffle":
                let index = try defaultMaterialize(selector, at: name)
                guard [.scalar(.i32), .scalar(.u32)].contains(types[index.type.index].kind) else {
                    throw wgslError(.typeMismatch, "Subgroup shuffle index must be i32 or u32", name)
                }
                mode = .shuffle(index.handle)
            default:
                let index = try coerce(selector, to: intern(.scalar(.u32)), at: name)
                switch name.text {
                case "subgroupShuffleDown": mode = .shuffleDown(index.handle)
                case "subgroupShuffleUp": mode = .shuffleUp(index.handle)
                case "subgroupShuffleXor": mode = .shuffleXor(index.handle)
                default: throw ShaderFailure.invalidHandle
                }
            }
        }
        _ = try eat(",")
        try expect(")")
        let result = try expressionNode(.subgroupOperationResult(type: argument.type), type: argument.type, at: name, emit: false)
        try statement(.subgroupGather(mode: mode, argument: argument.handle, result: result.handle), at: name)
        return result
    }
}
