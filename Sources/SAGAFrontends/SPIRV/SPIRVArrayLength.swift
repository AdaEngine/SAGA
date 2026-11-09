import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerSPVArrayLength(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.count(4)
        let resultKind = try module.kind(i.a[0], i)
        guard resultKind == .scalar(.u32) || resultKind == .scalar(.u64) else {
            throw i.fail("OpArrayLength requires a u32 or u64 result")
        }
        let source = try value(i.a[2], i, &statements)
        let actualType = expressions[source.expression.index].type
        let index = i.a[3]
        guard case .pointer(let structure, let space) = module.types[actualType.index].kind, case .storage = space,
              case .structure(let members, _) = module.types[structure.index].kind,
              let array = members.last?.type, index == members.count - 1,
              case .array(_, .dynamic, _) = module.types[array.index].kind else {
            throw i.fail("OpArrayLength requires the final runtime array of a storage block")
        }
        try checkBufferDescriptorNonUniform([i.a[2]], through: source.expression, at: i)
        let pointerType = try internLoweredType(.pointer(base: array, space: space))
        let pointer = try emitValue(pointerType, .accessIndex(base: source.expression, index: index), &statements, i)
        let length = IRExpressionKind.arrayLength(pointer)
        if resultKind == .scalar(.u32) {
            try result(i, length, &statements)
        } else {
            let word = try internLoweredType(.scalar(.u32))
            let value = try emitValue(word, length, &statements, i)
            try result(i, .cast(expression: value, target: .u64), &statements)
        }
    }
}
